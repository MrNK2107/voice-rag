import os
import subprocess
import sqlite3
import sys
import threading
import traceback
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from gradio import Server
import spaces

from app.config import settings
from app.harness import VoiceRAGHarness
from app.schemas import RagResponse, TextRequest

app = Server()
harness: VoiceRAGHarness | None = None
_startup_log = []
_build_started = False


@spaces.GPU
def _gpu_placeholder():
    pass


def _log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    entry = f"[{ts}] {msg}"
    _startup_log.append(entry)
    print(entry, flush=True)


@app.on_event("startup")
def startup_event():
    _log(f"Python: {sys.executable}")
    _log(f"CWD: {os.getcwd()}")
    _log(f"QDRANT_PATH={settings.qdrant_path}")
    _log(f"SQLITE_FTS_PATH={settings.sqlite_fts_path}")
    _log(f"QDRANT_URL={settings.qdrant_url}")
    _log(f"HF_TOKEN={'set' if os.environ.get('HF_TOKEN') else 'NOT SET'}")
    _log(f"SARVAM_API_KEY={'set' if os.environ.get('SARVAM_API_KEY') else 'NOT SET'}")
    _log(f"Files in /app: {os.listdir('.')}")

    Path(settings.sqlite_fts_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)

    has_data = False
    if os.path.exists(settings.sqlite_fts_path):
        try:
            conn = sqlite3.connect(settings.sqlite_fts_path)
            count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
            conn.close()
            has_data = count > 0
            _log(f"SQLite has {count} chunks")
        except Exception as e:
            _log(f"SQLite check error: {e}")

    if not has_data:
        _log("No index data. Starting background build...")
        global _build_started
        _build_started = True
        threading.Thread(target=_build_and_init, daemon=True).start()
    else:
        _log("Index exists. Initializing harness...")
        _init_harness()


def _build_and_init():
    try:
        _log("Running build_index.py...")
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        result = subprocess.run(
            [sys.executable, "-u", "scripts/build_index.py", "--languages", "hin", "--max-rows", "500"],
            capture_output=True, text=True, timeout=1200, env=env,
        )
        _log(f"build_index exit code: {result.returncode}")
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-40:]:
                _log(f"  {line}")
        if result.stderr:
            for line in result.stderr.strip().split("\n")[-40:]:
                _log(f"ERR: {line}")
    except subprocess.TimeoutExpired:
        _log("build_index TIMED OUT (1200s)")
    except Exception as e:
        _log(f"build_index EXCEPTION: {e}")
        traceback.print_exc()

    _init_harness()


def _init_harness():
    global harness
    _log("Creating VoiceRAGHarness...")
    try:
        harness = VoiceRAGHarness()
        _log("Harness READY!")
    except Exception as e:
        _log(f"Harness FAILED: {e}")
        traceback.print_exc()


@app.get("/health")
def health_check():
    return {"status": "ok", "harness_loaded": harness is not None}


@app.get("/status")
def status():
    qdrant_exists = os.path.exists(settings.qdrant_path)
    sqlite_exists = os.path.exists(settings.sqlite_fts_path)
    sqlite_size = os.path.getsize(settings.sqlite_fts_path) if sqlite_exists else 0
    return {
        "harness_loaded": harness is not None,
        "build_started": _build_started,
        "qdrant_path_exists": qdrant_exists,
        "sqlite_exists": sqlite_exists,
        "sqlite_size_bytes": sqlite_size,
        "log": _startup_log[-50:],
    }


@app.post("/api/ask-text", response_model=RagResponse)
def ask_text(req: TextRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness loading.")
    return harness.ask_text(req.query)


@app.post("/api/ask-audio", response_model=RagResponse)
async def ask_audio(file: UploadFile = File(...)):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness loading.")
    audio_bytes = await file.read()
    return harness.ask_audio(
        audio_bytes=audio_bytes,
        filename=file.filename or "audio.webm",
        content_type=file.content_type or "audio/webm",
    )


@app.get("/")
def serve_ui():
    if Path("web/index.html").exists():
        return FileResponse("web/index.html")
    return {"message": "Voice RAG API is running."}


app.launch()
