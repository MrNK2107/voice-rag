import os
import subprocess
import sqlite3
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from gradio import Server
import spaces

from app.config import settings
from app.harness import VoiceRAGHarness
from app.schemas import RagResponse, TextRequest

app = Server()
harness: VoiceRAGHarness | None = None
_index_ready = threading.Event()


@spaces.GPU
def _gpu_placeholder():
    pass


def _ensure_sqlite_tables():
    Path(settings.sqlite_fts_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(settings.sqlite_fts_path)
    cur = conn.cursor()
    cur.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
    USING fts5(chunk_id UNINDEXED, text, title, language, strategy, tokenize='unicode61')
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chunks_meta (
        chunk_id TEXT PRIMARY KEY, text TEXT, title TEXT,
        language TEXT, strategy TEXT, parent_doc_id TEXT
    )
    """)
    conn.commit()
    conn.close()


def _needs_index_build():
    if not os.path.exists(settings.sqlite_fts_path):
        return True
    try:
        conn = sqlite3.connect(settings.sqlite_fts_path)
        count = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0]
        conn.close()
        return count == 0
    except Exception:
        return True


def _build_index_background():
    global harness
    try:
        print("[BUILD] Starting index build in background...")
        result = subprocess.run(
            [sys.executable, "scripts/build_index.py", "--languages", "hin", "--max-rows", "500"],
            capture_output=True, text=True, timeout=1200,
        )
        print(f"[BUILD] Exit code: {result.returncode}")
        if result.stdout:
            for line in result.stdout.strip().split("\n")[-30:]:
                print(f"[BUILD] {line}")
        if result.returncode != 0 and result.stderr:
            for line in result.stderr.strip().split("\n")[-30:]:
                print(f"[BUILD ERR] {line}")
    except subprocess.TimeoutExpired:
        print("[BUILD] Index build timed out after 1200s")
    except Exception as e:
        print(f"[BUILD] Exception: {e}")

    print("[BUILD] Index build done. Initializing harness...")
    try:
        harness = VoiceRAGHarness()
        _index_ready.set()
        print("[BUILD] Harness initialized successfully.")
    except Exception as e:
        print(f"[BUILD] Harness init FAILED: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("startup")
def startup_event():
    print(f"[STARTUP] QDRANT_PATH={settings.qdrant_path}")
    print(f"[STARTUP] SQLITE_FTS_PATH={settings.sqlite_fts_path}")
    print(f"[STARTUP] QDRANT_URL={settings.qdrant_url}")

    _ensure_sqlite_tables()

    if _needs_index_build():
        print("[STARTUP] Index empty. Kicking off background build...")
        threading.Thread(target=_build_index_background, daemon=True).start()
    else:
        print("[STARTUP] Index exists. Initializing harness...")
        try:
            harness = VoiceRAGHarness()
            _index_ready.set()
            print("[STARTUP] Harness initialized.")
        except Exception as e:
            print(f"[STARTUP] Harness init FAILED: {e}")
            import traceback
            traceback.print_exc()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "harness_loaded": harness is not None,
    }


@app.post("/api/ask-text", response_model=RagResponse)
def ask_text(req: TextRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized yet. Index is building in background, try again in a few minutes.")
    return harness.ask_text(req.query)


@app.post("/api/ask-audio", response_model=RagResponse)
async def ask_audio(file: UploadFile = File(...)):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized yet. Index is building in background, try again in a few minutes.")

    audio_bytes = await file.read()
    filename = file.filename or "audio.webm"
    content_type = file.content_type or "audio/webm"

    return harness.ask_audio(
        audio_bytes=audio_bytes,
        filename=filename,
        content_type=content_type,
    )


@app.get("/")
def serve_ui():
    if Path("web/index.html").exists():
        return FileResponse("web/index.html")
    return {"message": "Voice RAG API is running."}


app.launch()
