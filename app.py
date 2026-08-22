import os
import subprocess
import sqlite3
import sys
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


@app.on_event("startup")
def startup_event():
    global harness

    print(f"[STARTUP] QDRANT_PATH={settings.qdrant_path}")
    print(f"[STARTUP] SQLITE_FTS_PATH={settings.sqlite_fts_path}")
    print(f"[STARTUP] QDRANT_URL={settings.qdrant_url}")

    _ensure_sqlite_tables()

    if _needs_index_build():
        print("[STARTUP] Index empty or missing. Building from HF dataset (~5-10 min)...")
        try:
            result = subprocess.run(
                [sys.executable, "scripts/build_index.py", "--languages", "hin", "--max-rows", "500"],
                capture_output=True, text=True, timeout=900,
            )
            print(f"[STARTUP] build_index exit code: {result.returncode}")
            if result.stdout:
                lines = result.stdout.strip().split("\n")
                print(f"[STARTUP] stdout (last 20 lines):\n" + "\n".join(lines[-20:]))
            if result.returncode != 0 and result.stderr:
                lines = result.stderr.strip().split("\n")
                print(f"[STARTUP] stderr (last 20 lines):\n" + "\n".join(lines[-20:]))
        except subprocess.TimeoutExpired:
            print("[STARTUP] Index build timed out after 900s")
        except Exception as e:
            print(f"[STARTUP] Index build exception: {e}")
    else:
        print("[STARTUP] Index already has data, skipping build.")

    print("[STARTUP] Initializing VoiceRAGHarness...")
    try:
        harness = VoiceRAGHarness()
        print("[STARTUP] VoiceRAGHarness initialized successfully.")
    except Exception as e:
        print(f"[STARTUP] VoiceRAGHarness init FAILED: {e}")
        import traceback
        traceback.print_exc()
        harness = None


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "harness_loaded": harness is not None,
    }


@app.post("/api/ask-text", response_model=RagResponse)
def ask_text(req: TextRequest):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")
    return harness.ask_text(req.query)


@app.post("/api/ask-audio", response_model=RagResponse)
async def ask_audio(file: UploadFile = File(...)):
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")

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
