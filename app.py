import os
import subprocess
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


@app.on_event("startup")
def startup_event():
    global harness

    if not os.path.exists(settings.sqlite_fts_path):
        print(f"[STARTUP] Index not found at {settings.sqlite_fts_path}. Building...")
        result = subprocess.run(
            [sys.executable, "scripts/build_index.py", "--languages", "hin", "--max-rows", "500"],
            capture_output=True, text=True, timeout=900,
        )
        print(f"[STARTUP] build_index.py stdout:\n{result.stdout[-2000:]}")
        if result.returncode != 0:
            print(f"[STARTUP] build_index.py FAILED (rc={result.returncode}):\n{result.stderr[-2000:]}")
        else:
            print("[STARTUP] Index build complete.")
    else:
        print(f"[STARTUP] Index found at {settings.sqlite_fts_path}, skipping build.")

    print("[STARTUP] Initializing VoiceRAGHarness...")
    harness = VoiceRAGHarness()
    print("[STARTUP] VoiceRAGHarness initialized successfully.")


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
