"""Hugging Face Spaces entry point (``python app.py``, port 7860).

Serves the same API and web app as ``app.main`` (landing page, /app, /admin,
widget, collections, insights) on a Gradio ``Server``.

* ``VOUCH_DEMO=1`` (or ``VOICE_RAG_DEMO=1``): offline sample collections, ready
  in seconds, no model download. Recommended for a public demo Space.
* Otherwise: the multilingual e5 embedder plus Qdrant embedded storage. If no
  legacy MSMARCO-XI index exists, a small one is built in the background
  (``scripts/build_index.py``), and user collections work as soon as the
  embedder has loaded.
"""
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("FORCE_CUDA", "0")

import sqlite3
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import app.main as vouch  # noqa: E402
from app.config import settings  # noqa: E402

_log_entries = []


def _log(msg):
    entry = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    _log_entries.append(entry)
    print(entry, flush=True)


def create_server():
    from fastapi.staticfiles import StaticFiles
    from gradio import Server

    server = Server()
    server.include_router(vouch.app.router)
    if os.path.exists("web"):
        server.mount("/static", StaticFiles(directory="web"), name="static")

    @server.get("/status")
    def status():
        return {
            "harness_loaded": vouch.harness is not None,
            "demo_mode": vouch.demo_mode_enabled(),
            "sqlite_exists": os.path.exists(settings.sqlite_fts_path),
            "log": _log_entries[-50:],
        }

    return server


def _init_harness():
    try:
        vouch.harness = vouch.build_production_harness()
        _log("Harness READY")
    except Exception as e:
        _log(f"Harness FAILED: {e}")
        traceback.print_exc()


def _build_legacy_index_then_init():
    try:
        _log("No legacy index: running build_index.py (Hindi, 500 rows)...")
        result = subprocess.run(
            [sys.executable, "-u", "scripts/build_index.py", "--languages", "hin", "--max-rows", "500"],
            capture_output=True, text=True, timeout=1200,
        )
        _log(f"build_index exit code: {result.returncode}")
        for line in (result.stdout or "").strip().split("\n")[-10:]:
            _log(f"  {line}")
    except Exception as e:
        _log(f"build_index EXCEPTION: {e}")
    _init_harness()


def init():
    if vouch.demo_mode_enabled():
        from app.demo import build_demo_harness

        vouch.harness = build_demo_harness()
        _log("Demo harness READY (offline sample collections)")
        return
    Path(settings.sqlite_fts_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    has_data = False
    if os.path.exists(settings.sqlite_fts_path):
        try:
            conn = sqlite3.connect(settings.sqlite_fts_path)
            has_data = conn.execute("SELECT COUNT(*) FROM chunks_meta").fetchone()[0] > 0
            conn.close()
        except Exception as e:
            _log(f"SQLite check: {e}")
    if has_data or os.environ.get("VOUCH_SKIP_LEGACY_INDEX"):
        threading.Thread(target=_init_harness, daemon=True).start()
    else:
        threading.Thread(target=_build_legacy_index_then_init, daemon=True).start()


if __name__ == "__main__":
    try:  # ZeroGPU Spaces need one @spaces.GPU function to exist; CPU Spaces don't.
        import spaces

        @spaces.GPU
        def _gpu_placeholder():
            pass
    except ImportError:
        pass
    server = create_server()
    init()
    server.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
