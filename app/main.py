"""Vouch API and web app.

    python -m uvicorn app.main:app --port 8000        # your documents (e5 + Qdrant)
    VOICE_RAG_DEMO=1 python -m uvicorn app.main:app   # offline, sample collections

Pages: /  (landing), /app (ask), /admin (collections + insights),
/embed/{collection} (widget frame), /widget.js (the embeddable script).
"""
import hmac
import os
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.agent import AgentRequest, AgentResponse, HealthAgent, planner_from_env
from app.collection_store import CollectionError, CollectionNotFound
from app.config import settings
from app.harness import VoiceRAGHarness
from app.ingest import IngestError
from app.langid import LANGUAGE_NAMES, detect_language
from app.schemas import RagResponse, TextRequest
from app.stt import STTUnavailable, available_providers, choose_provider, make_stt
from app.stt_sarvam import SarvamSTT

harness: VoiceRAGHarness | None = None
_HarnessClass = VoiceRAGHarness
WEB = "web"
# A spoken question is seconds of audio; 25 MB is also the Whisper API's limit.
MAX_AUDIO_BYTES = 25 * 1024 * 1024


def demo_mode_enabled() -> bool:
    """VOICE_RAG_DEMO=1 (or VOUCH_DEMO=1) serves the bundled offline sample
    collections (no API key, no Qdrant server, no model download). See app/demo.py."""
    return any(os.environ.get(k, "").strip().lower() in {"1", "true", "yes", "on"}
               for k in ("VOICE_RAG_DEMO", "VOUCH_DEMO"))


def build_production_harness() -> VoiceRAGHarness:
    """Multilingual e5 embedder, Qdrant (server or embedded) for the legacy
    index, and a persistent CollectionStore for the user's own collections."""
    from qdrant_client import QdrantClient

    from app.analytics import QuestionLog
    from app.collection_store import CollectionStore
    from app.retriever import HybridRetriever
    from app.translate import translator_from_settings

    retriever = HybridRetriever()
    root = settings.collections_dir
    os.makedirs(root, exist_ok=True)
    store = CollectionStore(retriever.model, root, qdrant=QdrantClient(path=os.path.join(root, "qdrant")))
    log = QuestionLog(settings.analytics_path) if settings.analytics_enabled else None
    return VoiceRAGHarness(retriever=retriever, collections=store, translator=translator_from_settings(),
                           question_log=log)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global harness
    if demo_mode_enabled():
        from app.demo import build_demo_harness

        print("Demo mode: building offline sample collections...")
        harness = build_demo_harness()
    else:
        print("Initializing harness and preloading the embedding model...")
        if VoiceRAGHarness is _HarnessClass:
            harness = build_production_harness()
        else:  # tests swap in a factory for a prepared offline harness
            harness = VoiceRAGHarness()
    print("Harness ready.")
    try:
        yield
    finally:
        harness = None


app = FastAPI(
    title="Vouch: grounded, multilingual voice answers from your documents",
    description="Ask out loud in any language; get cited answers from your own collections, or an honest abstention.",
    version="2.0.0",
    lifespan=lifespan,
)


def _need_harness() -> VoiceRAGHarness:
    if not harness:
        raise HTTPException(status_code=503, detail="Harness not initialized")
    return harness


def _need_store():
    h = _need_harness()
    if h.collections is None:
        raise HTTPException(status_code=501, detail="Collections are not enabled on this server.")
    return h.collections


def _is_admin(token: Optional[str]) -> bool:
    if not settings.admin_token:
        return True
    return hmac.compare_digest((token or "").encode("utf-8"), settings.admin_token.encode("utf-8"))


def _check_admin(token: Optional[str]) -> None:
    if not _is_admin(token):
        raise HTTPException(status_code=401, detail="Admin token required (X-Admin-Token header).")


# ---- health / config ----------------------------------------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "harness_loaded": harness is not None,
        "demo_mode": demo_mode_enabled(),
    }


@app.get("/api/config")
def config():
    h = harness
    store = h.collections if h else None
    examples = {}
    if demo_mode_enabled():
        from app.demo import demo_payload

        examples = demo_payload().get("collections", {})
    return {
        "product": settings.product_name,
        "demo_mode": demo_mode_enabled(),
        "stt_providers": available_providers(),
        "stt_default": choose_provider(),
        "translation": bool(h and h.translator),
        "collections_enabled": store is not None,
        "admin_token_required": bool(settings.admin_token),
        "languages": LANGUAGE_NAMES,
        "examples": examples,
    }


@app.get("/api/demo")
def demo_info():
    """Scripted demo questions + corpus sources (only populated in demo mode)."""
    if not demo_mode_enabled():
        return {"enabled": False, "script": [], "sources": {}}
    from app.demo import demo_payload

    return {"enabled": True, **demo_payload()}


class DetectRequest(BaseModel):
    text: str = Field(max_length=1_000_000)


@app.post("/api/detect-language")
def detect(req: DetectRequest):
    return detect_language(req.text).dict()


# ---- asking -------------------------------------------------------------------
@app.post("/api/ask-text", response_model=RagResponse)
def ask_text(req: TextRequest):
    h = _need_harness()
    try:
        return h.ask_text(req.query, collection=req.collection or None, language=req.language,
                          channel=req.channel if req.channel in {"text", "voice", "widget"} else "text")
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{req.collection}'.")


@app.post("/api/agent", response_model=AgentResponse)
def agent_ask(req: AgentRequest):
    """Health demo's tool-using agent loop (triage -> clarify / facility lookup /
    cited retrieval) with a step trace."""
    h = _need_harness()
    return HealthAgent(h, planner=planner_from_env()).run(req.query, req.slots)


@app.post("/api/ask-audio", response_model=RagResponse)
async def ask_audio(file: UploadFile = File(...), collection: Optional[str] = Form(None),
                    language: Optional[str] = Form(None), stt: Optional[str] = Form(None)):
    h = _need_harness()
    audio_bytes = await file.read(MAX_AUDIO_BYTES + 1)
    if len(audio_bytes) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail=f"Audio is larger than {MAX_AUDIO_BYTES // (1024 * 1024)} MB.")
    provider = None
    # Offline / test harnesses carry their own STT; production picks a provider
    # per request (Sarvam for Indian languages, Whisper for the rest).
    if (stt and stt != "auto") or isinstance(h.stt, SarvamSTT):
        try:
            provider = make_stt(language, stt)
        except STTUnavailable as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        return h.ask_audio(audio_bytes=audio_bytes, filename=file.filename or "audio.webm",
                           content_type=file.content_type or "audio/webm",
                           collection=collection or None, language=language, stt=provider)
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{collection}'.")


# ---- collections --------------------------------------------------------------
class CollectionCreate(BaseModel):
    name: str
    title: str = ""
    description: str = ""


class TextIngest(BaseModel):
    text: str
    title: str = "Pasted text"


class UrlIngest(BaseModel):
    url: str


@app.get("/api/collections")
def list_collections():
    return {"collections": _need_store().list()}


@app.post("/api/collections", status_code=201)
def create_collection(req: CollectionCreate, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    try:
        return _need_store().create(req.name, req.title, req.description)
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/collections/{name}")
def get_collection(name: str, x_admin_token: Optional[str] = Header(None)):
    store = _need_store()
    try:
        summary = store.summary(name)
        # Summaries are public (the widget shows the title); with an admin token
        # configured, the document list is for admins only.
        admin = _is_admin(x_admin_token)
        return {**summary, "document_list": store.documents(name) if admin else []}
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{name}'.")


@app.delete("/api/collections/{name}")
def delete_collection(name: str, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    try:
        _need_store().delete(name)
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{name}'.")
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": name}


@app.delete("/api/collections/{name}/documents/{doc_id}")
def delete_document(name: str, doc_id: str, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    try:
        rec = _need_store().delete_document(name, doc_id)
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail="Unknown collection or document.")
    except CollectionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"deleted": rec, "collection": _need_store().summary(name)}


def _ingest(fn, name: str, *args, **kw):
    try:
        added = fn(name, *args, **kw)
    except CollectionNotFound:
        raise HTTPException(status_code=404, detail=f"Unknown collection '{name}'.")
    except (CollectionError, IngestError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"added": added, "collection": _need_store().summary(name)}


@app.post("/api/collections/{name}/files")
async def upload_files(name: str, files: List[UploadFile] = File(...), x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    store = _need_store()
    limit = settings.max_upload_mb * 1024 * 1024
    added, errors = [], []
    for f in files:
        data = await f.read(limit + 1)
        if len(data) > limit:
            errors.append({"file": f.filename, "error": f"Larger than {settings.max_upload_mb} MB."})
            continue
        try:
            added += store.add_file(name, f.filename or "upload.txt", data)
        except CollectionNotFound:
            raise HTTPException(status_code=404, detail=f"Unknown collection '{name}'.")
        except (CollectionError, IngestError) as e:
            errors.append({"file": f.filename, "error": str(e)})
    if not added and errors:
        return JSONResponse(status_code=400, content={"added": [], "errors": errors})
    return {"added": added, "errors": errors, "collection": store.summary(name)}


@app.post("/api/collections/{name}/text")
def ingest_text(name: str, req: TextIngest, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text is empty.")
    return _ingest(_need_store().add_text, name, req.text, req.title)


@app.post("/api/collections/{name}/url")
def ingest_url(name: str, req: UrlIngest, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    return _ingest(_need_store().add_url, name, req.url)


# ---- insights -----------------------------------------------------------------
@app.get("/api/insights")
def insights(collection: Optional[str] = None, days: int = 30, x_admin_token: Optional[str] = Header(None)):
    _check_admin(x_admin_token)
    h = _need_harness()
    if h.question_log is None:
        return {"enabled": False}
    return {"enabled": True, **h.question_log.summary(collection or None, days=max(1, min(days, 365)))}


# ---- web ----------------------------------------------------------------------
if os.path.exists(WEB):
    app.mount("/static", StaticFiles(directory=WEB), name="static")


def _page(name: str):
    path = os.path.join(WEB, name)
    if os.path.exists(path):
        return FileResponse(path)
    return {"message": "Vouch API is running. Web UI files not found."}


@app.get("/")
def serve_landing():
    return _page("index.html")


@app.get("/app")
def serve_app():
    return _page("app.html")


@app.get("/admin")
def serve_admin():
    return _page("admin.html")


@app.get("/embed/{collection}")
def serve_embed(collection: str):
    return _page("embed.html")


@app.get("/demo/health")
def serve_health_demo():
    """The original Swasthya Saathi demo page (agent trace, scripted chips)."""
    return _page("health-demo.html")


@app.get("/widget.js")
def serve_widget():
    path = os.path.join(WEB, "widget.js")
    if not os.path.exists(path):
        raise HTTPException(status_code=404)
    return FileResponse(path, media_type="application/javascript")
