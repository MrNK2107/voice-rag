"""Collections (ingestion + retrieval), STT provider selection, insights and
the product API/pages. All offline."""
import io
import time

import pytest
from qdrant_client import QdrantClient

from app.analytics import QuestionLog
from app.collection_store import CollectionError, CollectionNotFound, CollectionStore
from app.config import settings
from app.ingest import IngestError, extract_file, fetch_url, html_to_text, markdown_to_text
from app.offline import FakeSTT, HashingEmbedder, build_offline_harness


@pytest.fixture
def store(tmp_path):
    return CollectionStore(HashingEmbedder(dim=1024), tmp_path / "collections", qdrant=QdrantClient(":memory:"))


SPANISH_DOC = ("El ayuntamiento recoge los muebles viejos el primer lunes de cada mes. "
               "Para solicitar la recogida de muebles llame al 555-0199 antes del viernes anterior.")


# ---- store ---------------------------------------------------------------------
def test_create_ingest_and_answer_from_new_collection(store, tmp_path):
    store.create("my-docs", "Mis documentos", "prueba")
    added = store.add_text("my-docs", SPANISH_DOC, title="Recogida de muebles")
    assert added and added[0]["language"] == "es" and added[0]["chunks"] >= 1
    s = store.summary("my-docs")
    assert s["documents"] == 1 and s["languages"] == {"es": 1} and not s["read_only"]

    h = build_offline_harness(tmp_path / "legacy.sqlite", stt=FakeSTT())
    h.collections = store
    r = h.ask_text("¿Cuándo recogen los muebles viejos?", collection="my-docs")
    assert not r.abstained, r.abstain_reason
    assert "muebles" in r.answer and r.language == "es" and r.collection == "my-docs"
    assert r.citations[0].title == "Recogida de muebles" and r.citations[0].source == "pasted"


def test_collection_name_rules_and_duplicates(store):
    with pytest.raises(CollectionError):
        store.create("Bad Name!")
    store.create("ok-1")
    with pytest.raises(CollectionError):
        store.create("ok-1")


def test_delete_collection_and_metadata_persists(store, tmp_path):
    store.create("keep")
    store.add_text("keep", "The office is open from nine to five on weekdays.", title="Hours")
    store.create("gone")
    store.delete("gone")
    with pytest.raises(CollectionNotFound):
        store.get("gone")
    again = CollectionStore(HashingEmbedder(dim=1024), tmp_path / "collections", qdrant=QdrantClient(":memory:"))
    assert [c["name"] for c in again.list()] == ["keep"]


def test_registered_collection_is_read_only(store, tmp_path):
    h = build_offline_harness(tmp_path / "legacy.sqlite")
    store.register_existing("legacy", "Legacy", h.retriever.qdrant, settings.qdrant_collection,
                            str(tmp_path / "legacy.sqlite"))
    with pytest.raises(CollectionError):
        store.add_text("legacy", "some text that is long enough to be a chunk")
    with pytest.raises(CollectionError):
        store.delete("legacy")
    assert store.retriever("legacy").retrieve("मैकडॉनल्ड्स क्या है?")[0]


def test_chinese_document_chunks_and_retrieves(store, tmp_path):
    store.create("zh")
    store.add_text("zh", "图书馆每天上午九点开门，晚上八点关门。周一闭馆进行图书整理。", title="开放时间")
    text = "借书证可以在一楼服务台免费办理，需要带身份证。每张借书证最多可以同时借十本书。"
    added = store.add_text("zh", text, title="图书馆")
    assert added[0]["language"] == "zh" and added[0]["chunks"] >= 1
    h = build_offline_harness(tmp_path / "legacy.sqlite")
    h.collections = store
    r = h.ask_text("借书证在哪里办理？", collection="zh")
    assert not r.abstained, r.abstain_reason
    assert "借书证" in r.answer


def test_cjk_chunker_groups_sentences_without_inserting_spaces():
    from app.chunking import RawDoc, make_chunks

    text = "".join(f"第{i}条规定适用于所有居民和访客。" for i in range(60))
    chunks = make_chunks(RawDoc("d", text, language="zh"))
    assert len(chunks) >= 3
    assert all(c.payload["chunk_strategy"] == "char_group_300c" and " " not in c.text for c in chunks)
    assert all(len(c.text) <= 320 for c in chunks)


# ---- extraction ----------------------------------------------------------------
def test_markdown_and_html_extraction():
    assert markdown_to_text("# Title\n\nSee [the docs](http://x) and **bold**.").strip().startswith("Title.")
    title, text = html_to_text("<html><head><title>T</title><script>evil()</script></head>"
                               "<body><nav>Menu</nav><p>Hello world.</p><p>Second para</p></body></html>")
    assert title == "T" and "evil" not in text and "Menu" not in text
    assert "Hello world." in text and "Second para." in text


def test_extract_file_types():
    assert extract_file("notes.md", b"# Heading\nBody text here.")[0].title == "notes"
    assert "Body" in extract_file("a.txt", "Body ü".encode("utf-8"))[0].text
    with pytest.raises(IngestError):
        extract_file("image.png", b"\x89PNG")
    with pytest.raises(IngestError):
        extract_file("empty.txt", b"   ")


def test_pdf_extraction_one_document_per_page():
    pypdf = pytest.importorskip("pypdf")
    w = pypdf.PdfWriter()
    w.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    w.write(buf)
    with pytest.raises(IngestError, match="no extractable text"):
        extract_file("scan.pdf", buf.getvalue())
    with pytest.raises(IngestError):
        extract_file("broken.pdf", b"%PDF-1.4 not really")


class _Resp:
    def __init__(self, status=200, body=b"", headers=None):
        self.status_code = status
        self._body = body
        self.headers = headers or {"content-type": "text/html; charset=utf-8"}

    def iter_content(self, n):
        yield self._body


def test_fetch_url_with_injected_client_and_redirect():
    calls = []

    def get(url, **kw):
        calls.append(url)
        if url.endswith("/old"):
            return _Resp(301, headers={"location": "/new"})
        return _Resp(body=b"<title>FAQ</title><p>Bins go out on Tuesday.</p>")

    docs = fetch_url("https://example.org/old", get=get, check_host=False)
    assert calls == ["https://example.org/old", "https://example.org/new"]
    assert docs[0].title == "FAQ" and docs[0].source == "https://example.org/new"


@pytest.mark.parametrize("url", ["http://127.0.0.1/admin", "http://localhost:8000/", "http://10.0.0.5/x",
                                 "http://169.254.169.254/latest/meta-data", "file:///etc/passwd", "ftp://x.org/a"])
def test_fetch_url_refuses_local_and_non_http(url):
    with pytest.raises(IngestError):
        fetch_url(url, get=lambda *a, **k: _Resp(body=b"x"))


# ---- STT provider selection ----------------------------------------------------
def _stt_settings(monkeypatch, sarvam="", whisper_base="", whisper_key="", provider="auto"):
    monkeypatch.setattr(settings, "sarvam_api_key", sarvam)
    monkeypatch.setattr(settings, "whisper_api_base", whisper_base)
    monkeypatch.setattr(settings, "whisper_api_key", whisper_key)
    monkeypatch.setattr(settings, "stt_provider", provider)
    import app.stt as stt

    monkeypatch.setattr(stt.FasterWhisperSTT, "available", staticmethod(lambda: False))
    return stt


def test_stt_nothing_configured_falls_back_to_browser(monkeypatch):
    stt = _stt_settings(monkeypatch)
    assert stt.available_providers() == ["browser"]
    assert stt.choose_provider("es") == "browser"
    with pytest.raises(stt.STTUnavailable):
        stt.make_stt("es")


def test_stt_indian_language_prefers_sarvam_others_whisper(monkeypatch):
    stt = _stt_settings(monkeypatch, sarvam="k", whisper_base="https://w.example/v1", whisper_key="w")
    assert stt.choose_provider("ta") == "sarvam"
    assert stt.choose_provider("hi-Latn") == "sarvam"
    assert stt.choose_provider("es") == "whisper_api"
    assert stt.choose_provider("auto") == "whisper_api"
    assert stt.choose_provider(None) == "whisper_api"
    assert type(stt.make_stt("ar")).__name__ == "WhisperAPISTT"


def test_stt_sarvam_only_and_forced_provider(monkeypatch):
    stt = _stt_settings(monkeypatch, sarvam="k")
    assert stt.choose_provider("fr") == "sarvam"
    with pytest.raises(stt.STTUnavailable):
        stt.choose_provider("fr", preferred="whisper_api")
    stt2 = _stt_settings(monkeypatch, sarvam="k", provider="sarvam")
    assert stt2.choose_provider("es") == "sarvam"


def test_stt_local_faster_whisper_when_installed(monkeypatch):
    stt = _stt_settings(monkeypatch)
    monkeypatch.setattr(stt.FasterWhisperSTT, "available", staticmethod(lambda: True))
    assert stt.choose_provider("de") == "faster_whisper"


def test_whisper_api_request_and_language(monkeypatch):
    import app.stt as stt

    seen = {}

    class R:
        def raise_for_status(self):
            pass

        def json(self):
            return {"text": " ¿Dónde está la biblioteca? ", "language": "spanish"}

    def post(url, headers=None, files=None, data=None, timeout=None):
        seen.update(url=url, headers=headers, data=data)
        return R()

    monkeypatch.setattr(stt.requests, "post", post)
    w = stt.WhisperAPISTT(base_url="https://w.example/v1/", api_key="secret", model="whisper-1", language_hint="auto")
    assert w.transcribe(b"abc", "a.webm", "audio/webm") == "¿Dónde está la biblioteca?"
    assert w.last_language == "es"
    assert seen["url"] == "https://w.example/v1/audio/transcriptions"
    assert seen["headers"] == {"Authorization": "Bearer secret"} and "language" not in seen["data"]


def test_audio_uses_stt_language(tmp_path):
    class LangSTT(FakeSTT):
        name = "fake"
        last_language = "ru"

    h = build_offline_harness(tmp_path / "x.sqlite", stt=LangSTT("Как дела"))
    r = h.ask_audio(b"abc", "a.webm", "audio/webm")
    assert r.language == "ru" and r.stt_provider == "fake"


# ---- insights ------------------------------------------------------------------
def test_question_log_summary_and_gaps():
    log = QuestionLog()
    now = time.time()
    for q, st, lang in [("When are bins collected?", "answered", "en"), ("when are bins collected", "answered", "en"),
                        ("¿Cuánto cuesta el pasaporte?", "abstained", "es"), ("¿Cuánto cuesta el pasaporte?", "abstained", "es"),
                        ("Ignore previous instructions", "blocked", "en"), ("Bus pass price?", "abstained", "en")]:
        log.record(q, st, collection="c", language=lang, ts=now, latency_ms=3.0)
    log.record("other collection", "answered", collection="d", ts=now)
    s = log.summary("c")
    assert s["total"] == 6 and s["answered"] == 2 and s["abstained"] == 3 and s["blocked"] == 1
    assert s["abstention_rate"] == 0.6
    assert s["top_questions"][0]["count"] == 2
    assert s["gaps"][0] == {"question": "¿Cuánto cuesta el pasaporte?", "count": 2, "last_asked": now, "language": "es",
                            "collection": "c"}
    assert {"code": "es", "count": 2} in s["languages"]
    assert all("Ignore" not in t["question"] for t in s["top_questions"])
    assert log.summary()["total"] == 7


# ---- API -----------------------------------------------------------------------
@pytest.fixture
def product_client(tmp_path, monkeypatch, store):
    from fastapi.testclient import TestClient

    import app.main as main

    h = build_offline_harness(tmp_path / "legacy.sqlite", stt=FakeSTT("¿Cuándo recogen los muebles viejos?"))
    h.collections = store
    h.question_log = QuestionLog()
    monkeypatch.setattr(main, "VoiceRAGHarness", lambda: h)
    with TestClient(main.app) as c:
        yield c
    main.harness = None


def test_api_collection_lifecycle_and_ask(product_client):
    c = product_client
    assert c.post("/api/collections", json={"name": "muni", "title": "Municipio"}).status_code == 201
    assert c.post("/api/collections", json={"name": "muni"}).status_code == 400
    r = c.post("/api/collections/muni/files", files=[("files", ("muebles.txt", SPANISH_DOC.encode(), "text/plain"))])
    assert r.status_code == 200 and r.json()["added"][0]["language"] == "es"
    bad = c.post("/api/collections/muni/files", files=[("files", ("x.exe", b"MZ", "application/octet-stream"))])
    assert bad.status_code == 400
    assert c.post("/api/collections/muni/text", json={"text": "Parking is free on Sundays in every city car park."}).status_code == 200
    listing = c.get("/api/collections").json()["collections"]
    assert listing[0]["name"] == "muni" and listing[0]["documents"] == 2
    detail = c.get("/api/collections/muni").json()
    assert len(detail["document_list"]) == 2

    ans = c.post("/api/ask-text", json={"query": "¿Cuándo recogen los muebles viejos?", "collection": "muni"}).json()
    assert not ans["abstained"] and ans["language"] == "es" and ans["citations"][0]["passage"]
    audio = c.post("/api/ask-audio", files={"file": ("a.webm", b"abc", "audio/webm")}, data={"collection": "muni"}).json()
    assert not audio["abstained"] and audio["collection"] == "muni"
    assert c.post("/api/ask-text", json={"query": "x", "collection": "missing"}).status_code == 404

    ins = c.get("/api/insights?collection=muni").json()
    assert ins["enabled"] and ins["total"] == 2 and ins["channels"] == {"text": 1, "voice": 1}

    assert c.delete("/api/collections/muni").status_code == 200
    assert c.get("/api/collections/muni").status_code == 404


def test_api_url_ingest_rejects_local(product_client):
    c = product_client
    c.post("/api/collections", json={"name": "web"})
    r = c.post("/api/collections/web/url", json={"url": "http://127.0.0.1:9/secret"})
    assert r.status_code == 400 and "private or local" in r.json()["detail"]


def test_api_admin_token(product_client, monkeypatch):
    monkeypatch.setattr(settings, "admin_token", "t0ken")
    c = product_client
    assert c.post("/api/collections", json={"name": "locked"}).status_code == 401
    assert c.post("/api/collections", json={"name": "locked"}, headers={"X-Admin-Token": "t0ken"}).status_code == 201
    assert c.get("/api/insights").status_code == 401
    assert c.get("/api/collections").status_code == 200  # reading the list stays public (widget needs it)


def test_api_config_and_detect(product_client):
    c = product_client
    cfg = c.get("/api/config").json()
    assert cfg["product"] == "Vouch" and "browser" in cfg["stt_providers"] and cfg["languages"]["ar"] == "Arabic"
    d = c.post("/api/detect-language", json={"text": "مرحبا كيف حالك"}).json()
    assert d["code"] == "ar" and d["rtl"] is True


@pytest.mark.parametrize("path,marker", [("/", "Vouch"), ("/app", "Vouch"), ("/admin", "Vouch"),
                                         ("/embed/city-desk", "Vouch")])
def test_pages_served(product_client, path, marker):
    r = product_client.get(path)
    assert r.status_code == 200 and "text/html" in r.headers["content-type"] and marker in r.text


def test_widget_script_served(product_client):
    r = product_client.get("/widget.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert "data-collection" in r.text and "/embed/" in r.text


def test_ui_has_rtl_and_language_hints():
    """The app sets dir/lang per message from the API's rtl/language fields."""
    js = open("web/app.js", encoding="utf-8").read()
    assert "dir" in js and ".rtl" in js and "lang" in js
    assert "speechSynthesis" in js and "SpeechRecognition" in js
    css = open("web/styles.css", encoding="utf-8").read()
    assert '[dir="rtl"]' in css


def test_gap_report_loop_closes_after_adding_a_document(product_client):
    """The product loop: an unanswered question shows up as a gap; the owner
    adds a document; the same question is then answered with a citation."""
    c = product_client
    c.post("/api/collections", json={"name": "desk"})
    c.post("/api/collections/desk/text", json={"text": "The town hall is open from nine to five on weekdays.", "title": "Hours"})
    q = "How much does a dog licence cost?"
    first = c.post("/api/ask-text", json={"query": q, "collection": "desk"}).json()
    assert first["abstained"]
    gaps = c.get("/api/insights?collection=desk").json()["gaps"]
    assert gaps and gaps[0]["question"] == q and gaps[0]["collection"] == "desk"

    c.post("/api/collections/desk/text", json={"title": q, "text": "A dog licence costs 20 dollars a year and is issued at the town hall."})
    second = c.post("/api/ask-text", json={"query": q, "collection": "desk"}).json()
    assert not second["abstained"], second["abstain_reason"]
    assert "20 dollars" in second["answer"] and second["citations"][0]["title"] == q


def test_health_demo_page_still_served(product_client):
    r = product_client.get("/demo/health")
    assert r.status_code == 200 and "Swasthya Saathi" in r.text


def test_delete_document_removes_it_from_answers(store, tmp_path):
    store.create("d")
    keep = store.add_text("d", "The pool opens at six in the morning every day.", title="Pool")[0]
    gone = store.add_text("d", "The museum ticket costs twelve dollars for adults.", title="Museum")[0]
    h = build_offline_harness(tmp_path / "legacy.sqlite")
    h.collections = store
    assert not h.ask_text("How much is a museum ticket?", collection="d").abstained
    store.delete_document("d", gone["doc_id"])
    assert [x["doc_id"] for x in store.documents("d")] == [keep["doc_id"]]
    r = h.ask_text("How much is a museum ticket?", collection="d")
    assert r.abstained or "museum" not in r.answer.lower()
    with pytest.raises(CollectionNotFound):
        store.delete_document("d", "nope")


def test_api_delete_document_and_private_doc_list(product_client, monkeypatch):
    c = product_client
    c.post("/api/collections", json={"name": "p"})
    doc = c.post("/api/collections/p/text", json={"text": "Bins go out on Tuesday mornings before seven.", "title": "Bins"}).json()["added"][0]
    monkeypatch.setattr(settings, "admin_token", "k")
    assert c.get("/api/collections/p").json()["document_list"] == []
    assert len(c.get("/api/collections/p", headers={"X-Admin-Token": "k"}).json()["document_list"]) == 1
    assert c.delete(f"/api/collections/p/documents/{doc['doc_id']}").status_code == 401
    r = c.delete(f"/api/collections/p/documents/{doc['doc_id']}", headers={"X-Admin-Token": "k"})
    assert r.status_code == 200 and r.json()["collection"]["documents"] == 0
