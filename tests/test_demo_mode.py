"""The scripted health demo (app/demo.py) must keep behaving as scripted."""
import pytest

from app.demo import (
    ADVISORY_EN,
    ADVISORY_HI,
    DEMO_SCRIPT,
    HEALTH_DOCS,
    SOURCES,
    build_demo_harness,
    red_flag_advisory,
)


@pytest.fixture
def demo_harness(tmp_path):
    return build_demo_harness(tmp_path / "chunks.sqlite")


def _outcome(resp):
    if resp.abstained and "blocked" in (resp.abstain_reason or "").lower():
        return "blocked"
    return "abstain" if resp.abstained else "answer"


@pytest.mark.parametrize("step", DEMO_SCRIPT, ids=[s["shows"] for s in DEMO_SCRIPT])
def test_demo_script_step(demo_harness, step):
    resp = demo_harness.ask_text(step["query"])
    assert _outcome(resp) == step["expect"]
    if step["doc"]:
        assert resp.citations and resp.citations[0].parent_doc_id == step["doc"]
        assert resp.grounded


def test_every_demo_doc_has_a_cited_source():
    for d in HEALTH_DOCS:
        assert d.doc_id.rsplit("_", 1)[0] in SOURCES


def test_red_flag_advisory_language_and_negatives():
    assert red_flag_advisory("He is unconscious after the fall") == ADVISORY_EN
    assert red_flag_advisory("बच्चा बेहोश हो गया है") == ADVISORY_HI
    assert red_flag_advisory("How do I wash hands properly?") is None


def test_scripted_stt_audio_flow(demo_harness):
    resp = demo_harness.ask_audio(b"\x00" * 10, "q.webm", "audio/webm")
    assert resp.transcript == DEMO_SCRIPT[0]["query"]
    assert not resp.abstained


def test_demo_mode_api(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setenv("VOICE_RAG_DEMO", "1")
    with TestClient(main.app) as c:
        assert c.get("/health").json() == {"status": "ok", "harness_loaded": True, "demo_mode": True}
        info = c.get("/api/demo").json()
        assert info["enabled"] and len(info["script"]) == len(DEMO_SCRIPT)
        r = c.post("/api/ask-text", json={"query": "साँप के काटने पर क्या करें?"}).json()
        assert not r["abstained"] and r["advisory"] == ADVISORY_HI
        assert r["citations"][0]["title"]
    main.harness = None


def test_demo_endpoint_disabled_by_default(client):
    assert client.get("/api/demo").json()["enabled"] is False
