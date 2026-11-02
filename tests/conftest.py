"""
Shared fixtures. Everything here runs fully offline:

* Sarvam STT  -> app.offline.FakeSTT (and `requests.post` is blocked outright)
* Embeddings  -> app.offline.HashingEmbedder (no torch / HF download)
* Qdrant      -> in-memory QdrantClient(":memory:")
* SQLite FTS5 -> real FTS5 index in a per-test tmp dir
* Generation is already extractive (no LLM call exists in the pipeline).
"""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
# app/main.py resolves web/ relative to the CWD at import time.
os.chdir(ROOT)

from app.config import settings  # noqa: E402
from app.offline import (  # noqa: E402
    OFFLINE_MIN_DENSE_MARGIN,
    OFFLINE_MIN_DENSE_SCORE,
    FakeSTT,
    build_offline_harness,
    build_offline_index,
)


@pytest.fixture(autouse=True)
def offline_settings(monkeypatch):
    """Calibrate thresholds for the hashing embedder and make sure no test can
    accidentally reach the Sarvam API."""
    monkeypatch.delenv("VOICE_RAG_DEMO", raising=False)
    monkeypatch.setattr(settings, "sarvam_api_key", "")
    # demo mode mutates these globals; monkeypatch restores them after each test
    monkeypatch.setattr(settings, "corpus_name", settings.corpus_name)
    monkeypatch.setattr(settings, "stt_cache_enabled", False)
    monkeypatch.setattr(settings, "min_dense_score", OFFLINE_MIN_DENSE_SCORE)
    monkeypatch.setattr(settings, "min_dense_margin", OFFLINE_MIN_DENSE_MARGIN)
    monkeypatch.setattr(settings, "min_query_coverage", settings.min_query_coverage)

    import requests

    def _no_network(*args, **kwargs):
        raise AssertionError("Network access attempted during offline tests")

    monkeypatch.setattr(requests, "post", _no_network)
    yield


@pytest.fixture
def sqlite_path(tmp_path):
    return tmp_path / "chunks.sqlite"


@pytest.fixture
def offline_index(sqlite_path):
    qdrant, embedder, chunks = build_offline_index(sqlite_path)
    return {"qdrant": qdrant, "embedder": embedder, "chunks": chunks, "sqlite_path": str(sqlite_path)}


@pytest.fixture
def retriever(offline_index):
    from app.retriever import HybridRetriever

    return HybridRetriever(
        model=offline_index["embedder"],
        qdrant=offline_index["qdrant"],
        sqlite_path=offline_index["sqlite_path"],
    )


@pytest.fixture
def fake_stt():
    return FakeSTT()


@pytest.fixture
def harness(sqlite_path, fake_stt):
    return build_offline_harness(sqlite_path, stt=fake_stt)


@pytest.fixture
def client(harness, monkeypatch):
    """FastAPI TestClient whose startup hook builds the offline harness."""
    from fastapi.testclient import TestClient

    import app.main as main

    monkeypatch.setattr(main, "VoiceRAGHarness", lambda: harness)
    with TestClient(main.app) as c:
        yield c
    main.harness = None
