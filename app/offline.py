"""
Offline stand-ins for the network / heavyweight components of the pipeline.

Used by the test-suite (tests/conftest.py) and by `scripts/benchmark_offline.py`
so that the full STT -> retrieval -> generation -> grounding path can run on any
machine (including CI) with no Sarvam key, no Qdrant server, no HF download and
no torch install.

  * HashingEmbedder  - deterministic bag-of-words hashing embedder that mimics the
                       SentenceTransformer.encode() API (e5 "query:"/"passage:"
                       prefixes are stripped).
  * FakeSTT          - returns a canned transcript (or raises, to test failures).
  * build_offline_index - builds an in-memory Qdrant collection + a real SQLite
                       FTS5 index (same schema as scripts/build_index.py).
  * build_offline_harness - wires the above into a VoiceRAGHarness.
"""
from __future__ import annotations

import dataclasses
import hashlib
import sqlite3
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
import regex
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chunking import Chunk, RawDoc, make_chunks
from app.config import settings
from app.textutil import fts_text, tokenize

_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)
_IGNORED = {
    "query", "passage", "title",
    "what", "is", "the", "a", "an", "of", "to", "in", "and", "are", "how", "who",
    "क्या", "है", "हैं", "के", "का", "की", "में", "से", "को", "और", "एक", "यह",
}


class HashingEmbedder:
    """Deterministic, dependency-free replacement for SentenceTransformer."""

    def __init__(self, dim: int = 384):
        self.dim = dim

    def get_sentence_embedding_dimension(self) -> int:
        return self.dim

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "little") % self.dim

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for tok in tokenize(text.lower()):
            if tok in _IGNORED:
                continue
            vec[self._bucket(tok)] += 1.0
        return vec

    def encode(self, texts: Sequence[str], normalize_embeddings: bool = True, **_) -> np.ndarray:
        out = np.stack([self._embed_one(t) for t in texts])
        if normalize_embeddings:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out = out / norms
        return out


class FakeSTT:
    """Drop-in for SarvamSTT. Records calls; optionally raises."""

    def __init__(self, transcript: str = "मैकडॉनल्ड्स क्या है?", error: Optional[Exception] = None):
        self.transcript = transcript
        self.error = error
        self.calls: List[tuple] = []

    def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> str:
        self.calls.append((len(audio_bytes), filename, content_type))
        if self.error is not None:
            raise self.error
        return self.transcript


# A tiny bilingual corpus covering a handful of distinct topics.
SAMPLE_DOCS: List[RawDoc] = [
    RawDoc("doc_mcd", "मैकडॉनल्ड्स दुनिया की सबसे बड़ी फास्ट फूड रेस्टोरेंट श्रृंखला है। मैकडॉनल्ड्स की स्थापना अमेरिका में हुई थी।",
           title="मैकडॉनल्ड्स", language="hin"),
    RawDoc("doc_goa", "गोवा भारत का एक पश्चिमी तटीय राज्य है। गोवा की राजधानी पणजी है जो मांडवी नदी पर स्थित है।",
           title="गोवा", language="hin"),
    RawDoc("doc_corp", "कॉर्पोरेशन एक कानूनी इकाई है जो अपने शेयरधारकों से अलग अस्तित्व रखती है। कॉर्पोरेशन संपत्ति रख सकता है।",
           title="कॉर्पोरेशन", language="hin"),
    RawDoc("doc_honesty", "ईमानदारी का अर्थ है सच्चाई और नैतिक सिद्धांतों का पालन करना। ईमानदारी विश्वास की नींव है।",
           title="ईमानदारी", language="hin"),
    RawDoc("doc_photo", "Photosynthesis is the process plants use to convert sunlight, water and carbon dioxide into glucose and oxygen.",
           title="Photosynthesis", language="eng"),
    RawDoc("doc_python", "Python is a high level programming language known for readable syntax and a large standard library.",
           title="Python", language="eng"),
    RawDoc("doc_everest", "Mount Everest is the highest mountain above sea level, located in the Himalayas on the Nepal China border.",
           title="Mount Everest", language="eng"),
    RawDoc("doc_coffee", "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from Coffea plants.",
           title="Coffee", language="eng"),
    RawDoc("doc_cricket", "क्रिकेट बल्ले और गेंद से खेला जाने वाला खेल है जिसमें ग्यारह खिलाड़ियों की दो टीमें होती हैं।",
           title="क्रिकेट", language="hin"),
    RawDoc("doc_taj", "ताजमहल आगरा में यमुना नदी के किनारे स्थित सफेद संगमरमर का मकबरा है जिसे शाहजहाँ ने बनवाया था।",
           title="ताजमहल", language="hin"),
    RawDoc("doc_monsoon", "मानसून दक्षिण एशिया में मौसमी हवाओं का बदलाव है जो जून से सितंबर तक भारी वर्षा लाता है।",
           title="मानसून", language="hin"),
    RawDoc("doc_insulin", "Insulin is a hormone produced by the pancreas that regulates the amount of glucose in the blood.",
           title="Insulin", language="eng"),
]


def _ensure_sqlite(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    cur = conn.cursor()
    cur.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
        "chunk_id UNINDEXED, text, title, language, strategy, tokenize='unicode61')"
    )
    cur.execute(
        "CREATE TABLE IF NOT EXISTS chunks_meta (chunk_id TEXT PRIMARY KEY, text TEXT, "
        "title TEXT, language TEXT, strategy TEXT, parent_doc_id TEXT)"
    )
    conn.commit()
    return conn


def build_offline_index(
    sqlite_path: Path,
    docs: Optional[Iterable[RawDoc]] = None,
    embedder: Optional[HashingEmbedder] = None,
    collection: Optional[str] = None,
):
    """Index `docs` into an in-memory Qdrant + an on-disk SQLite FTS5 file.

    Returns (qdrant_client, embedder, chunks).
    """
    embedder = embedder or HashingEmbedder()
    collection = collection or settings.qdrant_collection
    docs = list(docs if docs is not None else SAMPLE_DOCS)

    chunks: List[Chunk] = []
    for d in docs:
        # make_chunks mutates doc.text (normalisation) - copy to keep SAMPLE_DOCS pristine
        chunks.extend(make_chunks(dataclasses.replace(d, metadata=dict(d.metadata))))

    qdrant = QdrantClient(":memory:")
    qdrant.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=embedder.dim, distance=Distance.COSINE),
    )
    vectors = embedder.encode([f"passage: {c.text}" for c in chunks], normalize_embeddings=True)
    qdrant.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=c.chunk_id, vector=v.tolist(), payload={**c.payload, "text": c.text})
            for c, v in zip(chunks, vectors)
        ],
    )

    conn = _ensure_sqlite(Path(sqlite_path))
    cur = conn.cursor()
    for c in chunks:
        p = c.payload
        cur.execute(
            "INSERT OR REPLACE INTO chunks_meta VALUES (?, ?, ?, ?, ?, ?)",
            (c.chunk_id, c.text, p.get("title", ""), p.get("language", ""),
             p.get("chunk_strategy", ""), p.get("parent_doc_id", "")),
        )
        cur.execute(
            "INSERT INTO chunks_fts(chunk_id, text, title, language, strategy) VALUES (?, ?, ?, ?, ?)",
            (c.chunk_id, fts_text(c.text), p.get("title", ""), p.get("language", ""), p.get("chunk_strategy", "")),
        )
    conn.commit()
    conn.close()
    return qdrant, embedder, chunks


# The bag-of-words hashing embedder produces much lower absolute cosine scores
# than e5 (no semantic smoothing), so its relevance thresholds are calibrated
# separately. Off-topic queries score ~0.0 against this corpus; on-topic ~0.3-0.6.
OFFLINE_MIN_DENSE_SCORE = 0.20
OFFLINE_MIN_DENSE_MARGIN = 0.10


def build_offline_harness(sqlite_path: Path, stt=None, docs: Optional[Iterable[RawDoc]] = None):
    from app.harness import VoiceRAGHarness
    from app.retriever import HybridRetriever

    qdrant, embedder, _ = build_offline_index(sqlite_path, docs)
    retriever = HybridRetriever(model=embedder, qdrant=qdrant, sqlite_path=str(sqlite_path))
    return VoiceRAGHarness(stt=stt or FakeSTT(), retriever=retriever)
