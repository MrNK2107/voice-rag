"""Named collections of the user's own documents.

A collection is one Qdrant collection (dense vectors) plus one SQLite FTS5
file (lexical index), queried together by HybridRetriever. Metadata (title,
description, documents, languages) lives in ``<root>/collections.json``.

The store is embedder-agnostic: production passes the multilingual e5 model,
tests and the offline demo pass app.offline.HashingEmbedder. Existing indexes
(the legacy MSMARCO-XI index, the health demo) can be registered as read-only
collections with ``register_existing``.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import shutil
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.chunking import RawDoc, make_chunks
from app.ingest import ExtractedDoc, extract_file, fetch_url
from app.langid import detect_language, normalize_code
from app.textutil import fts_text

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
EMBED_BATCH = 64


class CollectionError(ValueError):
    pass


class CollectionNotFound(KeyError):
    pass


def slugify(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:48]


def _ensure_sqlite(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5("
        "chunk_id UNINDEXED, text, title, language, strategy, tokenize='unicode61')"
    )
    conn.execute(
        "CREATE TABLE IF NOT EXISTS chunks_meta (chunk_id TEXT PRIMARY KEY, text TEXT, "
        "title TEXT, language TEXT, strategy TEXT, parent_doc_id TEXT)"
    )
    conn.commit()
    conn.close()


class CollectionStore:
    def __init__(self, embedder: Any, root: Path | str, qdrant: Optional[QdrantClient] = None):
        self.embedder = embedder
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.qdrant = qdrant if qdrant is not None else QdrantClient(path=str(self.root / "qdrant"))
        self._meta_path = self.root / "collections.json"
        self._lock = threading.RLock()
        self._meta: Dict[str, Dict] = {}
        # name -> (qdrant client, qdrant collection, sqlite path) for registered indexes
        self._external: Dict[str, tuple] = {}
        self._retrievers: Dict[str, Any] = {}
        if self._meta_path.exists():
            try:
                self._meta = json.loads(self._meta_path.read_text(encoding="utf-8"))
            except ValueError:
                # Keep the unreadable file for recovery; the next save would
                # otherwise overwrite every collection's metadata.
                backup = self._meta_path.with_name(f"collections.json.corrupt-{int(time.time())}")
                os.replace(self._meta_path, backup)
                print(f"collections.json is unreadable; moved to {backup.name}")
                self._meta = {}

    # ---- metadata ----------------------------------------------------------
    def _save(self) -> None:
        persist = {k: v for k, v in self._meta.items() if not v.get("external")}
        # Write-then-rename, so a crash mid-write cannot leave a truncated file.
        tmp = self._meta_path.with_name(self._meta_path.name + ".tmp")
        tmp.write_text(json.dumps(persist, ensure_ascii=False, indent=1), encoding="utf-8")
        os.replace(tmp, self._meta_path)

    def _qname(self, name: str) -> str:
        return f"vouch_{name}"

    def _sqlite(self, name: str) -> Path:
        return self.root / name / "chunks.sqlite"

    def exists(self, name: str) -> bool:
        return name in self._meta

    def get(self, name: str) -> Dict:
        if name not in self._meta:
            raise CollectionNotFound(name)
        return self._meta[name]

    def list(self) -> List[Dict]:
        return [self.summary(n) for n in sorted(self._meta, key=lambda n: self._meta[n].get("created_at", 0))]

    def summary(self, name: str) -> Dict:
        m = self.get(name)
        docs = m.get("documents", [])
        langs: Dict[str, int] = {}
        for d in docs:
            langs[d.get("language", "und")] = langs.get(d.get("language", "und"), 0) + 1
        return {
            "name": name,
            "title": m.get("title") or name,
            "description": m.get("description", ""),
            "preset": m.get("preset"),
            "read_only": bool(m.get("external")),
            "documents": len(docs),
            "chunks": sum(d.get("chunks", 0) for d in docs),
            "languages": langs,
            "created_at": m.get("created_at"),
        }

    def documents(self, name: str) -> List[Dict]:
        return list(self.get(name).get("documents", []))

    # ---- lifecycle ---------------------------------------------------------
    def create(self, name: str, title: str = "", description: str = "", preset: Optional[str] = None) -> Dict:
        name = (name or "").strip().lower()
        if not NAME_RE.match(name):
            raise CollectionError("Collection names use lowercase letters, digits and dashes (max 48).")
        with self._lock:
            if name in self._meta:
                raise CollectionError(f"Collection '{name}' already exists.")
            dim = self.embedder.get_sentence_embedding_dimension()
            qn = self._qname(name)
            if self.qdrant.collection_exists(qn):
                self.qdrant.delete_collection(qn)
            self.qdrant.create_collection(qn, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            _ensure_sqlite(self._sqlite(name))
            self._meta[name] = {"name": name, "title": title or name, "description": description,
                                "preset": preset, "created_at": time.time(), "documents": []}
            self._save()
        return self.summary(name)

    def register_existing(self, name: str, title: str, qdrant: QdrantClient, qdrant_collection: str,
                          sqlite_path: str, documents: Iterable[Dict] = (), description: str = "",
                          preset: Optional[str] = None) -> Dict:
        """Expose an index built elsewhere as a read-only collection."""
        with self._lock:
            self._external[name] = (qdrant, qdrant_collection, str(sqlite_path))
            self._meta[name] = {"name": name, "title": title, "description": description, "preset": preset,
                                "created_at": time.time(), "external": True, "documents": list(documents)}
            self._retrievers.pop(name, None)
        return self.summary(name)

    def delete(self, name: str) -> None:
        with self._lock:
            m = self.get(name)
            if m.get("external"):
                raise CollectionError(f"'{name}' is a built-in collection and cannot be deleted.")
            qn = self._qname(name)
            if self.qdrant.collection_exists(qn):
                self.qdrant.delete_collection(qn)
            shutil.rmtree(self.root / name, ignore_errors=True)
            del self._meta[name]
            self._retrievers.pop(name, None)
            self._save()

    # ---- ingestion ---------------------------------------------------------
    def add_documents(self, name: str, docs: Iterable[ExtractedDoc]) -> List[Dict]:
        """Chunk, embed and index documents. Returns their metadata records."""
        m = self.get(name)
        if m.get("external"):
            raise CollectionError(f"'{name}' is a built-in collection and is read-only.")
        added = []
        with self._lock:
            for d in docs:
                text = (d.text or "").strip()
                if not text:
                    continue
                lang = detect_language(text[:2000]).code
                # A file or URL is identified by where it came from, so uploading
                # an edited version replaces the old one. Pasted text has no
                # source, so its content identifies it.
                key = f"{name}|{d.source}|{d.page}"
                if not d.source or d.source == "pasted":
                    key += "|" + hashlib.sha1(text.encode("utf-8")).hexdigest()
                doc_id = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
                raw = RawDoc(doc_id, text, title=d.title, language=lang, source_type="upload",
                             metadata={"source": d.source, "page": d.page})
                chunks = make_chunks(dataclasses.replace(raw, metadata=dict(raw.metadata)))
                if not chunks:
                    continue
                if any(x["doc_id"] == doc_id for x in m["documents"]):
                    self._remove_doc_chunks(name, doc_id)
                self._index_chunks(name, chunks)
                rec = {"doc_id": doc_id, "title": d.title, "source": d.source, "page": d.page,
                       "language": lang, "chunks": len(chunks), "chars": len(text), "added_at": time.time()}
                m["documents"] = [x for x in m["documents"] if x["doc_id"] != doc_id] + [rec]
                added.append(rec)
            self._save()
        return added

    def _index_chunks(self, name: str, chunks) -> None:
        qn = self._qname(name)
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i:i + EMBED_BATCH]
            vecs = self.embedder.encode([f"passage: {c.text}" for c in batch], normalize_embeddings=True)
            self.qdrant.upsert(qn, points=[
                PointStruct(id=c.chunk_id, vector=[float(x) for x in v], payload={**c.payload, "text": c.text})
                for c, v in zip(batch, vecs)
            ])
        conn = sqlite3.connect(str(self._sqlite(name)))
        for c in chunks:
            p = c.payload
            conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (c.chunk_id,))
            conn.execute("INSERT OR REPLACE INTO chunks_meta VALUES (?, ?, ?, ?, ?, ?)",
                         (c.chunk_id, c.text, p.get("title", ""), p.get("language", ""),
                          p.get("chunk_strategy", ""), p.get("parent_doc_id", "")))
            conn.execute("INSERT INTO chunks_fts(chunk_id, text, title, language, strategy) VALUES (?, ?, ?, ?, ?)",
                         (c.chunk_id, fts_text(c.text), p.get("title", ""), p.get("language", ""),
                          p.get("chunk_strategy", "")))
        conn.commit()
        conn.close()

    def _remove_doc_chunks(self, name: str, doc_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        self.qdrant.delete(self._qname(name), points_selector=FilterSelector(filter=Filter(
            must=[FieldCondition(key="parent_doc_id", match=MatchValue(value=doc_id))])))
        conn = sqlite3.connect(str(self._sqlite(name)))
        ids = [r[0] for r in conn.execute("SELECT chunk_id FROM chunks_meta WHERE parent_doc_id = ?", (doc_id,))]
        conn.executemany("DELETE FROM chunks_fts WHERE chunk_id = ?", [(i,) for i in ids])
        conn.execute("DELETE FROM chunks_meta WHERE parent_doc_id = ?", (doc_id,))
        conn.commit()
        conn.close()

    def delete_document(self, name: str, doc_id: str) -> Dict:
        """Remove one document's chunks from both indexes and its metadata."""
        with self._lock:
            m = self.get(name)
            if m.get("external"):
                raise CollectionError(f"'{name}' is a built-in collection and is read-only.")
            rec = next((d for d in m["documents"] if d["doc_id"] == doc_id), None)
            if rec is None:
                raise CollectionNotFound(f"{name}/{doc_id}")
            self._remove_doc_chunks(name, doc_id)
            m["documents"] = [d for d in m["documents"] if d["doc_id"] != doc_id]
            self._save()
            return rec

    def add_text(self, name: str, text: str, title: str = "Pasted text", source: str = "pasted") -> List[Dict]:
        return self.add_documents(name, [ExtractedDoc(title or "Pasted text", text, source)])

    def add_file(self, name: str, filename: str, data: bytes) -> List[Dict]:
        return self.add_documents(name, extract_file(filename, data))

    def add_url(self, name: str, url: str, **kw) -> List[Dict]:
        return self.add_documents(name, fetch_url(url, **kw))

    def add_raw_docs(self, name: str, docs: Iterable[RawDoc]) -> List[Dict]:
        """Index RawDocs (presets / examples); keeps their own language tags."""
        return self.add_documents(name, [ExtractedDoc(d.title, d.text, d.metadata.get("source", d.doc_id))
                                         for d in docs])

    # ---- retrieval ---------------------------------------------------------
    def retriever(self, name: str):
        from app.retriever import HybridRetriever

        self.get(name)
        # Fast path without the lock: add_documents holds it while embedding a
        # whole upload, and questions must not wait for that.
        cached = self._retrievers.get(name)
        if cached is not None:
            return cached
        with self._lock:
            if name not in self._retrievers:
                if name in self._external:
                    client, qn, sqlite_path = self._external[name]
                else:
                    client, qn, sqlite_path = self.qdrant, self._qname(name), str(self._sqlite(name))
                self._retrievers[name] = HybridRetriever(model=self.embedder, qdrant=client,
                                                         sqlite_path=sqlite_path, warmup=False, collection=qn)
            return self._retrievers[name]

    def title(self, name: str) -> str:
        return self.get(name).get("title") or name


def doc_language(d: RawDoc) -> str:
    return normalize_code(d.language) if d.language else detect_language(d.text).code
