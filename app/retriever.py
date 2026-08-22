import regex
import sqlite3
from collections import defaultdict
from typing import Dict, List

from qdrant_client import QdrantClient
from qdrant_client.models import SearchParams
from sentence_transformers import SentenceTransformer

from app.config import settings
from app.schemas import RetrievedContext

# stdlib re's Unicode \w excludes combining marks, shredding Indic-script conjuncts
# (see app/generator.py for the full explanation). Use the same \p{L}\p{M}\p{N}
# tokenizer here so lexical search sees real Hindi/Indic words, not fragments.
_WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)

_STOPWORDS = {
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
    "is", "are", "was", "were", "be", "been", "being",
    "the", "a", "an", "this", "that", "these", "those",
    "of", "in", "on", "at", "to", "for", "and", "or", "do", "does", "did",
    # Hindi function words (MSMARCO-XI's currently indexed language)
    "क्या", "है", "हैं", "था", "थी", "थे", "के", "का", "की", "में", "से",
    "को", "पर", "और", "या", "एक", "यह", "वह", "कि", "जो",
}


def escape_fts_query(q: str) -> str:
    terms = [t for t in _WORD_PATTERN.findall(q) if t.lower() not in _STOPWORDS]
    return " OR ".join(terms[:20]) if terms else ""


class HybridRetriever:
    def __init__(self):
        self.model = SentenceTransformer(settings.embed_model, device="cpu")

        try:
            self.qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key or None,
                timeout=2.0,
            )
            # test connectivity
            self.qdrant.get_collections()
        except Exception:
            # Fallback to local embedded Qdrant database
            self.qdrant = QdrantClient(path=settings.qdrant_path)

        self.sqlite_path = settings.sqlite_fts_path

        # Preload / warmup embedding model on startup
        self.model.encode(["query: warmup"], normalize_embeddings=True)

    def _get_sqlite_conn(self):
        return sqlite3.connect(self.sqlite_path, check_same_thread=False)

    def dense_search(self, query: str, limit: int = 40) -> List[Dict]:
        qvec = self.model.encode([f"query: {query}"], normalize_embeddings=True)[0]

        try:
            res = self.qdrant.query_points(
                collection_name=settings.qdrant_collection,
                query=qvec.tolist(),
                limit=limit,
                with_payload=True,
                search_params=SearchParams(hnsw_ef=32),
            )
            result = res.points
        except Exception:
            try:
                result = self.qdrant.search(
                    collection_name=settings.qdrant_collection,
                    query_vector=qvec.tolist(),
                    limit=limit,
                    with_payload=True,
                    search_params=SearchParams(hnsw_ef=32),
                )
            except Exception as e:
                print(f"Qdrant search error: {e}")
                result = []

        hits = []
        for rank, r in enumerate(result, start=1):
            payload = getattr(r, "payload", {}) or {}
            hits.append({
                "chunk_id": str(r.id),
                "rank": rank,
                "score": float(r.score),
                "text": payload.get("text", ""),
                "strategy": payload.get("chunk_strategy", ""),
                "language": payload.get("language", ""),
                "parent_doc_id": payload.get("parent_doc_id", ""),
                "title": payload.get("title", ""),
                "source": "dense",
            })

        return hits

    def lexical_search(self, query: str, limit: int = 20) -> List[Dict]:
        fts_query = escape_fts_query(query)
        if not fts_query:
            return []

        try:
            conn = self._get_sqlite_conn()
            cur = conn.cursor()
            rows = cur.execute(
                """
                SELECT 
                    f.chunk_id,
                    m.text,
                    m.title,
                    m.language,
                    m.strategy,
                    m.parent_doc_id,
                    bm25(chunks_fts) as score
                FROM chunks_fts f
                JOIN chunks_meta m ON f.chunk_id = m.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
            conn.close()
        except Exception as e:
            return []

        hits = []
        for rank, row in enumerate(rows, start=1):
            chunk_id, text, title, language, strategy, parent_doc_id, score = row
            hits.append({
                "chunk_id": chunk_id,
                "rank": rank,
                "score": float(-score),
                "text": text,
                "strategy": strategy,
                "language": language,
                "parent_doc_id": parent_doc_id,
                "title": title,
                "source": "lexical",
            })

        return hits

    def fuse(self, dense_hits: List[Dict], lexical_hits: List[Dict], top_k: int = 6) -> List[RetrievedContext]:
        k = 60
        candidates = {}

        for hit in dense_hits:
            cid = hit["chunk_id"]
            if cid not in candidates:
                candidates[cid] = dict(hit)
                candidates[cid]["rrf_score"] = 0.0
                candidates[cid]["dense_score"] = hit["score"]
                candidates[cid]["lexical_score"] = None

            candidates[cid]["rrf_score"] += 1.0 / (k + hit["rank"])

        for hit in lexical_hits:
            cid = hit["chunk_id"]
            if cid not in candidates:
                candidates[cid] = dict(hit)
                candidates[cid]["rrf_score"] = 0.0
                candidates[cid]["dense_score"] = None
                candidates[cid]["lexical_score"] = hit["score"]

            candidates[cid]["rrf_score"] += 1.0 / (k + hit["rank"])
            candidates[cid]["lexical_score"] = hit["score"]

        ranked = sorted(
            candidates.values(),
            key=lambda x: x["rrf_score"],
            reverse=True,
        )

        selected = []
        parent_counts = defaultdict(int)
        strategy_counts = defaultdict(int)

        for item in ranked:
            parent = item.get("parent_doc_id") or item["chunk_id"]
            strategy = item.get("strategy") or ""

            if parent_counts[parent] >= 2:
                continue

            if strategy and strategy_counts[strategy] >= 3:
                continue

            selected.append(item)
            parent_counts[parent] += 1
            if strategy:
                strategy_counts[strategy] += 1

            if len(selected) >= top_k:
                break

        contexts = []

        for item in selected:
            contexts.append(
                RetrievedContext(
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    score=float(item["rrf_score"]),
                    dense_score=item.get("dense_score"),
                    lexical_score=item.get("lexical_score"),
                    strategy=item.get("strategy"),
                    language=item.get("language"),
                    parent_doc_id=item.get("parent_doc_id"),
                    title=item.get("title"),
                )
            )

        return contexts

    def confidence_from_dense_hits(self, dense_hits: List[Dict]) -> Dict[str, float]:
        """
        Absolute cosine similarity from e5-style embeddings doesn't generalize as a
        relevance threshold across corpus sizes (empirically, unrelated-query "noise
        floor" scores climb as the corpus grows - see GUARDRAILS.md). Instead we use
        the MARGIN between the top hit and the mean of the tail candidates (rank
        10-40): for a genuinely relevant query the top hit stands out well above the
        rest; for an off-topic query, everything - including the "best" match -
        looks similarly mediocre, so the margin collapses. This generalizes across
        corpus size/language, unlike a fixed absolute score.
        """
        scores = [h["score"] for h in dense_hits]
        if not scores:
            return {"top_dense": 0.0, "margin": 0.0}

        top = scores[0]
        tail = scores[10:40] if len(scores) > 10 else scores[1:]
        tail_mean = (sum(tail) / len(tail)) if tail else top

        return {"top_dense": top, "margin": top - tail_mean}

    def retrieve(self, query: str):
        dense = self.dense_search(query)
        lexical = self.lexical_search(query)
        contexts = self.fuse(dense, lexical)
        confidence = self.confidence_from_dense_hits(dense)
        return contexts, confidence
