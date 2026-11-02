"""Unit tests for hybrid retrieval: RRF fusion, diversity caps, FTS escaping,
confidence margin, and the dense / lexical legs against real (offline) indexes."""
import pytest

from app.retriever import HybridRetriever, escape_fts_query


def hit(cid, rank, score=0.5, parent=None, strategy=None, source="dense"):
    return {
        "chunk_id": cid,
        "rank": rank,
        "score": score,
        "text": f"text {cid}",
        "strategy": strategy if strategy is not None else f"strat_{cid}",
        "language": "hin",
        "parent_doc_id": parent or f"p_{cid}",
        "title": cid,
        "source": source,
    }


@pytest.fixture
def bare():
    """A retriever with no backends - enough for the pure fuse/confidence logic."""
    return HybridRetriever(model=object(), qdrant=object(), sqlite_path=":memory:", warmup=False)


# ---------------------------------------------------------------- escape_fts_query
def test_escape_fts_query_drops_stopwords_and_punctuation():
    assert escape_fts_query("What is the capital of Goa?") == "capital OR Goa"
    assert escape_fts_query("गोवा की राजधानी क्या है?") == "गोवा OR राजधानी"


def test_escape_fts_query_keeps_indic_conjuncts_whole():
    # stdlib \w would shred this into fragments
    assert escape_fts_query("कॉर्पोरेशन") == "कॉर्पोरेशन"


def test_escape_fts_query_empty_and_injection_safe():
    assert escape_fts_query("what is the") == ""
    assert escape_fts_query('"; DROP TABLE x; --') == "DROP OR TABLE OR x"


def test_escape_fts_query_caps_terms():
    q = " ".join(f"term{i}" for i in range(50))
    assert len(escape_fts_query(q).split(" OR ")) == 20


# ---------------------------------------------------------------- fuse (RRF)
def test_fuse_rrf_rewards_agreement_between_legs(bare):
    dense = [hit("a", 1), hit("b", 2), hit("c", 3)]
    lexical = [hit("c", 1, score=9.0, source="lexical"), hit("d", 2, score=5.0, source="lexical")]
    out = bare.fuse(dense, lexical)
    ids = [c.chunk_id for c in out]
    # c appears in both lists -> 1/63 + 1/61 beats a's single 1/61
    assert ids[0] == "c"
    assert set(ids) == {"a", "b", "c", "d"}
    c = out[0]
    assert c.score == pytest.approx(1 / 63 + 1 / 61)
    assert c.dense_score == 0.5 and c.lexical_score == 9.0


def test_fuse_scores_are_monotonic(bare):
    out = bare.fuse([hit(str(i), i) for i in range(1, 6)], [])
    scores = [c.score for c in out]
    assert scores == sorted(scores, reverse=True)
    assert [c.chunk_id for c in out] == ["1", "2", "3", "4", "5"]


def test_fuse_lexical_only_hit_has_no_dense_score(bare):
    out = bare.fuse([], [hit("x", 1, score=3.0, source="lexical")])
    assert out[0].dense_score is None and out[0].lexical_score == 3.0


def test_fuse_caps_two_chunks_per_parent_doc(bare):
    dense = [hit(f"c{i}", i, parent="same", strategy=f"s{i}") for i in range(1, 5)]
    dense.append(hit("other", 5, parent="p2"))
    out = bare.fuse(dense, [])
    parents = [c.parent_doc_id for c in out]
    assert parents.count("same") == 2
    assert "p2" in parents


def test_fuse_caps_three_chunks_per_strategy(bare):
    dense = [hit(f"c{i}", i, strategy="micro") for i in range(1, 6)]
    dense.append(hit("z", 6, strategy="macro"))
    out = bare.fuse(dense, [])
    strategies = [c.strategy for c in out]
    assert strategies.count("micro") == 3
    assert "macro" in strategies


def test_fuse_respects_top_k(bare):
    dense = [hit(str(i), i, strategy=f"s{i}") for i in range(1, 20)]
    assert len(bare.fuse(dense, [], top_k=4)) == 4
    assert len(bare.fuse(dense, [])) == 6


def test_fuse_empty(bare):
    assert bare.fuse([], []) == []


# ---------------------------------------------------------------- confidence margin
def test_confidence_margin_uses_tail_ranks_10_to_40(bare):
    scores = [0.9] + [0.5] * 9 + [0.3] * 30
    conf = bare.confidence_from_dense_hits([{"score": s} for s in scores])
    assert conf["top_dense"] == 0.9
    assert conf["margin"] == pytest.approx(0.6)


def test_confidence_short_list_uses_rest_as_tail(bare):
    conf = bare.confidence_from_dense_hits([{"score": 0.8}, {"score": 0.6}, {"score": 0.4}])
    assert conf["margin"] == pytest.approx(0.3)


def test_confidence_single_and_empty(bare):
    assert bare.confidence_from_dense_hits([]) == {"top_dense": 0.0, "margin": 0.0}
    # A one-chunk collection has no tail: the margin is the top score, so a new
    # collection with a single uploaded note can still answer (it used to be 0,
    # which made every one-document collection abstain).
    assert bare.confidence_from_dense_hits([{"score": 0.7}])["margin"] == 0.7


# ---------------------------------------------------------------- live legs (offline indexes)
def test_dense_search_ranks_matching_doc_first(retriever):
    hits = retriever.dense_search("Mount Everest Himalayas", limit=5)
    assert hits[0]["parent_doc_id"] == "doc_everest"
    assert [h["rank"] for h in hits] == list(range(1, len(hits) + 1))
    assert all(h["source"] == "dense" for h in hits)


def test_dense_search_survives_backend_errors(retriever, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("qdrant down")

    monkeypatch.setattr(retriever.qdrant, "query_points", boom)
    assert retriever.dense_search("anything") == []


def test_lexical_search_hits_fts5(retriever):
    hits = retriever.lexical_search("photosynthesis sunlight")
    assert hits
    assert hits[0]["parent_doc_id"] == "doc_photo"
    # bm25() is negated so higher == better
    assert hits[0]["score"] >= hits[-1]["score"]
    assert all(h["source"] == "lexical" for h in hits)


def test_lexical_search_stopword_only_query_returns_nothing(retriever):
    assert retriever.lexical_search("what is the") == []


def test_lexical_search_missing_db_is_graceful(retriever, tmp_path):
    retriever.sqlite_path = str(tmp_path / "missing" / "nope.sqlite")
    assert retriever.lexical_search("coffee") == []


@pytest.mark.parametrize(
    "query, expected_parent",
    [
        ("गोवा की राजधानी", "doc_goa"),
        ("insulin pancreas hormone", "doc_insulin"),
        ("ताजमहल आगरा", "doc_taj"),
        ("Python programming language", "doc_python"),
    ],
)
def test_hybrid_retrieve_top1(retriever, query, expected_parent):
    contexts, confidence = retriever.retrieve(query)
    assert contexts[0].parent_doc_id == expected_parent
    assert confidence["margin"] > 0


def test_hybrid_retrieve_off_topic_has_low_confidence(retriever):
    _, confidence = retriever.retrieve("distance to Mars")
    assert confidence["top_dense"] < 0.2
