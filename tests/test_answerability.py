"""On-topic questions the corpus cannot answer must abstain, not get an
off-target extractive answer (the "double dose of ORS" miss, 2026-11-01).

The labelled set is tests/data/demo_answerability.json (written before the fix).
"""
import json
from pathlib import Path

import pytest

from app.demo import build_demo_harness, configure_demo_settings
from app.generator import AnswerGenerator, query_coverage
from app.schemas import RetrievedContext

DATA = json.loads((Path(__file__).parent / "data" / "demo_answerability.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def _shared_harness(tmp_path_factory):
    return build_demo_harness(tmp_path_factory.mktemp("ans") / "chunks.sqlite")


@pytest.fixture
def demo_harness(_shared_harness):
    # conftest restores the global settings after every test; re-apply demo mode.
    configure_demo_settings()
    return _shared_harness


def test_double_dose_ors_abstains(demo_harness):
    resp = demo_harness.ask_text("Is double the dose of ORS harmful for a child?")
    assert resp.abstained
    assert not resp.citations
    assert "vaccine" not in resp.answer.lower()
    assert "do not answer" in (resp.abstain_reason or "")


def test_scripted_ors_question_still_answered(demo_harness):
    resp = demo_harness.ask_text("How do I prepare ORS for a child with diarrhoea?")
    assert not resp.abstained
    assert resp.citations[0].parent_doc_id == "ors_en"


def test_query_coverage_basics():
    cov, missing = query_coverage("Is double the dose of ORS harmful for a child?",
                                  "Keep the card safe because it records every vaccine dose for the child.")
    assert cov == pytest.approx(0.4)
    assert set(missing) == {"double", "ors", "harmful"}
    # Inflections match by shared prefix (child/children, खतरे/खतरा).
    assert query_coverage("symptoms in children", "Common symptom of a child")[0] == 1.0
    assert query_coverage("डेंगू के खतरे", "डेंगू का खतरा")[0] == 1.0
    # Hindi light stemming: बचें / बचने share the stem बच.
    assert query_coverage("लू से कैसे बचें?", "लू से बचने के लिए पानी पिएँ")[0] == 1.0
    # A shared prefix alone is not enough for short words.
    assert query_coverage("heat stroke", "health centre")[1] == ["heat", "stroke"]
    # Question words alone do not count; an empty query is fully covered.
    assert query_coverage("what is it?", "anything")[0] == 1.0


def _ctx(cid, doc, text, rank_score):
    return RetrievedContext(chunk_id=cid, text=text, score=rank_score, strategy="t", language="eng",
                            title=None, parent_doc_id=doc)


def test_supporting_sentence_comes_from_the_same_document():
    ctxs = [
        _ctx("a", "tb_en", "TB treatment lasts at least six months without missing doses.", 0.9),
        _ctx("b", "snake_en", "Take the person to a hospital as fast as possible after the bite.", 0.8),
    ]
    answer, cits = AnswerGenerator().generate_extractive("How long does TB treatment take?", ctxs)
    assert {c.parent_doc_id for c in cits} == {"tb_en"}
    assert "hospital" not in answer


def _outcomes(harness, items):
    return [harness.ask_text(it["q"]) for it in items]


def test_answerability_set_rates(demo_harness):
    # Floors, not exact numbers: see README "Answerability" for the measured values.
    ans = _outcomes(demo_harness, DATA["answerable"])
    una = _outcomes(demo_harness, DATA["unanswerable"])
    answered = sum(not r.abstained for r in ans)
    abstained = sum(r.abstained for r in una)
    assert answered >= 18, answered
    assert abstained >= 16, abstained
    for it, r in zip(DATA["answerable"], ans):
        if not r.abstained:
            assert r.citations[0].parent_doc_id == it["doc"], it["q"]


def test_heldout_answerability_floors(demo_harness):
    # tests/data/demo_answerability_heldout.json was written after the rule was
    # frozen. Measured 2 Nov: 11/14 answered (the 3 misses abstain at the
    # retrieval guard, before the coverage gate), 14/16 unanswerable abstained.
    held = json.loads((Path(__file__).parent / "data" / "demo_answerability_heldout.json").read_text(encoding="utf-8"))
    answered = sum(not r.abstained for r in _outcomes(demo_harness, held["answerable"]))
    abstained = sum(r.abstained for r in _outcomes(demo_harness, held["unanswerable"]))
    assert answered >= 10, answered
    assert abstained >= 13, abstained
