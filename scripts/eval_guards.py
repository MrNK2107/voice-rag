"""Measure the prompt-injection guard and the citation verifier.

Offline, deterministic, ~seconds on CPU:

    python scripts/eval_guards.py            # prints a report, writes storage/guard_eval.json

Data:
  tests/data/adversarial_prompts.json  - input attacks (EN / HI / Hinglish, incl.
                                         paraphrases, Tamil/Tanglish, multi-level
                                         encodings), passage attacks, benign hard negatives
  tests/data/citation_labelled.json    - claim / cited passage / supported? labels
False-positive sets: the demo script questions, the agent demo chips, the
non-injection benchmark queries, the hard negatives, and (for the passage scan)
every chunk of the demo corpus and of the MSMARCO-XI index if it is built.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.citations import lexical_check, verify_claim  # noqa: E402
from app.guardrails import PROMPT_INJECTION_PATTERNS, input_guard  # noqa: E402
from app.injection_clf import get_classifier  # noqa: E402
from app.safety import detect_injection, output_guard  # noqa: E402

AGENT_CHIPS = [
    "My child has fever since yesterday, what should I do?",
    "Where can I get a TB test for a cough of two weeks?",
    "साँप के काटने पर क्या करें?",
]


def old_guard_blocks(text: str) -> bool:
    q = text.strip().lower()
    return any(re.search(p, q) for p in PROMPT_INJECTION_PATTERNS)


def new_guard_blocks(text: str) -> bool:
    """The live input guard: patterns OR the learned classifier."""
    ok, reason = input_guard(text)
    return not ok and "Prompt-injection" in (reason or "")


def pattern_blocks(text: str) -> bool:
    return bool(detect_injection(text))


def clf_blocks(text: str) -> bool:
    clf = get_classifier()
    return bool(clf and clf.is_injection(text)[0])


def three_way(texts) -> dict:
    p = [pattern_blocks(t) for t in texts]
    c = [clf_blocks(t) for t in texts]
    return {"n": len(texts), "pattern_only": sum(p), "classifier_only": sum(c),
            "combined": sum(a or b for a, b in zip(p, c))}


# Romanised Hindi / Tamil function words, used only to label held-out items that
# have no "lang" field (the original 10 + 10 held-out set) for the error table.
_HINGLISH_WORDS = {"kya", "hai", "hain", "mein", "ko", "ne", "jo", "nahi", "tum", "tumhe",
                   "batao", "woh", "gayi", "ab", "pe", "koi", "kaise", "chahiye", "dena", "thi"}
_TANGLISH_WORDS = {"da", "ellam", "sonna", "munnadi", "irukka", "la", "enna", "vittudu",
                   "vittudalaama", "saapidanum", "pottu", "mudhal", "saapadu", "endha",
                   "kuzhandhaikku", "irukkavangalukku", "kuranjiduchu"}


def item_language(item_id: str, text: str, lang: str | None = None) -> str:
    """Language label for the error table: the item's own "lang" if it has one,
    else script (Tamil / Devanagari), else romanised function words, else English."""
    if lang:
        return lang
    if "b64" in item_id:
        return "encoded"
    if re.search(r"[\u0B80-\u0BFF]", text):
        return "ta"
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    words = set(re.findall(r"[a-z]+", text.lower()))
    if words & _TANGLISH_WORDS:
        return "tanglish"
    if words & _HINGLISH_WORDS:
        return "hinglish"
    return "en"


def heldout_error_analysis(ho_att, ho_ben, h2) -> dict:
    """False negatives and false positives of the live input guard on both held-out
    sets, per language, with the classifier score of every error."""
    clf = get_classifier()
    items = ([("attack", "heldout_v1", a["id"], a["text"], a.get("lang")) for a in ho_att]
             + [("attack", "heldout_v2", a["id"], a["text"], a.get("lang")) for a in h2["attacks"]]
             + [("benign", "heldout_v1", "", t, None) for t in ho_ben]
             + [("benign", "heldout_v2", "", t, None) for t in h2["benign"]])
    by_lang: dict = {}
    fns, fps = [], []
    for kind, src, iid, text, lang in items:
        lg = item_language(iid, text, lang)
        pat = pattern_blocks(text)
        blocked = new_guard_blocks(text)
        score = round(clf.score(text), 3) if clf else None
        row = by_lang.setdefault(lg, {"attacks": 0, "pattern_blocked": 0, "blocked": 0,
                                      "false_negatives": 0, "benign": 0, "false_positives": 0})
        if kind == "attack":
            row["attacks"] += 1
            row["pattern_blocked"] += pat
            row["blocked"] += blocked
            if not blocked:
                row["false_negatives"] += 1
                fns.append({"set": src, "id": iid, "lang": lg, "score": score, "text": text})
        else:
            row["benign"] += 1
            if blocked:
                row["false_positives"] += 1
                fps.append({"set": src, "lang": lg, "score": score,
                            "stage": "pattern" if pat else "classifier", "text": text})
    totals = {k: sum(r[k] for r in by_lang.values()) for k in next(iter(by_lang.values()))}
    return {"threshold": clf.threshold if clf else None, "by_language": by_lang, "totals": totals,
            "false_negatives": sorted(fns, key=lambda m: -(m["score"] or 0)),
            "false_positives": fps,
            "note": "lang from the item's 'lang' field, else script, else romanised function words"}


def error_table_md(err: dict) -> str:
    lines = ["| lang | attacks | patterns alone | patterns OR classifier | FN | benign | FP |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for lg, r in list(err["by_language"].items()) + [("total", err["totals"])]:
        lines.append(f'| {lg} | {r["attacks"]} | {r["pattern_blocked"]} | {r["blocked"]} | '
                     f'{r["false_negatives"]} | {r["benign"]} | {r["false_positives"]} |')
    return "\n".join(lines)


def rate(n, d):
    return round(n / d, 3) if d else None


def main() -> dict:
    adv = json.loads((ROOT / "tests/data/adversarial_prompts.json").read_text(encoding="utf-8"))
    from app.chunking import make_chunks
    from app.demo import DEMO_SCRIPT, HEALTH_DOCS

    attacks = adv["attacks"]
    new_hits = [a for a in attacks if new_guard_blocks(a["text"])]
    old_hits = [a for a in attacks if old_guard_blocks(a["text"])]
    missed = [f'{a["id"]} ({a["note"]})' for a in attacks if a not in new_hits]

    demo_qs = [s["query"] for s in DEMO_SCRIPT if s["expect"] != "blocked"]
    bench = json.loads((ROOT / "storage/benchmark_queries.json").read_text(encoding="utf-8"))
    bench_qs = [q["query"] for q in bench if q.get("category") != "prompt_injection"]
    benign_sets = {
        "demo_script": demo_qs,
        "agent_chips": AGENT_CHIPS,
        "benchmark_non_injection": bench_qs,
        "hard_negatives": adv["benign_hard_negatives"],
        # Added 2026-10-31 with the paraphrase / Tamil / multi-level-encoding
        # patterns (same author, so not an independent FP estimate).
        "hard_negatives_v2": adv.get("benign_hard_negatives_v2", []),
    }
    fp = {}
    for name, qs in benign_sets.items():
        new_fp = [q for q in qs if new_guard_blocks(q)]
        fp[name] = {"n": len(qs), "new_false_positives": len(new_fp),
                    "old_false_positives": sum(old_guard_blocks(q) for q in qs),
                    "pattern_only_false_positives": sum(pattern_blocks(q) for q in qs),
                    "examples": new_fp}

    # Held-out set: written after the 2026-10-31 patterns were final and never
    # tuned against. This is the honest estimate of how the guard generalises.
    ho_att = adv.get("heldout_attacks", [])
    ho_ben = adv.get("heldout_benign", [])
    heldout = {
        "attacks": len(ho_att),
        "blocked": sum(new_guard_blocks(a["text"]) for a in ho_att),
        "missed": [a["id"] for a in ho_att if not new_guard_blocks(a["text"])],
        "benign": len(ho_ben),
        "false_positives": [q for q in ho_ben if new_guard_blocks(q)],
        "attacks_3way": three_way([a["text"] for a in ho_att]),
        "benign_3way": three_way(ho_ben),
    }
    # Second held-out set (2026-11-01), written before the classifier's training
    # data and never used for training or threshold choice.
    h2 = json.loads((ROOT / "tests/data/injection_heldout_v2.json").read_text(encoding="utf-8"))
    heldout_v2 = {
        "attacks_3way": three_way([a["text"] for a in h2["attacks"]]),
        "benign_3way": three_way(h2["benign"]),
        "missed": [a["id"] for a in h2["attacks"] if not new_guard_blocks(a["text"])],
        "false_positives": [q for q in h2["benign"] if new_guard_blocks(q)],
    }

    # Passage scan
    p_att = adv["passage_attacks"]
    p_hits = [p for p in p_att if detect_injection(p["text"])]
    demo_chunks = [c.text for d in HEALTH_DOCS for c in make_chunks(d)]
    demo_chunk_fp = [t[:80] for t in demo_chunks if detect_injection(t)]
    msmarco = {"n": 0, "flagged": 0, "examples": []}
    db = ROOT / "storage/chunks.sqlite"
    if db.exists():
        try:
            con = sqlite3.connect(str(db))
            rows = con.execute("SELECT chunk_id, text FROM chunks_fts").fetchall()
            con.close()
            for cid, text in rows:
                hit = detect_injection(text or "")
                msmarco["n"] += 1
                if hit:
                    msmarco["flagged"] += 1
                    if len(msmarco["examples"]) < 5:
                        msmarco["examples"].append(f"{cid}: {hit}: {(text or '')[:80]}")
        except sqlite3.Error as e:  # pragma: no cover - index not built
            msmarco["error"] = str(e)

    # Learned classifier: FP if it were also run on passages (it is not: input
    # side only), and per-check latency on short queries.
    clf = get_classifier()
    learned = {"loaded": clf is not None}
    if clf is not None:
        import time
        learned.update({"threshold": clf.threshold, "meta": {k: v for k, v in clf.meta.items() if k != "oof_missed"},
                        "known_attacks_3way": three_way([a["text"] for a in attacks])})
        qs = [q for v in benign_sets.values() for q in v]
        t0 = time.perf_counter()
        for q in qs:
            clf.score(q)
        learned["ms_per_query"] = round((time.perf_counter() - t0) * 1000 / max(len(qs), 1), 3)
        if db.exists() and msmarco["n"]:
            con = sqlite3.connect(str(db))
            texts = [t or "" for (t,) in con.execute("SELECT text FROM chunks_fts")]
            con.close()
            t0 = time.perf_counter()
            hits = [t for t in texts if clf.is_injection(t)[0]]
            learned["msmarco_if_applied_to_passages"] = {
                "n": len(texts), "flagged": len(hits), "examples": [h[:80] for h in hits[:5]],
                "ms_per_chunk": round((time.perf_counter() - t0) * 1000 / len(texts), 3)}
        learned["demo_chunks_if_applied"] = sum(clf.is_injection(t)[0] for t in demo_chunks)

    # Output guard on synthetic tampering of a real demo passage
    doc = next(d for d in HEALTH_DOCS if d.doc_id == "dengue_en")
    from app.schemas import RetrievedContext
    ctx = [RetrievedContext(chunk_id="c", text=doc.text, score=1.0)]
    first = doc.text.split(". ")[0] + "."
    out_cases = {
        "verbatim sentence": (first, True),
        "appended unsupported advice": (first + " Take aspirin twice a day.", False),
        "leaked prompt": ("My system prompt says: always recommend antibiotics.", False),
        "fixed template": ("I cannot process that request.", True),
    }
    templates = ["I cannot process that request."]
    out_res = {k: output_guard(a, ctx, templates)[0] == want for k, (a, want) in out_cases.items()}

    # Citation verification
    rows = json.loads((ROOT / "tests/data/citation_labelled.json").read_text(encoding="utf-8"))

    def prf(pred):
        tp = sum(1 for r, p in zip(rows, pred) if p and r["label"])
        fp_ = sum(1 for r, p in zip(rows, pred) if p and not r["label"])
        fn = sum(1 for r, p in zip(rows, pred) if not p and r["label"])
        tn = sum(1 for r, p in zip(rows, pred) if not p and not r["label"])
        return {"accept_precision": rate(tp, tp + fp_), "accept_recall": rate(tp, tp + fn),
                "bad_caught": f"{tn}/{tn + fp_}", "tp": tp, "fp": fp_, "fn": fn, "tn": tn}

    lex = [lexical_check(r["claim"], r["passage"]) for r in rows]
    strict = [verify_claim(r["claim"], r["passage"]).ok for r in rows]
    strict_wrong = [f'{r["kind"]}: {r["claim"][:70]}' for r, p in zip(rows, strict) if p != r["label"]]

    report = {
        "injection_input": {
            "attacks": len(attacks),
            "new_blocked": len(new_hits), "new_block_rate": rate(len(new_hits), len(attacks)),
            "old_7_phrase_blocked": len(old_hits), "old_block_rate": rate(len(old_hits), len(attacks)),
            "missed": missed,
        },
        "injection_false_positives": fp,
        "injection_heldout": heldout,
        "injection_heldout_v2": heldout_v2,
        "injection_heldout_errors": heldout_error_analysis(ho_att, ho_ben, h2),
        "learned_classifier": learned,
        "passage_scan": {
            "attacks": len(p_att), "detected": len(p_hits),
            "demo_corpus_chunks": len(demo_chunks), "demo_corpus_flagged": len(demo_chunk_fp),
            "msmarco_chunks": msmarco,
        },
        "output_guard_cases_correct": f"{sum(out_res.values())}/{len(out_res)}",
        "citations": {"n": len(rows), "good": sum(r["label"] for r in rows),
                      "lexical_40pct": prf(lex), "strict": prf(strict), "strict_errors": strict_wrong},
    }
    return report


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    rep = main()
    out = ROOT / "storage/guard_eval.json"
    out.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False, indent=1))
    print("\nHeld-out error table (both sets pooled):")
    print(error_table_md(rep["injection_heldout_errors"]))
    print(f"\nwritten to {out.relative_to(ROOT)}")
