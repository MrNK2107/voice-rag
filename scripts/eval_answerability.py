"""
Answerability eval on the bundled demo corpus.

Runs tests/data/demo_answerability.json (answerable questions vs on-topic
questions the corpus cannot answer) through the demo harness at several
min_query_coverage values and writes storage/answerability_eval.json.

    python scripts/eval_answerability.py                      # dev set (rule was tuned on it)
    python scripts/eval_answerability.py --data tests/data/demo_answerability_heldout.json

Offline, about 1 s, < 300 MB.
"""
import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from app.config import settings  # noqa: E402
from app.demo import DEMO_MIN_QUERY_COVERAGE, build_demo_harness  # noqa: E402

THRESHOLDS = [0.0, 0.5, DEMO_MIN_QUERY_COVERAGE, 0.75]


def run(harness, data, threshold):
    settings.min_query_coverage = threshold
    out = {"threshold": threshold, "answerable": [], "unanswerable": []}
    for cls in ("answerable", "unanswerable"):
        for it in data[cls]:
            r = harness.ask_text(it["q"])
            row = {"q": it["q"], "abstained": r.abstained, "reason": r.abstain_reason,
                   "doc": r.citations[0].parent_doc_id if r.citations else None}
            if cls == "answerable":
                row["correct_doc"] = (not r.abstained) and row["doc"] == it["doc"]
            out[cls].append(row)
    a, u = out["answerable"], out["unanswerable"]
    out["summary"] = {
        "answerable_answered": sum(not r["abstained"] for r in a),
        "answerable_answered_correct_doc": sum(r["correct_doc"] for r in a),
        "answerable_total": len(a),
        "unanswerable_abstained": sum(r["abstained"] for r in u),
        "unanswerable_total": len(u),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "tests" / "data" / "demo_answerability.json"),
                    help="labelled set; tests/data/demo_answerability_heldout.json is the held-out one")
    ap.add_argument("--out", default=None, help="default: storage/answerability_eval[_heldout].json")
    args = ap.parse_args()
    data = json.loads(Path(args.data).read_text(encoding="utf-8"))
    if args.out is None:
        suffix = "_heldout" if "heldout" in Path(args.data).name else ""
        args.out = str(ROOT / "storage" / f"answerability_eval{suffix}.json")
    harness = build_demo_harness(Path(tempfile.mkdtemp(prefix="ans_eval_")) / "chunks.sqlite")
    results = [run(harness, data, t) for t in THRESHOLDS]
    settings.min_query_coverage = DEMO_MIN_QUERY_COVERAGE
    print(f"{'min_query_coverage':>18}  answerable answered (right doc)  unanswerable abstained")
    for r in results:
        s = r["summary"]
        tag = "  <- demo default" if r["threshold"] == DEMO_MIN_QUERY_COVERAGE else ""
        print(f"{r['threshold']:>18.2f}  {s['answerable_answered']:>2}/{s['answerable_total']} "
              f"({s['answerable_answered_correct_doc']})                  "
              f"{s['unanswerable_abstained']:>2}/{s['unanswerable_total']}{tag}")
    Path(args.out).write_text(json.dumps({"data": Path(args.data).name, "demo_default": DEMO_MIN_QUERY_COVERAGE, "runs": results},
                                         ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
