"""Train the learned injection detector (app/injection_clf.py).

    python scripts/train_injection_clf.py          # writes app/models/injection_clf.npz

Deterministic: same data -> same model (no random init, full-batch Adam, a
hash-based train/validation split). Data:
  * tests/data/injection_train.json : hand-written attacks + benign (incl. hard
    negatives such as "can I ignore the pain"), plus templates expanded
    deterministically
  * the 54 known attacks in tests/data/adversarial_prompts.json["attacks"]
Never used here: either held-out set, the benign FP-evaluation sets, MSMARCO.

Threshold: hand-written items are split into 5 folds by a hash of the text;
for each fold a model is trained on the other 4 folds + all template
expansions (templates never go to validation: they are near-duplicates of each
other and would make validation look easier than real input). The threshold is
the smallest value above every out-of-fold benign score (zero validation false
positives), rounded up to 0.01 and never below 0.5. The shipped model is then
trained on everything with that threshold.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.injection_clf import MODEL_PATH, InjectionClassifier, build_matrix, fit  # noqa: E402
from app.safety import normalize_text  # noqa: E402

TRAIN = ROOT / "tests/data/injection_train.json"
ADV = ROOT / "tests/data/adversarial_prompts.json"
VERSION = "1"
N_FOLDS = 5


def data_digest() -> str:
    """Hash of the parsed training inputs (line-ending / whitespace independent)."""
    blob = "".join(json.dumps(json.loads(p.read_text(encoding="utf-8")), sort_keys=True, ensure_ascii=False)
                   for p in (TRAIN, ADV))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def expand(templates) -> list:
    out = []
    for t in templates:
        combos = list(itertools.product(*t["parts"]))
        for c in combos[:: t.get("stride", 1)]:
            out.append(" ".join(p for p in c if p).strip())
    return out


def load_data():
    d = json.loads(TRAIN.read_text(encoding="utf-8"))
    adv = json.loads(ADV.read_text(encoding="utf-8"))
    hand = [(t, 1) for t in d["attacks"]] + [(a["text"], 1) for a in adv["attacks"]] + [(t, 0) for t in d["benign"]]
    templ = [(t, 1) for t in expand(d["attack_templates"])] + [(t, 0) for t in expand(d["benign_templates"])]
    seen, hand_u, templ_u = set(), [], []
    for bucket, items in ((hand_u, hand), (templ_u, templ)):
        for t, y in items:
            k = normalize_text(t)
            if k and k not in seen:
                seen.add(k)
                bucket.append((t, y))
    return hand_u, templ_u


def fold(text: str) -> int:
    return zlib.crc32(normalize_text(text).encode("utf-8")) % N_FOLDS


def _fit_model(items, threshold=0.5) -> InjectionClassifier:
    rows = build_matrix([t for t, _ in items])
    y = np.array([lab for _, lab in items])
    buckets, idf, w, b = fit(rows, y)
    return InjectionClassifier(buckets, idf, w, b, threshold=threshold)


def main() -> dict:
    hand, templ = load_data()
    t0 = time.perf_counter()
    # Out-of-fold validation scores for every hand-written item.
    oof = []
    for k in range(N_FOLDS):
        val = [x for x in hand if fold(x[0]) == k]
        clf_k = _fit_model([x for x in hand if fold(x[0]) != k] + templ)
        oof.extend((clf_k.score(t), lab, t) for t, lab in val)
    max_benign = max((s for s, lab, _ in oof if lab == 0), default=0.0)
    threshold = max(0.5, float(np.ceil((max_benign + 1e-9) * 100) / 100))
    val_att = [s for s, lab, _ in oof if lab == 1]
    final = _fit_model(hand + templ, threshold)
    buckets = np.array(sorted(final._w), dtype=np.int64)
    idf = np.array([final._idf[b] for b in buckets.tolist()])
    w = np.array([final._w[b] for b in buckets.tolist()])
    b = final.bias
    meta = {
        "version": VERSION,
        "data_sha256": data_digest(),
        "n_train": len(hand) + len(templ), "n_train_attacks": sum(lab for _, lab in hand + templ),
        "n_hand_written": len(hand), "n_template": len(templ), "folds": N_FOLDS,
        "oof_attacks": len(val_att), "oof_attacks_blocked": int(sum(s > threshold for s in val_att)),
        "oof_benign": len(oof) - len(val_att), "oof_max_benign_score": round(max_benign, 4),
        "threshold": threshold,
        "oof_missed": [t for s, lab, t in oof if lab == 1 and s <= threshold],
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODEL_PATH, buckets=buckets.astype(np.int32), idf=idf.astype(np.float32),
                        weights=w.astype(np.float32), bias=np.float64(b), threshold=np.float64(threshold),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    meta["model_bytes"] = MODEL_PATH.stat().st_size
    meta["train_seconds"] = round(time.perf_counter() - t0, 2)
    return meta


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(main(), ensure_ascii=False, indent=1))
