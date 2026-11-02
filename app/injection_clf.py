"""Learned second-stage prompt-injection detector (input side).

What it is: hashed character n-gram (2-5) + word (1-2) TF-IDF features and a
logistic regression, ~numpy only, CPU, sub-millisecond per short query. It runs
AFTER the pattern guard in app/safety.py and only adds blocks:

    blocked = pattern_guard(text) or score(text) > threshold

Why: the pattern list blocks every attack it was written against but only 2/10
of a held-out set of new paraphrases. A classifier over sub-word features
generalises a little better across wording and across English / Hindi /
Hinglish / Tamil / Tanglish. It is still small and trained on ~1k synthetic
examples by one author, so it is defence in depth, not a security boundary;
see GUARDRAILS.md for the measured numbers and failure cases.

The model file (app/models/injection_clf.npz) is produced deterministically by
scripts/train_injection_clf.py from tests/data/injection_train.json; the
threshold stored in it was chosen on a validation split for zero false
positives there, never on the held-out sets.
"""
from __future__ import annotations

import json
import logging
import math
import re
import zlib
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from app.safety import decode_candidates, normalize_text

MODEL_PATH = Path(__file__).resolve().parent / "models" / "injection_clf.npz"
_LFS_POINTER_PREFIX = b"version https://git-lfs"
logger = logging.getLogger(__name__)
N_BUCKETS = 1 << 18
CHAR_NGRAMS = (2, 3, 4, 5)
_WORD = re.compile(r"\w+", re.UNICODE)
_MAX_CHARS = 1000  # inputs are capped at 1000 chars by input_guard anyway


def _bucket(token: str) -> int:
    return zlib.crc32(token.encode("utf-8")) & (N_BUCKETS - 1)


def featurize(text: str) -> Dict[int, float]:
    """Raw (sublinear) term frequencies per hash bucket for `text`."""
    norm = normalize_text(text)[:_MAX_CHARS]
    counts: Dict[int, int] = {}
    padded = f" {norm} "
    for n in CHAR_NGRAMS:
        for i in range(len(padded) - n + 1):
            b = _bucket("c" + padded[i:i + n])
            counts[b] = counts.get(b, 0) + 1
    words = _WORD.findall(norm)
    for i, w in enumerate(words):
        b = _bucket("w" + w)
        counts[b] = counts.get(b, 0) + 1
        if i + 1 < len(words):
            b = _bucket("b" + w + " " + words[i + 1])
            counts[b] = counts.get(b, 0) + 1
    return {b: 1.0 + math.log(c) for b, c in counts.items()}


class InjectionClassifier:
    def __init__(self, buckets: np.ndarray, idf: np.ndarray, weights: np.ndarray,
                 bias: float, threshold: float, meta: Optional[dict] = None):
        self._idf = dict(zip(buckets.tolist(), idf.astype(np.float64).tolist()))
        self._w = dict(zip(buckets.tolist(), weights.astype(np.float64).tolist()))
        self.bias = float(bias)
        self.threshold = float(threshold)
        self.meta = meta or {}

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "InjectionClassifier":
        z = np.load(path or MODEL_PATH, allow_pickle=False)
        meta = json.loads(str(z["meta"])) if "meta" in z.files else {}
        return cls(z["buckets"], z["idf"], z["weights"], float(z["bias"]), float(z["threshold"]), meta)

    def _score_one(self, text: str) -> float:
        feats = featurize(text)
        vec = {b: tf * self._idf[b] for b, tf in feats.items() if b in self._idf}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        z = self.bias + sum(v * self._w[b] for b, v in vec.items()) / norm
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, z))))

    def score(self, text: str) -> float:
        """Probability-like injection score; max over the text and any
        base64/hex/URL-encoded payloads hidden in it."""
        if not text or not normalize_text(text):
            return 0.0
        best = self._score_one(text)
        for decoded in decode_candidates(text):
            best = max(best, self._score_one(decoded))
        return best

    def is_injection(self, text: str) -> Tuple[bool, float]:
        s = self.score(text)
        return s > self.threshold, s


@lru_cache(maxsize=1)
def get_classifier() -> Optional[InjectionClassifier]:
    """The shipped model, or None if the file is missing (guard then falls
    back to patterns only)."""
    try:
        with open(MODEL_PATH, "rb") as fh:
            if fh.read(len(_LFS_POINTER_PREFIX)) == _LFS_POINTER_PREFIX:
                # .npz is tracked with Git LFS (.gitattributes); a clone or CI checkout
                # without LFS gets the pointer file, and the guard silently loses stage 2.
                logger.warning("%s is a Git LFS pointer, not the model: run `git lfs pull`. "
                               "Injection guard falls back to patterns only.", MODEL_PATH)
                return None
        return InjectionClassifier.load()
    except (OSError, KeyError, ValueError):
        logger.warning("Injection classifier model missing or unreadable at %s; patterns only.", MODEL_PATH)
        return None


def classify_injection(text: str) -> Optional[str]:
    """Return "learned_classifier(p=0.93)" if the model flags `text`, else None."""
    clf = get_classifier()
    if clf is None:
        return None
    hit, s = clf.is_injection(text)
    return f"learned_classifier(p={s:.2f})" if hit else None


# ---- training helpers (used by scripts/train_injection_clf.py and tests) ----

def build_matrix(texts: Iterable[str]) -> List[Dict[int, float]]:
    return [featurize(t) for t in texts]


def fit(rows: List[Dict[int, float]], y: np.ndarray, l2: float = 1e-3, iters: int = 500,
        lr: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Deterministic full-batch logistic regression (Adam) on L2-normalised
    TF-IDF rows. Classes are re-weighted to equal total weight.
    Returns (buckets, idf, weights, bias)."""
    n = len(rows)
    df: Dict[int, int] = {}
    for r in rows:
        for b in r:
            df[b] = df.get(b, 0) + 1
    buckets = np.array(sorted(df), dtype=np.int64)
    col = {b: i for i, b in enumerate(buckets.tolist())}
    idf = np.array([math.log((1 + n) / (1 + df[b])) + 1.0 for b in buckets.tolist()])

    ri, ci, vals = [], [], []
    for i, r in enumerate(rows):
        v = np.array([tf * idf[col[b]] for b, tf in r.items()])
        v /= (np.linalg.norm(v) or 1.0)
        ri.extend([i] * len(r))
        ci.extend(col[b] for b in r)
        vals.extend(v.tolist())
    ri, ci, vals = np.array(ri), np.array(ci), np.array(vals)

    y = y.astype(np.float64)
    pos = y.sum()
    sw = np.where(y == 1, n / (2 * max(pos, 1)), n / (2 * max(n - pos, 1)))
    w = np.zeros(len(buckets))
    b0 = 0.0
    m_w = np.zeros_like(w); v_w = np.zeros_like(w); m_b = v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    for t in range(1, iters + 1):
        z = np.bincount(ri, weights=vals * w[ci], minlength=n) + b0
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        g = sw * (p - y) / n
        gw = np.bincount(ci, weights=vals * g[ri], minlength=len(w)) + l2 * w
        gb = g.sum()
        m_w = beta1 * m_w + (1 - beta1) * gw; v_w = beta2 * v_w + (1 - beta2) * gw * gw
        m_b = beta1 * m_b + (1 - beta1) * gb; v_b = beta2 * v_b + (1 - beta2) * gb * gb
        a = lr * math.sqrt(1 - beta2 ** t) / (1 - beta1 ** t)
        w -= a * m_w / (np.sqrt(v_w) + eps)
        b0 -= a * m_b / (math.sqrt(v_b) + eps)
    return buckets, idf, w, b0
