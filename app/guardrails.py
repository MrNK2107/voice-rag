import re
import regex
from typing import List, Optional, Tuple

from app.config import settings
from app.schemas import RetrievedContext

_WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)


UNSAFE_PATTERNS = [
    r"\bmake a bomb\b",
    r"\bbuild a weapon\b",
    r"\bkill myself\b",
    r"\bsuicide\b",
    r"\bchild sexual\b",
    r"\bsteal password\b",
    r"\bphishing\b",
    r"\bcredit card dump\b",
    r"\bhack system\b",
]

PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"reveal your system prompt",
    r"show developer message",
    r"override your rules",
    r"act as unrestricted",
    r"forget your instructions",
]


def input_guard(query: str) -> Tuple[bool, Optional[str]]:
    q = query.strip().lower()

    if not q:
        return False, "Empty transcript received."

    if len(q) > 1000:
        return False, "Query is too long (exceeds 1000 characters)."

    for pat in UNSAFE_PATTERNS:
        if re.search(pat, q):
            return False, "Unsafe query blocked by safety policy."

    for pat in PROMPT_INJECTION_PATTERNS:
        if re.search(pat, q):
            return False, "Prompt-injection attempt detected and blocked."

    return True, None


def retrieval_guard(contexts: List[RetrievedContext], confidence: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    if not contexts:
        return False, "No relevant context found in MSMARCO-XI dataset."

    # NOTE: absolute e5 cosine similarity does NOT generalize as a relevance
    # threshold across corpus sizes - empirically, the "noise floor" score for
    # completely unrelated queries climbed from ~0.70-0.74 on a 20-chunk test
    # corpus to ~0.75-0.84 on the real 4751-chunk MSMARCO-XI index, while genuine
    # matches scored 0.88-0.94. A fixed absolute threshold tuned on one corpus size
    # silently breaks on another. We instead require the top hit to have a
    # meaningful MARGIN over the mean of the tail candidates (see
    # HybridRetriever.confidence_from_dense_hits) - calibrated against 2 known-
    # relevant queries pulled from the indexed corpus (margin 0.076, 0.112) vs 4
    # known off-topic queries (margin 0.018-0.040). min_dense_score is kept only as
    # a sanity backstop against a degenerate/empty index.
    if confidence is None:
        best_dense = max(
            [c.dense_score for c in contexts if c.dense_score is not None],
            default=0.0,
        )
        if best_dense < settings.min_dense_score:
            return False, "The question appears off-topic or unsupported by the dataset."
        return True, None

    if confidence.get("top_dense", 0.0) < settings.min_dense_score:
        return False, "The question appears off-topic or unsupported by the dataset."

    if confidence.get("margin", 0.0) < settings.min_dense_margin:
        return False, "The question appears off-topic or unsupported by the dataset."

    return True, None


def grounding_check(answer: str, contexts: List[RetrievedContext]) -> bool:
    if not answer.strip():
        return False

    context_text = " ".join(c.text.lower() for c in contexts)

    answer_tokens = {
        t for t in _WORD_PATTERN.findall(answer.lower())
        if len(t) >= 2
    }

    if not answer_tokens:
        return False

    supported = sum(1 for t in answer_tokens if t in context_text)
    ratio = supported / max(len(answer_tokens), 1)

    return ratio >= 0.40

