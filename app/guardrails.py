import re
import regex
from typing import List, Optional, Tuple

from app.config import settings
from app.injection_clf import classify_injection
from app.langid import detect_language
from app.safety import detect_injection, normalize_text
from app.schemas import RetrievedContext
from app.textutil import split_sentences

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

# Kept for reference / backwards compatibility. The live check is
# app.safety.detect_injection, which normalises text first (case, NFKC,
# zero-width, look-alike letters, leetspeak) and has a much broader English,
# Hindi and Hinglish pattern set. These 7 phrases are all still caught by it.
PROMPT_INJECTION_PATTERNS = [
    r"ignore previous instructions",
    r"ignore all instructions",
    r"reveal your system prompt",
    r"show developer message",
    r"override your rules",
    r"act as unrestricted",
    r"forget your instructions",
]


# Languages the learned injection classifier was trained on.
_CLASSIFIER_LANGS = {"en", "hi", "hi-Latn", "ta", "und"}


def classifier_applies(text: str) -> bool:
    """False for text confidently in a Latin-script language the classifier
    never saw (Spanish, French, German...). Other scripts keep the classifier:
    unseen character n-grams only lower its score, so it can't add false
    positives there."""
    lang = detect_language(text, use_fallback=False)
    if lang.script != "Latin":
        return True
    return lang.code in _CLASSIFIER_LANGS or lang.confidence < 0.4


def input_guard(query: str) -> Tuple[bool, Optional[str]]:
    q = query.strip().lower()

    # Length first: normalising a megabyte of text costs seconds of CPU.
    if len(q) > 1000:
        return False, "Query is too long (exceeds 1000 characters)."

    if not normalize_text(q):
        return False, "Empty transcript received."

    q_norm = normalize_text(q)
    for pat in UNSAFE_PATTERNS:
        if re.search(pat, q) or re.search(pat, q_norm):
            return False, "Unsafe query blocked by safety policy."

    if detect_injection(query):
        return False, "Prompt-injection attempt detected and blocked."

    # Second stage: small learned classifier, only adds blocks (pattern OR clf).
    # It was trained on English, Hindi, Hinglish, Tamil and Tanglish only; on
    # other Latin-script languages its scores drift towards the threshold
    # (a benign French question scored 0.55 vs 0.59), so those rely on the
    # pattern stage, which has per-language rules (app/safety.py WORLD_PATTERNS).
    if (settings.injection_classifier_enabled and classifier_applies(query)
            and classify_injection(query)):
        return False, "Prompt-injection attempt detected and blocked."

    # Also score each sentence on its own: a benign sentence (or one in a
    # language the classifier skips) in front of an attack would otherwise
    # dilute its score or switch the stage off for the whole query.
    if settings.injection_classifier_enabled:
        sentences = [s for s in split_sentences(query) if s.strip()]
        if len(sentences) > 1:
            for sent in sentences:
                if classifier_applies(sent) and classify_injection(sent):
                    return False, "Prompt-injection attempt detected and blocked."

    return True, None


def retrieval_guard(contexts: List[RetrievedContext], confidence: Optional[dict] = None) -> Tuple[bool, Optional[str]]:
    if not contexts:
        return False, f"No relevant context found in the {settings.corpus_name}."

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

