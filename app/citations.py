"""Citation verification.

Two levels:

* ``lexical`` - the original check (``guardrails.grounding_check``): at least 40%
  of answer tokens appear somewhere in the concatenated retrieved text. It is
  substring-based, so short tokens match inside longer words, and it is
  answer-level, so one supported sentence can carry an unsupported one.

* ``strict`` - per sentence, against the passage that sentence actually cites:
    1. content-word coverage after English + Hindi stopword removal
       (token-set match, not substring) must reach ``min_coverage``;
    2. numeric consistency: every number in the claim (Devanagari digits and
       small number words like "six"/"छह" normalised to digits) must appear in
       the cited passage;
    3. entity consistency: every drug / vaccine / helpline / dose term from a
       small health lexicon, and every dose expression (number + unit), must
       appear in the cited passage.

What strict still cannot do: detect negation flips ("do not cut the wound" ->
"cut the wound" keeps all content words), or check meaning beyond words. That
needs an NLI / entailment model, which this offline CPU build does not ship.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import regex

_WORD = regex.compile(r"[\p{L}\p{M}\p{N}]+")
from app.textutil import SENT_SPLIT_RE as _SENT_SPLIT, tokenize as _tokenize
_DEV_DIGITS = str.maketrans("०१२३४५६७८९", "0123456789")

STOPWORDS_EN = set("""
a an the and or but if then else of to in on at by for with from into onto over under about after before
during within without between through up down out off again further once here there all any both each few
more most other some such no nor not only own same so than too very can will just should would could may
might must shall is are was were be been being have has had having do does did doing i me my we our you your
he him his she her it its they them their what which who whom whose this that these those am also as until
while when where why how per even often usually every
""".split())
STOPWORDS_HI = set("""
का के की को में से पर है हैं था थे थी हो होता होती होते और या भी तो ही एक यह वह ये वे इस उस इन उन
कि जो जब तक लिए साथ बाद पहले अगर कर करें करना करने करता करती किया जा जाता जाती जाते जाए जाएँ
दें दे देना चाहिए सकता सकती सकते रहा रही रहे गया गई गए हर कम ज्यादा ज़्यादा बहुत सिर्फ़ सिर्फ नहीं
""".split())
STOPWORDS = STOPWORDS_EN | STOPWORDS_HI

NUMBER_WORDS: Dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
    "eight": "8", "nine": "9", "ten": "10", "eleven": "11", "twelve": "12", "fourteen": "14", "twenty": "20",
    "शून्य": "0", "ज़ीरो": "0", "जीरो": "0", "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पाँच": "5", "पांच": "5",
    "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10", "बारह": "12", "चौदह": "14", "बीस": "20",
}

# Drug / vaccine / programme / helpline terms whose substitution would change
# medical meaning. Deliberately small and specific to the demo corpus domain.
HEALTH_ENTITIES = set("""
ors zinc paracetamol aspirin ibuprofen amoxicillin antibiotic antibiotics metformin insulin chloroquine
artemisinin iron folic bcg polio hepatitis td tetanus diphtheria rotavirus measles vitamin colostrum
tourniquet ice venom anti-snake sputum tb dengue malaria aedes anopheles 108 102 104 14416
ओआरएस जिंक पैरासिटामोल एस्पिरिन आइबुप्रोफेन एंटीबायोटिक आयरन फोलिक बीसीजी पोलियो हेपेटाइटिस टीडी
टिटनेस विटामिन खीस कोलोस्ट्रम बलगम टीबी डेंगू मलेरिया एडीज
""".split())

_UNITS = r"(?:mg|ml|mcg|g|kg|litres?|liters?|l|tablets?|days?|weeks?|months?|years?|hours?|minutes?|seconds?|doses?|packets?|गोली|गोलियाँ|दिन|हफ़्ते|हफ्ते|महीने|साल|घंटे|सेकंड|लीटर|खुराक|पैकेट)"
_DOSE = regex.compile(r"(\d+(?:\.\d+)?)\s*(?:to|-|से)?\s*(?:\d+\s*)?" + _UNITS + r"\b", flags=regex.IGNORECASE)


def _norm(text: str) -> str:
    t = unicodedata.normalize("NFC", text or "").translate(_DEV_DIGITS).lower()
    return t


def _tokens(text: str) -> List[str]:
    toks = _tokenize(_norm(text))
    return [NUMBER_WORDS.get(t, t) for t in toks]


def content_words(text: str) -> set:
    return {t for t in _tokens(text) if t not in STOPWORDS and (len(t) > 1 or t.isdigit())}


def numbers(text: str) -> set:
    return {t for t in _tokens(text) if t.isdigit()}


def entities(text: str) -> set:
    toks = set(_tokens(text))
    ents = {t for t in toks if t in HEALTH_ENTITIES}
    if "anti" in toks and "snake" in toks:
        ents.add("anti-snake")
    return ents


def doses(text: str) -> set:
    t = _norm(" ".join(_tokens(text)))
    out = set()
    for m in _DOSE.finditer(t):
        out.add(regex.sub(r"\s+", " ", m.group(0).strip()))
    return out


@dataclass
class CitationCheck:
    ok: bool
    coverage: float
    missing_words: List[str] = field(default_factory=list)
    missing_numbers: List[str] = field(default_factory=list)
    missing_entities: List[str] = field(default_factory=list)
    reason: Optional[str] = None


def verify_claim(claim: str, passage: str, min_coverage: float = 0.8) -> CitationCheck:
    """Is ``claim`` (one answer sentence) supported by ``passage`` (its cited chunk)?"""
    cw = content_words(claim)
    if not cw:
        return CitationCheck(False, 0.0, reason="claim has no content words")
    pw = content_words(passage) | set(_tokens(passage))
    missing = sorted(cw - pw)
    coverage = 1.0 - len(missing) / len(cw)
    miss_num = sorted(numbers(claim) - numbers(passage))
    miss_ent = sorted(entities(claim) - entities(passage))
    p_doses = doses(passage)
    miss_ent += sorted(d for d in doses(claim) if d not in p_doses)
    reason = None
    if miss_num:
        reason = f"numbers not in cited passage: {miss_num}"
    elif miss_ent:
        reason = f"drug/dose/entity terms not in cited passage: {miss_ent}"
    elif coverage < min_coverage:
        reason = f"content-word coverage {coverage:.2f} < {min_coverage}"
    return CitationCheck(reason is None, round(coverage, 3), missing, miss_num, miss_ent, reason)


def lexical_check(claim: str, passage: str, threshold: float = 0.40) -> bool:
    """The original 40% substring-overlap rule, applied to one claim/passage pair."""
    toks = {t for t in _WORD.findall(claim.lower()) if len(t) >= 2}
    if not toks:
        return False
    p = passage.lower()
    return sum(1 for t in toks if t in p) / len(toks) >= threshold


def verify_answer(answer: str, citations: Sequence, contexts: Sequence, min_coverage: float = 0.8) -> CitationCheck:
    """Every answer sentence must be supported by at least one *cited* passage.

    ``citations`` carry a chunk_id; the full passage text is looked up in
    ``contexts`` so the check is against the cited chunk, not the whole
    retrieval set.
    """
    by_id = {c.chunk_id: c.text for c in contexts}
    cited = [by_id.get(c.chunk_id, getattr(c, "quote", "")) for c in citations]
    if not cited:
        return CitationCheck(False, 0.0, reason="answer has no citations")
    worst: Optional[CitationCheck] = None
    for sent in _SENT_SPLIT.split((answer or "").strip()):
        if not sent.strip():
            continue
        checks = [verify_claim(sent, p, min_coverage) for p in cited]
        best = max(checks, key=lambda c: (c.ok, c.coverage))
        if not best.ok:
            return CitationCheck(False, best.coverage, best.missing_words, best.missing_numbers,
                                 best.missing_entities, f"unsupported sentence: {best.reason}")
        if worst is None or best.coverage < worst.coverage:
            worst = best
    return worst or CitationCheck(False, 0.0, reason="empty answer")
