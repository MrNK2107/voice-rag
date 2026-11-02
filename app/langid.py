"""Lightweight language identification for short questions and passages.

No model download and no required dependency. Two stages:

1. **Script.** Most writing systems map to one language (Tamil, Thai,
   Hangul, Greek...) or a small family that a few letters separate (Arabic vs
   Urdu vs Persian, Russian vs Ukrainian, Japanese kana vs Chinese Han).
2. **Latin-script text** is scored against short stopword lists for common
   languages, plus a few telltale letters (ñ, ß, ç, ã, ğ, ă...). Romanised
   Hindi ("ORS kaise banaye?") is recognised as ``hi-Latn``.

If the optional ``langdetect`` package is installed it is consulted only for
Latin/Cyrillic text that the stopword stage cannot decide; it is never needed
for the tests. Unknown input returns ``und`` (undetermined), never raises.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, asdict
from typing import Dict, Optional

import regex

# Languages written right-to-left (BCP-47 primary subtags).
RTL_LANGUAGES = {"ar", "ur", "fa", "he", "ps", "sd", "ug", "yi", "dv", "ckb"}

LANGUAGE_NAMES: Dict[str, str] = {
    "en": "English", "hi": "Hindi", "hi-Latn": "Hindi (romanised)", "bn": "Bengali", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam", "mr": "Marathi", "gu": "Gujarati",
    "pa": "Punjabi", "or": "Odia", "as": "Assamese", "ne": "Nepali", "si": "Sinhala",
    "ur": "Urdu", "ar": "Arabic", "fa": "Persian", "he": "Hebrew", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "th": "Thai", "lo": "Lao", "km": "Khmer", "my": "Burmese",
    "ru": "Russian", "uk": "Ukrainian", "bg": "Bulgarian", "sr": "Serbian", "el": "Greek",
    "hy": "Armenian", "ka": "Georgian", "am": "Amharic", "es": "Spanish", "fr": "French",
    "de": "German", "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "sv": "Swedish",
    "pl": "Polish", "tr": "Turkish", "vi": "Vietnamese", "id": "Indonesian", "sw": "Swahili",
    "ro": "Romanian", "cs": "Czech", "und": "Unknown",
}

# Unicode script -> default language.
_SCRIPT_LANG = {
    "Devanagari": "hi", "Bengali": "bn", "Tamil": "ta", "Telugu": "te", "Kannada": "kn",
    "Malayalam": "ml", "Gujarati": "gu", "Gurmukhi": "pa", "Oriya": "or", "Sinhala": "si",
    "Arabic": "ar", "Hebrew": "he", "Hangul": "ko", "Thai": "th", "Lao": "lo", "Khmer": "km",
    "Myanmar": "my", "Cyrillic": "ru", "Greek": "el", "Armenian": "hy", "Georgian": "ka",
    "Ethiopic": "am", "Han": "zh", "Hiragana": "ja", "Katakana": "ja", "Latin": "en",
}
_SCRIPTS = list(_SCRIPT_LANG)
_SCRIPT_RE = {s: regex.compile(rf"\p{{Script={s}}}") for s in _SCRIPTS}

# Letters that only Urdu / Persian use within the Arabic script.
_URDU_LETTERS = set("ٹڈڑںےۓھ")
_PERSIAN_LETTERS = set("پچژگ")
_UKRAINIAN_LETTERS = set("іїєґ")
_SERBIAN_LETTERS = set("ђћџљњј")
_BULGARIAN_HINT = set("ъ")  # frequent in Bulgarian, rare in Russian except after prefixes

_WORD = regex.compile(r"[\p{L}\p{M}']+")

# Short, high-frequency function words. Chosen to be distinctive between the
# languages listed; a word may appear in more than one list.
_LATIN_STOPWORDS: Dict[str, set] = {
    "en": set("the is are what how why when where who which of and to in for with does do can you this that it my i as all".split()),
    "es": set("el la los las es son qué que cómo como por para con del una un y en mi cuál cuándo dónde está tiene se".split()),
    "fr": set("le la les est sont quoi que comment pourquoi quand où qui des une un et en du pour avec mon je il c'est est-ce".split()),
    "de": set("der die das ist sind was wie warum wann wo wer und ein eine mit für von zu ich nicht kann ich".split()),
    "pt": set("o a os as é são que como por para com do da uma um e em não meu qual quando onde está você".split()),
    "it": set("il lo la gli le è sono che come perché quando dove chi di una un e con per non mio qual".split()),
    "nl": set("de het is zijn wat hoe waarom wanneer waar wie en een met voor van ik niet kan".split()),
    "sv": set("och är vad hur varför när var vem en ett med för av jag inte kan det".split()),
    "pl": set("jest są co jak dlaczego kiedy gdzie kto i w z na nie się to czy".split()),
    "tr": set("bir ve bu ne nasıl neden nerede kim ile için mi mı değil var".split()),
    "id": set("apa bagaimana mengapa kapan di mana siapa dan yang dengan untuk ini itu tidak adalah".split()),
    "sw": set("ni na ya wa kwa nini vipi kwa nini lini wapi nani hii hiyo si katika".split()),
    "vi": set("là và của có không gì thế nào tại sao khi đâu ai này cho với".split()),
    "ro": set("este sunt ce cum de ce când unde cine și cu pentru un o nu".split()),
    "cs": set("je jsou co jak proč kdy kde kdo a s na pro se není".split()),
    # Romanised Hindi / Hinglish: grammar words that English does not use.
    "hi-Latn": set("kya kaise kaisa kab kahan kahaan kitna kitne kitni kyun kyon kaun hai hain mein ka ki ke ko se aur nahi nahin karein kare chahiye banaye batao".split()),
}
_LATIN_CHAR_HINTS = {
    "es": "ñ¿¡", "de": "ßäöü", "fr": "çœèêëàâîôû", "pt": "ãõç", "tr": "ğışİ", "ro": "ășț",
    "pl": "ąęłńśźż", "cs": "řůěš", "vi": "ơưđạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ",
    "sv": "å",
}


@dataclass
class LanguageResult:
    code: str
    name: str
    script: str
    confidence: float
    rtl: bool

    def dict(self) -> dict:
        return asdict(self)


def _script_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for ch in text:
        if not (ch.isalpha() or unicodedata.category(ch).startswith("M")):
            continue
        for s in _SCRIPTS:
            if _SCRIPT_RE[s].match(ch):
                counts[s] = counts.get(s, 0) + 1
                break
    return counts


def dominant_script(text: str) -> Optional[str]:
    counts = _script_counts(text or "")
    if not counts:
        return None
    # Japanese text mixes Han with kana: any kana means Japanese.
    kana = counts.get("Hiragana", 0) + counts.get("Katakana", 0)
    if kana and counts.get("Han", 0):
        return "Hiragana" if counts.get("Hiragana", 0) >= counts.get("Katakana", 0) else "Katakana"
    return max(counts, key=counts.get)


def _latin_language(text: str) -> tuple[str, float]:
    lower = text.lower()
    words = _WORD.findall(lower)
    if not words:
        return "und", 0.0
    scores = {lang: 0.0 for lang in _LATIN_STOPWORDS}
    for w in words:
        for lang, sw in _LATIN_STOPWORDS.items():
            if w in sw:
                scores[lang] += 1.0
    for lang, chars in _LATIN_CHAR_HINTS.items():
        hits = sum(lower.count(c) for c in chars)
        if hits:
            scores[lang] += 1.5 * min(hits, 3)
    best = max(scores, key=scores.get)
    ranked = sorted(scores.values(), reverse=True)
    top, second = ranked[0], ranked[1]
    if top == 0:
        return "und", 0.0
    confidence = round(min(1.0, (top - second + 0.5) / (len(words) ** 0.5 + 1.0)), 2)
    if top == second:
        # A tie involving English goes to English (most common in mixed text).
        tied = [l for l, s in scores.items() if s == top]
        best = "en" if "en" in tied else tied[0]
        confidence = min(confidence, 0.3)
    if top < 2:
        # One short function word ("as", "da", "un") is weak evidence.
        confidence = min(confidence, 0.3)
    return best, max(confidence, 0.2)


def _langdetect_fallback(text: str) -> Optional[str]:
    try:  # optional dependency
        from langdetect import DetectorFactory, detect

        DetectorFactory.seed = 0
        return detect(text)
    except Exception:
        return None


def detect_language(text: str, use_fallback: bool = True) -> LanguageResult:
    """Best-effort language of ``text``. Never raises."""
    text = text or ""
    script = dominant_script(text)
    if script is None:
        return LanguageResult("und", LANGUAGE_NAMES["und"], "Unknown", 0.0, False)

    code = _SCRIPT_LANG.get(script, "und")
    confidence = 0.9
    chars = set(text)
    if script == "Arabic":
        if chars & _URDU_LETTERS:
            code = "ur"
        elif chars & _PERSIAN_LETTERS:
            code = "fa"
    elif script == "Cyrillic":
        if chars & _UKRAINIAN_LETTERS:
            code = "uk"
        elif chars & _SERBIAN_LETTERS:
            code = "sr"
        elif chars & _BULGARIAN_HINT and "ы" not in chars and "э" not in chars:
            code = "bg"
        confidence = 0.7
    elif script == "Devanagari":
        confidence = 0.8  # also Marathi / Nepali; Hindi is the default
    elif script == "Latin":
        code, confidence = _latin_language(text)
        if (code == "und" or confidence < 0.35) and use_fallback and len(text) >= 20:
            fb = _langdetect_fallback(text)
            if fb and fb in LANGUAGE_NAMES:
                code, confidence = fb, 0.5
        if code == "und":
            code, confidence = "en", 0.2

    return LanguageResult(
        code=code,
        name=LANGUAGE_NAMES.get(code, code),
        script=script,
        confidence=float(confidence),
        rtl=base_language(code) in RTL_LANGUAGES,
    )


def base_language(code: Optional[str]) -> str:
    """'hi-Latn' -> 'hi', 'pt-BR' -> 'pt', None -> 'und'."""
    if not code:
        return "und"
    return code.split("-")[0].split("_")[0].lower()


def is_rtl(code: Optional[str]) -> bool:
    return base_language(code) in RTL_LANGUAGES


# ISO 639-3 tags used by MSMARCO-XI / older payloads -> BCP-47.
_ISO3 = {"eng": "en", "hin": "hi", "tam": "ta", "tel": "te", "ben": "bn", "mar": "mr", "guj": "gu",
         "kan": "kn", "mal": "ml", "pan": "pa", "urd": "ur", "ori": "or", "asm": "as"}


def normalize_code(code: Optional[str]) -> str:
    if not code:
        return "und"
    c = code.strip()
    return _ISO3.get(c.lower(), c)
