"""Emergency triage: a deterministic red-flag check that runs on every
question, whatever collection it is asked against.

If the question describes a medical emergency (unconscious, not breathing,
chest pain, heavy bleeding, seizure, snake bite, poisoning...) the response
carries an advisory in the asker's language telling them to call emergency
services. It never blocks or replaces the grounded answer.

Keywords cover English, Hindi, Bengali, Tamil, Telugu, Marathi, Urdu, Arabic,
Persian, Chinese, Japanese, Korean, Russian, Ukrainian, Spanish, French,
German, Portuguese, Italian, Indonesian, Turkish, Swahili and Vietnamese.
Text in any other language simply gets no advisory: the check fails open
(no banner) rather than raising. The health demo (app/demo.py) keeps its own
India-specific 108/102 advisory.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

from app.i18n import message
from app.langid import detect_language

_KEYWORDS = [
    # English
    r"\bunconscious\b", r"\bnot breathing\b", r"\bcan'?t breathe\b", r"\bchest pain\b", r"\bseizures?\b",
    r"\bconvulsions?\b", r"\bheavy bleeding\b", r"\bbleeding heavily\b", r"\bsnake ?bite\b", r"\bbitten by a snake\b",
    r"\bheart attack\b", r"\bstroke\b", r"\bpoison(?:ed|ing)\b", r"\boverdose\b", r"\bfainted\b", r"\bchoking\b",
    # Hindi / Marathi (Devanagari)
    "बेहोश", "साँस नहीं", "सांस नहीं", "सीने में दर्द", "दौरा पड़", "ज़हर", "जहर", "साँप ने काट", "सांप ने काट", "छातीत दुखत", "शुद्ध हरपल",
    # Bengali, Tamil, Telugu
    "অজ্ঞান", "শ্বাস নিতে পারছে না", "বুকে ব্যথা", "সাপে কামড়",
    "மயக்கம்", "மூச்சு விட முடியவில்லை", "நெஞ்சு வலி", "பாம்பு கடி",
    "స్పృహ కోల్పో", "ఊపిరి ఆడటం లేదు", "ఛాతీ నొప్పి", "పాము కాటు",
    # Urdu, Arabic, Persian
    "بے ہوش", "سانس نہیں", "سینے میں درد", "سانپ نے کاٹ",
    "فاقد الوعي", "مغمى عليه", "لا يتنفس", "ألم في الصدر", "نزيف حاد", "لدغة ثعبان", "نوبة قلبية", "تسمم",
    "بیهوش", "نفس نمی‌کشد", "درد قفسه سینه", "مار گزیدگی",
    # Chinese, Japanese, Korean
    "昏迷", "失去知觉", "不能呼吸", "无法呼吸", "胸痛", "大出血", "抽搐", "被蛇咬", "心脏病发作", "中毒",
    "意識がない", "息をしていない", "胸の痛み", "大量出血", "けいれん", "蛇に噛まれ", "心臓発作",
    "의식이 없", "숨을 쉬지 않", "숨을 못 쉬", "가슴 통증", "심한 출혈", "발작", "뱀에 물",
    # Russian, Ukrainian
    "без сознания", "не дышит", "боль в груди", "сильное кровотечение", "судорог", "укусила змея", "сердечный приступ", "отравлени",
    "без свідомості", "не дихає", "біль у грудях", "сильна кровотеча",
    # Spanish, French, German, Portuguese, Italian
    r"\binconsciente\b", r"\bno respira\b", r"\bdolor (?:en el|de) pecho\b", r"\bsangrado abundante\b", r"\bconvulsi(?:ón|ones)\b",
    r"\bmordedura de serpiente\b", r"\bme mordió una serpiente\b", r"\bataque al corazón\b", r"\benvenenamiento\b",
    r"\binconscient\b", r"\bne respire plus\b", r"\bdouleur (?:à la|thoracique)\b", r"\bsaignement abondant\b",
    r"\bmorsure de serpent\b", r"\bcrise cardiaque\b", r"\bempoisonnement\b",
    r"\bbewusstlos\b", r"\batmet nicht\b", r"\bbrustschmerzen\b", r"\bstarke blutung\b", r"\bkrampfanfall\b",
    r"\bschlangenbiss\b", r"\bherzinfarkt\b", r"\bvergiftung\b",
    r"\bnão respira\b", r"\bdor no peito\b", r"\bsangramento intenso\b", r"\bpicada de cobra\b", r"\bataque cardíaco\b",
    r"\bincosciente\b", r"\bnon respira\b", r"\bdolore al petto\b", r"\bmorso di serpente\b", r"\binfarto\b",
    # Indonesian, Turkish, Swahili, Vietnamese
    r"\btidak sadarkan diri\b", r"\btidak bernapas\b", r"\bnyeri dada\b", r"\bdigigit ular\b",
    r"\bbilinçsiz\b", r"\bnefes almıyor\b", r"\bgöğüs ağrısı\b", r"\byılan sokması\b", r"\bkalp krizi\b",
    r"\bamepoteza fahamu\b", r"\bhapumui\b", r"\bmaumivu ya kifua\b", r"\bkuumwa na nyoka\b",
    "bất tỉnh", "không thở", "đau ngực", "chảy máu nhiều", "rắn cắn", "đau tim",
]
_RE = re.compile("|".join(_KEYWORDS), flags=re.IGNORECASE)


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").replace("‌", " ").casefold()


def is_emergency(text: str) -> bool:
    try:
        return bool(text) and bool(_RE.search(_norm(text)))
    except Exception:  # never let triage break an answer
        return False


def emergency_advisory(text: str, lang: Optional[str] = None) -> Optional[str]:
    """Localised call-for-help advisory, or None if no red flag is found."""
    if not is_emergency(text):
        return None
    code = lang or detect_language(text).code
    return message("emergency", code)
