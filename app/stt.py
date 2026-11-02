"""Speech-to-text providers and the rule that picks one.

Three server-side providers plus one in the browser:

* ``sarvam``          Sarvam AI (saarika). Best for Indian languages; needs SARVAM_API_KEY.
* ``whisper_api``     Any OpenAI-compatible ``/audio/transcriptions`` endpoint
                      (OpenAI, Groq, a local whisper.cpp / faster-whisper server).
                      Auto-detects ~100 languages; needs WHISPER_API_KEY (or none
                      for a local server) and WHISPER_API_BASE.
* ``faster_whisper``  Local CTranslate2 Whisper, loaded lazily on first use. No key,
                      no network after the model download. Optional dependency.
* ``browser``         The Web Speech API in the user's browser (web/app.js). The
                      server only ever sees the text, so it needs no provider.

Selection (``settings.stt_provider``):
  * an explicit name forces that provider;
  * ``auto`` (default): an Indian-language hint with a Sarvam key -> Sarvam;
    otherwise Whisper API if configured, then local faster-whisper if installed,
    then Sarvam if keyed. With nothing configured the UI falls back to the browser.

Every provider exposes ``transcribe(audio_bytes, filename, content_type) -> str``
and sets ``last_language`` (BCP-47 code or None) after each call.
"""
from __future__ import annotations

import importlib.util
from typing import List, Optional

import requests

from app.config import settings
from app.langid import base_language

# Languages Sarvam's saarika model covers (BCP-47 base codes).
SARVAM_LANGUAGES = {"hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "en"}
INDIAN_LANGUAGES = SARVAM_LANGUAGES - {"en"}


class STTUnavailable(RuntimeError):
    """No server-side speech-to-text is configured; use the browser recogniser."""


class WhisperAPISTT:
    """OpenAI-compatible transcription endpoint (``response_format=verbose_json``
    returns the detected language)."""

    name = "whisper_api"

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None,
                 model: Optional[str] = None, language_hint: Optional[str] = None):
        self.base_url = (base_url or settings.whisper_api_base).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.whisper_api_key
        self.model = model or settings.whisper_model
        self.language_hint = language_hint
        self.last_language: Optional[str] = None

    def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        data = {"model": self.model, "response_format": "verbose_json"}
        if self.language_hint and self.language_hint != "auto":
            data["language"] = base_language(self.language_hint)
        resp = requests.post(
            f"{self.base_url}/audio/transcriptions",
            headers=headers,
            files={"file": (filename or "audio.webm", audio_bytes, content_type or "audio/webm")},
            data=data,
            timeout=30,
        )
        resp.raise_for_status()
        payload = resp.json()
        self.last_language = _whisper_language_code(payload.get("language"))
        return (payload.get("text") or "").strip()


class FasterWhisperSTT:
    """Local Whisper via faster-whisper (optional dependency, lazy model load)."""

    name = "faster_whisper"
    _model = None

    def __init__(self, model_size: Optional[str] = None, language_hint: Optional[str] = None):
        self.model_size = model_size or settings.faster_whisper_model
        self.language_hint = language_hint
        self.last_language: Optional[str] = None

    @staticmethod
    def available() -> bool:
        return importlib.util.find_spec("faster_whisper") is not None

    def _load(self):
        if FasterWhisperSTT._model is None:
            from faster_whisper import WhisperModel  # optional

            FasterWhisperSTT._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        return FasterWhisperSTT._model

    def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> str:
        import io

        model = self._load()
        lang = base_language(self.language_hint) if self.language_hint and self.language_hint != "auto" else None
        segments, info = model.transcribe(io.BytesIO(audio_bytes), language=lang, vad_filter=True)
        text = " ".join(s.text.strip() for s in segments).strip()
        self.last_language = getattr(info, "language", None)
        return text


# Whisper's verbose_json reports English names ("hindi") or codes ("hi").
_WHISPER_NAMES = {
    "english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "bengali": "bn", "marathi": "mr",
    "gujarati": "gu", "kannada": "kn", "malayalam": "ml", "punjabi": "pa", "urdu": "ur", "arabic": "ar",
    "persian": "fa", "hebrew": "he", "chinese": "zh", "japanese": "ja", "korean": "ko", "russian": "ru",
    "ukrainian": "uk", "spanish": "es", "french": "fr", "german": "de", "portuguese": "pt", "italian": "it",
    "dutch": "nl", "turkish": "tr", "indonesian": "id", "swahili": "sw", "vietnamese": "vi", "thai": "th",
    "greek": "el", "polish": "pl",
}


def _whisper_language_code(lang: Optional[str]) -> Optional[str]:
    if not lang:
        return None
    lang = lang.strip().lower()
    return _WHISPER_NAMES.get(lang, lang if len(lang) <= 3 else None)


def _sarvam_ready() -> bool:
    key = settings.sarvam_api_key
    return bool(key) and key != "your_sarvam_api_key_here"


def _whisper_api_ready() -> bool:
    return bool(settings.whisper_api_base) and (bool(settings.whisper_api_key) or settings.whisper_api_keyless)


def available_providers() -> List[str]:
    """Server-side providers that can run right now, plus 'browser' (always)."""
    out = []
    if _sarvam_ready():
        out.append("sarvam")
    if _whisper_api_ready():
        out.append("whisper_api")
    if FasterWhisperSTT.available():
        out.append("faster_whisper")
    out.append("browser")
    return out


def choose_provider(language_hint: Optional[str] = None, preferred: Optional[str] = None) -> str:
    """Name of the provider to use for one request (see module docstring)."""
    preferred = (preferred or settings.stt_provider or "auto").lower()
    avail = available_providers()
    if preferred != "auto":
        if preferred not in avail:
            raise STTUnavailable(f"STT provider '{preferred}' is not configured.")
        return preferred
    hint = base_language(language_hint) if language_hint and language_hint != "auto" else None
    if hint in INDIAN_LANGUAGES and "sarvam" in avail:
        return "sarvam"
    for name in ("whisper_api", "faster_whisper", "sarvam"):
        if name in avail:
            return name
    return "browser"


def make_stt(language_hint: Optional[str] = None, preferred: Optional[str] = None):
    """Instantiate the chosen server-side provider. Raises STTUnavailable when
    only the browser recogniser is left."""
    name = choose_provider(language_hint, preferred)
    if name == "sarvam":
        from app.stt_sarvam import SarvamSTT

        return SarvamSTT()
    if name == "whisper_api":
        return WhisperAPISTT(language_hint=language_hint)
    if name == "faster_whisper":
        return FasterWhisperSTT(language_hint=language_hint)
    raise STTUnavailable("No server-side speech-to-text is configured. Use the browser microphone "
                         "(Web Speech API) or set SARVAM_API_KEY / WHISPER_API_BASE.")
