"""Optional translation of a verified answer into the asker's language.

Off by default. Vouch's answers are verbatim quotes, checked sentence by
sentence against the cited passage (app/citations.py) and by the output guard
(app/safety.py) *before* any translation. When ``TRANSLATE_ANSWERS=true`` and
an OpenAI-compatible chat endpoint is configured, the checked quote is then
translated; the response keeps the original quote in ``original_answer`` and
in every citation, and sets ``translated=true`` so the UI labels it.
"""
from __future__ import annotations

from typing import Optional, Protocol

import requests

from app.config import settings
from app.langid import LANGUAGE_NAMES, base_language


class Translator(Protocol):
    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str: ...


class OpenAICompatibleTranslator:
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, model: Optional[str] = None):
        self.base_url = (base_url or settings.translate_api_base).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.translate_api_key
        self.model = model or settings.translate_model

    def translate(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        target = LANGUAGE_NAMES.get(target_lang, LANGUAGE_NAMES.get(base_language(target_lang), target_lang))
        resp = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"} if self.api_key else {},
            json={
                "model": self.model,
                "temperature": 0,
                "messages": [
                    {"role": "system", "content": "Translate the user's text faithfully. Do not add, remove or "
                                                  "explain anything. Keep numbers, names and units unchanged. "
                                                  "Output only the translation."},
                    {"role": "user", "content": f"Target language: {target}\n\n{text}"},
                ],
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def translator_from_settings() -> Optional[Translator]:
    if settings.translate_answers and settings.translate_api_base:
        return OpenAICompatibleTranslator()
    return None
