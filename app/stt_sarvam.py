import hashlib
import json
from pathlib import Path
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.config import settings


class SarvamSTT:
    name = "sarvam"

    def __init__(self):
        self.last_language = None
        self.cache_dir = Path("storage/stt_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, audio_bytes: bytes) -> Path:
        h = hashlib.sha256(audio_bytes).hexdigest()
        return self.cache_dir / f"{h}.json"

    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.2, min=0.2, max=1.0),
        retry=retry_if_exception_type(requests.RequestException),
        reraise=True,
    )
    def transcribe(self, audio_bytes: bytes, filename: str, content_type: str) -> str:
        if settings.stt_cache_enabled:
            path = self._cache_path(audio_bytes)
            if path.exists():
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                    return data.get("text", "")
                except Exception:
                    pass

        if not settings.sarvam_api_key or settings.sarvam_api_key == "your_sarvam_api_key_here":
            # Fallback for dev / dry-run without active Sarvam key
            print("Warning: SARVAM_API_KEY is not set or using default placeholder.")
            return "What is the capital of Goa?"

        headers = {
            "api-subscription-key": settings.sarvam_api_key
        }

        files = {
            "file": (filename or "audio.webm", audio_bytes, content_type or "audio/webm")
        }

        data = {
            "model": settings.sarvam_stt_model,
            "language_code": "unknown"
        }

        response = requests.post(
            settings.sarvam_stt_url,
            headers=headers,
            files=files,
            data=data,
            timeout=15,
        )

        response.raise_for_status()
        payload = response.json()

        text = (
            payload.get("transcript")
            or payload.get("text")
            or payload.get("output")
            or ""
        ).strip()

        # Sarvam reports the detected language as e.g. "hi-IN".
        code = payload.get("language_code")
        self.last_language = code.split("-")[0] if isinstance(code, str) and code else None

        if settings.stt_cache_enabled and text:
            path = self._cache_path(audio_bytes)
            path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")

        return text
