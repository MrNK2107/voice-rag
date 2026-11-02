import os
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    sarvam_api_key: str = ""
    sarvam_stt_url: str = "https://api.sarvam.ai/speech-to-text"
    sarvam_stt_model: str = "saarika:v2.5"

    qdrant_url: str = "http://localhost:6333"
    qdrant_path: str = os.environ.get("QDRANT_PATH", "storage/qdrant_db")
    qdrant_api_key: str | None = None
    qdrant_collection: str = "msmarco_xi_chunks"

    # Multilingual by default (~100 languages, cross-lingual retrieval). For
    # stronger cross-language matching use "BAAI/bge-m3" (larger download).
    embed_model: str = "intfloat/multilingual-e5-small"
    sqlite_fts_path: str = os.environ.get("SQLITE_FTS_PATH", "storage/chunks.sqlite")

    generation_mode: str = "extractive"
    corpus_name: str = "MSMARCO-XI dataset"

    min_dense_score: float = 0.50  # loose backstop only; see retrieval_guard margin check
    min_dense_margin: float = 0.055
    max_context_chars: int = 1800
    stt_cache_enabled: bool = False
    # "strict": per-sentence citation check (stopword-filtered coverage + numbers
    # + drug/dose terms must be in the cited passage). "lexical": only the old
    # answer-level 40% token-overlap check. See app/citations.py.
    citation_check: str = "strict"
    citation_min_coverage: float = 0.8
    scan_retrieved_passages: bool = True
    output_guard_enabled: bool = True
    # Learned second-stage input detector (app/injection_clf.py). Input only:
    # retrieved passages are scanned by patterns alone (see GUARDRAILS.md).
    injection_classifier_enabled: bool = True
    # Abstain when the extractive answer contains less than this share of the
    # question's content words (app/generator.py query_coverage). 0 = off.
    # Demo mode sets 0.6 (app/demo.py); it is off for the MSMARCO index until
    # it has been calibrated there.
    min_query_coverage: float = 0.0

    # ---- Product-level settings (Vouch) ----
    product_name: str = "Vouch"
    # If set, collection writes and insights need the X-Admin-Token header.
    admin_token: str = ""
    # Named collections of the user's own documents (app/collections.py).
    collections_dir: str = os.environ.get("VOUCH_COLLECTIONS_DIR", "storage/collections")
    max_upload_mb: int = 20
    # Speech-to-text: auto | sarvam | whisper_api | faster_whisper (app/stt.py).
    stt_provider: str = "auto"
    whisper_api_base: str = ""          # e.g. https://api.openai.com/v1 or http://localhost:8080/v1
    whisper_api_key: str = ""
    whisper_api_keyless: bool = False   # true for a local server that needs no key
    whisper_model: str = "whisper-1"
    faster_whisper_model: str = "small"
    # Emergency red-flag advisory on every question (app/triage.py).
    emergency_triage: bool = True
    # Optional translation of the quoted answer into the asker's language
    # (app/translate.py). Off by default: answers stay verbatim quotes.
    translate_answers: bool = False
    translate_api_base: str = ""
    translate_api_key: str = ""
    translate_model: str = "gpt-4o-mini"
    # Question log for the Insights view (app/analytics.py).
    analytics_enabled: bool = True
    analytics_path: str = os.environ.get("VOUCH_ANALYTICS_PATH", "storage/analytics.sqlite")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
