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

    embed_model: str = "intfloat/multilingual-e5-small"
    sqlite_fts_path: str = os.environ.get("SQLITE_FTS_PATH", "storage/chunks.sqlite")

    generation_mode: str = "extractive"

    min_dense_score: float = 0.50  # loose backstop only; see retrieval_guard margin check
    min_dense_margin: float = 0.055
    max_context_chars: int = 1800
    stt_cache_enabled: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
