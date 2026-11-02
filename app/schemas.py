from typing import Optional
from pydantic import BaseModel, Field


class Citation(BaseModel):
    chunk_id: str
    score: float
    strategy: Optional[str] = None
    language: Optional[str] = None
    quote: str
    title: Optional[str] = None
    parent_doc_id: Optional[str] = None
    # The full cited passage (the UI highlights `quote` inside it) and where it came from.
    passage: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None


class RetrievedContext(BaseModel):
    chunk_id: str
    text: str
    score: float
    dense_score: Optional[float] = None
    lexical_score: Optional[float] = None
    strategy: Optional[str] = None
    language: Optional[str] = None
    parent_doc_id: Optional[str] = None
    title: Optional[str] = None
    source: Optional[str] = None
    page: Optional[int] = None


class RagResponse(BaseModel):
    transcript: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = True
    abstained: bool = False
    abstain_reason: Optional[str] = None
    # Set by the health demo's red-flag triage step (e.g. "call 108"); None otherwise.
    advisory: Optional[str] = None
    # Retrieved chunks dropped by the passage injection scan (chunk_id:pattern).
    quarantined_chunks: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    # Language of the question (BCP-47, from app/langid.py or the STT provider),
    # whether the UI should render it right-to-left, and the language of the
    # quoted answer (the source document's language unless translated).
    language: Optional[str] = None
    language_name: Optional[str] = None
    rtl: bool = False
    answer_language: Optional[str] = None
    translated: bool = False
    original_answer: Optional[str] = None
    collection: Optional[str] = None
    stt_provider: Optional[str] = None


class TextRequest(BaseModel):
    query: str
    collection: Optional[str] = None
    # Optional language hint from the UI ("auto" or a BCP-47 code).
    language: Optional[str] = None
    channel: str = "text"
