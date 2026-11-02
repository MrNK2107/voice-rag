import hashlib
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


from app.textutil import SENT_SPLIT_RE as _SENT_SPLIT, has_unspaced_script


@dataclass
class RawDoc:
    doc_id: str
    text: str
    title: str = ""
    language: str = ""
    source_type: str = ""
    query: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    payload: Dict[str, Any]


def normalize_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def stable_uuid(value: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, value))


def split_sentences(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    parts = _SENT_SPLIT.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 0]


def words(text: str) -> List[str]:
    return normalize_text(text).split()


def make_payload(doc: RawDoc, strategy: str, extra: Optional[Dict] = None) -> Dict:
    payload = {
        "parent_doc_id": doc.doc_id,
        "title": doc.title,
        "language": doc.language,
        "source_type": doc.source_type,
        "chunk_strategy": strategy,
        **doc.metadata,
    }
    if extra:
        payload.update(extra)
    return payload


def add_chunk(chunks: List[Chunk], doc: RawDoc, text: str, strategy: str, extra: Optional[Dict] = None):
    text = normalize_text(text)
    if len(text) < (10 if has_unspaced_script(text) else 30):
        return

    key = f"{doc.doc_id}|{strategy}|{text}"
    chunk_id = stable_uuid(key)

    payload = make_payload(doc, strategy, extra)
    chunks.append(
        Chunk(
            chunk_id=chunk_id,
            text=text,
            payload=payload,
        )
    )


def sliding_word_chunks(doc: RawDoc, size: int, overlap: int, strategy: str) -> List[Chunk]:
    ws = words(doc.text)
    chunks: List[Chunk] = []

    if len(ws) <= size:
        add_chunk(chunks, doc, doc.text, strategy, {"start_word": 0, "end_word": len(ws)})
        return chunks

    step = max(1, size - overlap)
    for start in range(0, len(ws), step):
        end = min(len(ws), start + size)
        text = " ".join(ws[start:end])
        if doc.title:
            text = f"Title: {doc.title}\n{text}"
        add_chunk(chunks, doc, text, strategy, {"start_word": start, "end_word": end})
        if end == len(ws):
            break

    return chunks


def sentence_group_chunks(
    doc: RawDoc,
    target_words: int = 140,
    overlap_sentences: int = 1,
) -> List[Chunk]:
    sents = split_sentences(doc.text)
    chunks: List[Chunk] = []

    current: List[str] = []
    current_len = 0
    chunk_index = 0

    i = 0
    while i < len(sents):
        sent = sents[i]
        sent_len = len(words(sent))

        if current and current_len + sent_len > target_words:
            text = " ".join(current)
            if doc.title:
                text = f"Title: {doc.title}\n{text}"
            add_chunk(
                chunks,
                doc,
                text,
                "sentence_group_140w",
                {"chunk_index": chunk_index},
            )
            chunk_index += 1

            if overlap_sentences > 0:
                current = current[-overlap_sentences:]
                current_len = sum(len(words(s)) for s in current)
            else:
                current = []
                current_len = 0

        current.append(sent)
        current_len += sent_len
        i += 1

    if current:
        text = " ".join(current)
        if doc.title:
            text = f"Title: {doc.title}\n{text}"
        add_chunk(
            chunks,
            doc,
            text,
            "sentence_group_140w",
            {"chunk_index": chunk_index},
        )

    return chunks


def char_group_chunks(doc: RawDoc, target_chars: int = 300) -> List[Chunk]:
    """Chunker for Chinese / Japanese / Thai text, which has no spaces, so the
    word-window strategies below would see one giant "word". Groups whole
    sentences up to ``target_chars`` characters, with one sentence overlap."""
    sents = split_sentences(doc.text)
    chunks: List[Chunk] = []
    current: List[str] = []
    for sent in sents:
        if current and sum(len(s) for s in current) + len(sent) > target_chars:
            text = "".join(current) if has_unspaced_script(current[-1]) else " ".join(current)
            add_chunk(chunks, doc, text, "char_group_300c", {"chunk_index": len(chunks)})
            current = current[-1:]
        current.append(sent)
    if current:
        text = "".join(current) if has_unspaced_script(current[-1]) else " ".join(current)
        add_chunk(chunks, doc, text, "char_group_300c", {"chunk_index": len(chunks)})
    return chunks


def make_chunks(doc: RawDoc) -> List[Chunk]:
    doc.text = normalize_text(doc.text)
    chunks: List[Chunk] = []
    if has_unspaced_script(doc.text) and len(words(doc.text)) * 8 < len(doc.text):
        # Mostly unspaced script (CJK / Thai): character-based chunks.
        if len(doc.text) <= 300:
            text = f"Title: {doc.title} {doc.text}" if doc.title else doc.text
            add_chunk(chunks, doc, text, "atomic_short_passage")
            return chunks
        return char_group_chunks(doc)
    n_words = len(words(doc.text))

    if n_words == 0:
        return []

    # 1. Atomic short passage
    if n_words <= 140:
        text = doc.text
        if doc.title:
            text = f"Title: {doc.title}\n{text}"
        add_chunk(chunks, doc, text, "atomic_short_passage")

    # 2. Metadata-aware title/intro chunk
    if doc.title and n_words > 80:
        intro = " ".join(words(doc.text)[:180])
        add_chunk(
            chunks,
            doc,
            f"Title: {doc.title}\nIntro: {intro}",
            "metadata_title_intro",
        )

    # 3. Query-passage fused chunk
    if doc.query and doc.source_type in {"positive_passages", "passages", "contexts", "documents", "positive"}:
        evidence = " ".join(words(doc.text)[:260])
        qa_text = f"Question: {doc.query}\nRelevant evidence: {evidence}"
        add_chunk(chunks, doc, qa_text, "qa_fused")

    # 4. Sentence-boundary semantic chunks
    if n_words > 90:
        chunks.extend(sentence_group_chunks(doc, target_words=140, overlap_sentences=1))

    # 5. Sliding windows
    if n_words > 120:
        chunks.extend(sliding_word_chunks(doc, size=80, overlap=20, strategy="micro_80w_20o"))
        chunks.extend(sliding_word_chunks(doc, size=180, overlap=40, strategy="standard_180w_40o"))

    # 6. Macro window only for long docs
    if n_words > 450:
        chunks.extend(sliding_word_chunks(doc, size=420, overlap=80, strategy="macro_420w_80o"))

    # Deduplicate by normalized text
    seen = set()
    deduped = []

    for c in chunks:
        h = hashlib.sha256(normalize_text(c.text).lower().encode("utf-8")).hexdigest()
        if h not in seen:
            seen.add(h)
            deduped.append(c)

    return deduped
