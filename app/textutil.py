"""Script-aware tokenisation and sentence splitting shared by the retriever,
the offline embedder and the extractive generator.

For space-separated scripts (Latin, Cyrillic, Arabic, Devanagari, Tamil...) a
token is a run of letters + combining marks + digits (``\\p{L}\\p{M}\\p{N}``; stdlib
``re`` shreds Indic conjuncts, see app/generator.py).

Chinese, Japanese and Thai are written without spaces, so a "word" run there
is a whole clause. Those runs are split into overlapping character bigrams
(the usual dictionary-free baseline for CJK retrieval). For non-CJK text the
output is exactly what the old ``\\p{L}\\p{M}\\p{N}`` tokenizer produced.
"""
from __future__ import annotations

from typing import List

import regex

WORD_RE = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)
# Scripts written without spaces between words.
_UNSPACED_RE = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}ー]+")
_UNSPACED_CHAR = regex.compile(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}\p{Lao}\p{Khmer}\p{Myanmar}ー]")

# Sentence ends: Latin/Cyrillic . ! ?, Devanagari danda, Arabic/Urdu question
# mark and full stop (followed by whitespace), and CJK full-width marks
# (no whitespace needed after them).
SENT_SPLIT_RE = regex.compile(r"(?<=[.!?।॥؟۔])\s+|(?<=[。！？])\s*")


def has_unspaced_script(text: str) -> bool:
    return bool(_UNSPACED_CHAR.search(text or ""))


def _bigrams(run: str) -> List[str]:
    if len(run) == 1:
        return [run]
    return [run[i:i + 2] for i in range(len(run) - 1)]


def tokenize(text: str) -> List[str]:
    """Tokens of ``text`` (not lower-cased). CJK/Thai runs become bigrams."""
    out: List[str] = []
    for tok in WORD_RE.findall(text or ""):
        if not _UNSPACED_CHAR.search(tok):
            out.append(tok)
            continue
        pos = 0
        for m in _UNSPACED_RE.finditer(tok):
            if m.start() > pos:
                out.append(tok[pos:m.start()])
            out.extend(_bigrams(m.group(0)))
            pos = m.end()
        if pos < len(tok):
            out.append(tok[pos:])
    return out


def is_unspaced_token(tok: str) -> bool:
    return bool(_UNSPACED_CHAR.search(tok))


def fts_text(text: str) -> str:
    """Text as stored in the FTS5 index: CJK/Thai characters separated by
    spaces, so the unicode61 tokenizer sees one token per character and a
    bigram can be matched as a two-token phrase."""
    if not has_unspaced_script(text):
        return text
    return _UNSPACED_CHAR.sub(lambda m: f" {m.group(0)} ", text)


def split_sentences(text: str) -> List[str]:
    return [s for s in SENT_SPLIT_RE.split(text or "") if s and s.strip()]
