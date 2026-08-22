# voice-rag — Project Context

## Overview

A voice-enabled Retrieval-Augmented Generation (RAG) system built for the HH Goa 2026 hackathon (Shortlisting Task 2). The system transcribes spoken input using Sarvam Speech-to-Text, retrieves relevant document chunks via hybrid dense (Qdrant) and lexical (SQLite FTS5) search across multi-strategy chunking, and produces grounded answers with verified citations — targeting sub-200ms post-transcription latency.

- **Backend**: FastAPI (Python 3.10+)
- **STT**: Sarvam Speech-to-Text (`saarika:v2.5`)
- **Dataset**: `ai4bharat/MSMARCO-XI` (multilingual, streamed from parquet via `huggingface_hub`)
- **Embeddings**: `intfloat/multilingual-e5-small`
- **Vector DB**: Qdrant (Cosine distance, HNSW `ef_search=32`)
- **Lexical DB**: SQLite FTS5 (BM25 score)
- **Fusion**: Reciprocal Rank Fusion (RRF) + Parent Doc / Strategy Diversity Filter
- **Answer Generation**: Fast Extractive Engine (<10ms)
- **Frontend**: Vanilla HTML/CSS/JS MediaRecorder UI (`web/index.html`)

## Repository Layout

```
voice-rag/
├── app/
│   ├── __init__.py           Package init
│   ├── main.py               FastAPI app: /health, /api/ask-text, /api/ask-audio
│   ├── config.py             Settings (pydantic-settings, env-driven)
│   ├── schemas.py            Pydantic request/response models
│   ├── harness.py            VoiceRAGHarness — orchestrates the whole pipeline
│   ├── stt_sarvam.py         Sarvam STT client (retries, dev fallback if no key)
│   ├── chunking.py           Multi-strategy adaptive chunker
│   ├── dataset_loader.py     Parses raw MSMARCO-XI rows into RawDoc objects
│   ├── retriever.py          Hybrid dense+lexical retrieval, RRF fusion, confidence signal
│   ├── generator.py          Extractive grounded answer generation
│   ├── guardrails.py         Input guard, retrieval guard, grounding check
│   └── latency.py            Stage-timing context manager
├── scripts/
│   ├── explore_dataset.py     Inspect the real dataset schema
│   ├── build_index.py         Index MSMARCO-XI into Qdrant + SQLite FTS5
│   ├── make_benchmark_queries.py  Generates storage/benchmark_queries.json
│   └── benchmark.py            Runs the pipeline over the query set, computes P50/P70/P100
├── web/
│   └── index.html             Minimal MediaRecorder UI
├── storage/
│   ├── .gitkeep
│   ├── benchmark_queries.json  30 test queries (15 relevant + 15 off-topic/unsafe/injection)
│   └── benchmark_results.json  Empirical latency metrics
├── Dockerfile                 Python 3.10-slim, installs deps, runs uvicorn
├── docker-compose.yml         Qdrant container (ports 6333/6334)
├── requirements.txt
├── .env.example
├── .gitignore                 .env, __pycache__, *.pyc, storage/*.sqlite, storage/qdrant_db/, storage/stt_cache/
├── README.md                  Quickstart guide, architecture, tech stack, known issues
├── PROJECT_LOG.md             Full build journey log (chronological)
├── CHUNKING.md                Multi-strategy chunking spec
├── GUARDRAILS.md              Guardrails & safety system spec
└── LATENCY.md                 Latency analytics & optimization strategy
```

## Architecture (Pipeline)

```
User Voice
  |
  v
Browser MediaRecorder UI (web/index.html)
  |
  v
FastAPI POST /api/ask-audio (app/main.py)
  |
  v
Sarvam Speech-to-Text (app/stt_sarvam.py) — retries, dev fallback
  |
  v
Input Guard (app/guardrails.py: input_guard)
  |  - unsafe-content patterns (bomb, suicide, phishing, etc.)
  |  - prompt-injection patterns (ignore instructions, reveal prompt, etc.)
  |  - empty/oversized transcript
  v
Hybrid Retriever (app/retriever.py)
  |--- Qdrant dense search (e5-small embeddings, hnsw_ef=32)
  |--- SQLite FTS5 lexical search (BM25, unicode61 tokenizer)
  |--- Reciprocal Rank Fusion (k=60) + parent-doc/strategy diversity filter
  v
Retrieval Confidence Guard (app/guardrails.py: retrieval_guard)
  |  - margin-based: top hit vs. mean of tail candidates (ranks 10-40)
  |  - threshold: MIN_DENSE_MARGIN=0.055 (calibrated on real data)
  v
Extractive Answer Generator (app/generator.py) — <2ms
  |  - term-overlap sentence selection from retrieved chunks only
  v
Grounding / Hallucination Guard (app/guardrails.py: grounding_check)
  |  - >=40% of answer tokens must exist in retrieved evidence
  v
Structured JSON Response (RagResponse schema)
```

All orchestration is handled by `VoiceRAGHarness` (`app/harness.py`), which owns retries on the STT call, stage-by-stage timing instrumentation (`app/latency.py`), and structured fallback responses (every failure mode returns a proper `RagResponse` with `abstain_reason`).

## Key Design Decisions

### 1. Extractive, Not Generative LLM
The 200ms budget is for everything *after* STT. An LLM API call would blow that budget by itself (800ms-2500ms). Extractive generation — picking the best-matching sentences straight out of retrieved chunks — runs in <2ms and is trivially grounded (only says things literally in retrieved text). Trade-off: answer fluency, acceptable for this task's latency target.

### 2. Margin-Based Retrieval Guard
Absolute e5 cosine similarity doesn't generalize as a relevance threshold across corpus sizes. The "noise floor" for unrelated queries climbs as the corpus grows (0.70-0.74 on 20 chunks → 0.75-0.84 on 4,751 chunks). The guard instead requires the top hit to stand out above the mean of tail candidates (ranks 10-40), which is relative and generalizes.

### 3. Unicode-Aware Tokenization
Stdlib `re`'s `\w` excludes combining marks (Mn/Mc Unicode categories), shredding Devanagari conjuncts ("कॉर्पोरेशन" → single chars). Fixed by using the `regex` package's `[\p{L}\p{M}\p{N}]` pattern in both `generator.py` (term-overlap scoring) and `retriever.py` (FTS5 query construction).

### 4. Direct Parquet Streaming
`ai4bharat/MSMARCO-XI` ships per-language parquet files behind a legacy `datasets` loading script that hangs indefinitely on modern library versions. `build_index.py` streams parquet files directly via `huggingface_hub.HfFileSystem` + `pyarrow` with column projection, bypassing the broken script entirely.

### 5. Multi-Strategy Chunking
6 chunking strategies (`atomic_short_passage`, `metadata_title_intro`, `qa_fused`, `sentence_group_140w`, `micro_80w_20o`, `standard_180w_40o`, `macro_420w_80o`) with SHA256-based deduplication and deterministic UUID-v5 chunk IDs.

## Critical Bugs Fixed (per PROJECT_LOG.md)

| # | Bug | Impact | Fix |
|---|---|---|---|
| 1 | Placeholder dataset (10 hand-written chunks, not real MSMARCO-XI) | Entire retrieval was meaningless | Rewrote `build_index.py` + `dataset_loader.py` to stream parquet files directly |
| 2 | Placeholder Sarvam API key → hardcoded fake transcript | STT was never real | Real key added, verified with live `200 OK` transcription |
| 3 | Retrieval guard used absolute threshold (0.80) calibrated on 20 chunks | Broke on real 4,751-chunk corpus | Replaced with margin-based guard |
| 4 | Devanagari tokenization via stdlib `re.\w+` shredded conjuncts | Corrupted term-overlap + FTS5 queries for all Hindi text | Switched to `regex` package's `[\p{L}\p{M}\p{N}]` |
| 5 | Stale data bug in index rebuild | Old chunks accumulated across rebuilds | Explicit `DROP TABLE` + `delete_collection` before recreate |

## Benchmark Results

Empirical measurement across 30 queries (15 real + 15 off-topic/unsafe/injection) on the real 4,751-chunk Hindi index:

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---|---|---|
| Input Guard | 0.03 | 0.03 | 0.04 |
| Dense + Lexical Search | 37.13 | 49.47 | 108.09 |
| Retrieval Guard | 0.00 | 0.00 | 0.01 |
| Grounded Answer Generator | 0.47 | 0.80 | 1.77 |
| Grounding Validator | 0.07 | 0.08 | 0.28 |
| **Post-STT RAG Total** | **36.65** | **46.75** | **108.8** |

- **Correctness**: 29/30 (96.7%) correct abstain-vs-answer decisions
- All 15 off-topic/unsafe/injection queries correctly abstained
- 14/15 real queries got correctly grounded answers
- P100 latency is ~108.8ms — well under the 200ms target with ~4x headroom
- Cloud STT (Sarvam) measured separately at ~1,226ms (external API round trip, not part of the 200ms budget)

## Current Status

All blocking issues resolved and verified against real data. The system is functional and ready for submission. See "Recommended Before Final Submission" in PROJECT_LOG.md for remaining items (additional languages, real speech testing, demo videos).

## Git Configuration

- **Remote**: `https://github.com/MrNK2107/voice-rag.git`
- **Branch**: `main`
- **Atomic commit settings**: `commit.verbose=true`, `commit.cleanup=extended`, `push.default=simple`, `pull.rebase=false`

## Quick Setup

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in SARVAM_API_KEY, HF_TOKEN
docker compose up -d    # start Qdrant
export HF_TOKEN=your_hf_token
python scripts/build_index.py  # index validation split (default: hin, ben, tam, urd, mar; 500 rows each)
uvicorn app.main:app --reload --port 8000
```

## Quick Benchmark

```bash
python scripts/benchmark.py --num-queries 30
```
