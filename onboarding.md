# Onboarding Guide — voice-rag

Welcome! This guide gets a new developer from "what is this repo?" to "I can run, test, and iterate on the Voice RAG system" in under 15 minutes.

---

## 1. What Is This Project?

A **voice-enabled Retrieval-Augmented Generation (RAG)** system. A user speaks a question into the browser, the pipeline:

1. Captures audio via the browser's MediaRecorder
2. Transcribes it using **Sarvam Speech-to-Text** (model: saarika:v2.5)
3. Retrieves relevant context from `ai4bharat/MSMARCO-XI` via hybrid dense (Qdrant) + lexical (SQLite FTS5) search
4. Generates a grounded extractive answer (<2ms)
5. Validates the answer is grounded in retrieved evidence
6. Returns structured JSON with citations and stage-by-stage timings

**Target**: Sub-200ms latency for the post-STT pipeline (measured P50 ~37ms, P100 ~109ms on a 4,751-chunk index).

---

## 2. Quick Start (3 Steps)

### Prerequisites
- Python 3.10+
- Docker & Docker Compose (for Qdrant vector DB)
- A Hugging Face token (for dataset access)
- A Sarvam API key (for speech-to-text)

### Step 1: Set up the environment

```bash
cd voice-rag
pip install -r requirements.txt
cp .env.example .env
# Edit .env and set your SARVAM_API_KEY
# Set HF_TOKEN as an environment variable when indexing
```

### Step 2: Start Qdrant and build the index

```bash
docker compose up -d
export HF_TOKEN=your_hf_token
python scripts/build_index.py
# Add more languages: python scripts/build_index.py --languages ben --max-rows 500 --append
```

### Step 3: Run the server

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 — the web UI loads automatically.

---

## 3. How the Code Is Organized

```
voice-rag/
├── app/                     # Application source (Python package)
│   ├── __init__.py          # Package marker
│   ├── main.py              # FastAPI app - HTTP endpoints
│   ├── config.py            # Settings / configuration (pydantic-settings)
│   ├── schemas.py           # Pydantic request/response models
│   ├── harness.py           # VoiceRAGHarness - pipeline orchestrator
│   ├── stt_sarvam.py        # Sarvam STT client (with retries)
│   ├── chunking.py          # Multi-strategy text chunker (6 strategies)
│   ├── dataset_loader.py    # MSMARCO-XI parquet row parser
│   ├── retriever.py         # Hybrid dense (Qdrant) + lexical (SQLite FTS5) retrieval
│   ├── generator.py         # Extractive answer generator (<2ms)
│   ├── guardrails.py        # Safety & grounding guards (3 layers)
│   └── latency.py           # Stage timing instrumentation
├── scripts/                 # Operational scripts
│   ├── build_index.py       # Build Qdrant + SQLite index from dataset
│   ├── benchmark.py         # Run latency benchmarks (P50/P70/P100)
│   ├── explore_dataset.py   # Inspect dataset schema
│   └── make_benchmark_queries.py  # Generate test query set
├── web/
│   └── index.html           # MediaRecorder UI (voice + text input)
├── storage/                 # Data (git-ignored except benchmark files)
│   ├── chunks.sqlite        # SQLite FTS5 index (generated)
│   ├── qdrant_db/           # Qdrant local DB (generated)
│   ├── stt_cache/           # STT cache (optional, git-ignored)
│   ├── benchmark_queries.json   # 30 test queries (tracked)
│   └── benchmark_results.json   # P50/P70/P100 results (tracked)
├── Dockerfile               # Production container
├── docker-compose.yml       # Qdrant container definition
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md              # High-level overview, setup, results
├── PROJECT_LOG.md         # Full build journey (bugs, fixes, decisions)
├── CHUNKING.md            # Chunking strategy spec
├── GUARDRAILS.md          # Guardrails & safety system spec
├── LATENCY.md             # Latency analytics & optimization
├── context.md             # Quick project context reference
└── onboarding.md          # This file
```

---

## 4. The Pipeline: Step by Step

### 4.1 HTTP Entrypoints (app/main.py)

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serves the web UI |
| `/health` | GET | Health check |
| `/api/ask-text` | POST | Submit a text query (fastest to test) |
| `/api/ask-audio` | POST | Submit an audio file (full voice path) |

### 4.2 Orchestration (app/harness.py)

`VoiceRAGHarness` is the pipeline conductor. Every request flows through:

1. **STT** — `self.stt.transcribe(audio_bytes, filename, content_type)`
2. **Input Guard** — `input_guard(transcript)` blocks unsafe content and prompt injection
3. **Retrieval** — `self.retriever.retrieve(query)` runs hybrid dense + lexical search with RRF fusion
4. **Retrieval Guard** — `retrieval_guard(contexts, confidence)` checks margin-based off-topic detection
5. **Generation** — `self.generator.generate_extractive(query, contexts)` extracts answer sentences
6. **Grounding Check** — `grounding_check(answer, contexts)` validates >= 40% token support from evidence

Every failure path returns a structured `RagResponse` with `abstained=True` and a human-readable `abstain_reason`. No raw crashes.

### 4.3 Data Models (app/schemas.py)

**RagResponse** is the primary output model:
- `transcript: str` — What STT heard
- `answer: str` — The answer (or refusal message)
- `citations: list[Citation]` — Source chunks used
- `grounded: bool` — Passed grounding check?
- `abstained: bool` — Did we refuse to answer?
- `abstain_reason: str | None` — Why we abstained (if we did)
- `timings_ms: dict` — Per-stage latency breakdown (stt_ms, input_guard_ms, retrieval_ms, etc.)

**RetrievedContext** is the per-chunk retrieval result: chunk_id, text, score, dense_score, lexical_score, strategy, language, parent_doc_id, title.

### 4.4 Configuration (app/config.py)

All settings come from environment variables via pydantic-settings. Key knobs:

| Variable | Default | Description |
|---|---|---|
| SARVAM_API_KEY | (empty) | Sarvam API key — required for real STT |
| QDRANT_URL | http://localhost:6333 | Qdrant server URL |
| QDRANT_COLLECTION | msmarco_xi_chunks | Collection name |
| EMBED_MODEL | intfloat/multilingual-e5-small | Sentence embedding model |
| SQLITE_FTS_PATH | storage/chunks.sqlite | SQLite FTS5 index path |
| MIN_DENSE_SCORE | 0.50 | Absolute dense score backstop |
| MIN_DENSE_MARGIN | 0.055 | Margin threshold for retrieval guard |
| STT_CACHE_ENABLED | false | Cache STT results locally |

---

## 5. Key Concepts to Understand

### 5.1 Why Extractive, Not LLM Generation?
The 200ms budget excludes cloud STT (~1,226ms). An LLM API call alone costs 800ms-2,500ms. Extractive generation picks sentences directly from retrieved chunks — under 2ms, and only says things literally present in evidence (trivially grounded). Trade-off: less fluent answers, but acceptable given the latency constraint.

### 5.2 The Margin-Based Retrieval Guard
A fixed cosine similarity threshold (e.g., 0.80) doesn't generalize across corpus sizes — the "noise floor" for unrelated queries rises as the corpus grows. The guard instead checks if the top hit's score stands out above the **mean of tail candidates** (ranks 10-40). This is relative, not absolute, so it doesn't drift. See GUARDRAILS.md for full calibration data.

### 5.3 The Devanagari Tokenization Fix
Python's stdlib `re` uses the pattern `\w+` which excludes Unicode combining marks (Devanagari vowel signs, viramas, etc.). This shredded Hindi conjuncts like "कॉर्पोरेशन" into single characters, breaking both term-overlap scoring and FTS5 query construction. Fixed by using the third-party `regex` package's pattern `\p{L}\p{M}\p{N}` — see generator.py:13 and retriever.py:16.

### 5.4 Direct Parquet Streaming
`ai4bharat/MSMARCO-XI` ships parquet files behind a legacy `datasets` loading script that hangs on modern library versions. `build_index.py` streams the parquet files directly via `huggingface_hub.HfFileSystem` + `pyarrow`, bypassing the broken script. It uses an `--append` flag so you can index one language per command invocation (each file has a one-time read cost — the parquet stores all rows in a single row group).

### 5.5 The 6 Chunking Strategies (app/chunking.py)
See CHUNKING.md for the full spec. Briefly:

| Strategy | Target | Use Case |
|---|---|---|
| atomic_short_passage | <=140 words | Preserve short passages as atomic units |
| metadata_title_intro | First 180 words | Title + intro as metadata anchor |
| qa_fused | Question + passage | Query-passage direct matching for QA |
| sentence_group_140w | ~140 words, 1 overlap | Sentence-boundary aware grouping |
| micro_80w_20o | 80 words, 20 overlap | Hyper-focused facts and entities |
| standard_180w_40o | 180 words, 40 overlap | General passage representation |
| macro_420w_80o | 420 words, 80 overlap | Long documents (>450 words) |

Chunks are deduplicated by SHA256-hashed normalized text and assigned deterministic UUID-v5 IDs.

---

## 6. Development Workflow

### Running the Benchmark
```bash
# Run latency benchmark (30 queries, P50/P70/P100)
python scripts/benchmark.py --num-queries 30
# Results are saved to storage/benchmark_results.json
```

### Testing the API
```bash
# Health check
curl http://localhost:8000/health

# Text query (fastest path, no STT needed)
curl -X POST http://localhost:8000/api/ask-text \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of Goa?"}'

# Voice query (requires microphone access in browser)
# Just open http://localhost:8000 and click "Start Voice Recording"
```

### Adding More Index Data
```bash
# Append a new language to the existing index
export HF_TOKEN=your_token
python scripts/build_index.py --languages ben --max-rows 500 --append

# Rebuild from scratch (wipes existing index)
python scripts/build_index.py --languages hin --max-rows 400
```

### Docker
```bash
# Start just Qdrant
docker compose up -d

# Build and run the app container
docker build -t voice-rag . && docker run -p 8000:8000 voice-rag
```

---

## 7. Known Issues & Gotchas

1. **Dataset download is slow** — Each parquet file is ~460MB (validation) or ~3.7GB (train) per language. The entire file must be read before any rows are available (single Parquet row group). Use `--max-rows` to limit ingestion per run.

2. **Shell argument mangling** — Devanagari text passed as inline shell arguments can get corrupted by terminal encoding issues. If testing via curl, send payloads from a UTF-8-encoded file instead of inline strings.

3. **Sarvam dev fallback** — If `SARVAM_API_KEY` is not set or is the placeholder string, `stt_sarvam.py` returns a hardcoded fallback transcript ("What is the capital of Goa?"). This is intentional for development.
