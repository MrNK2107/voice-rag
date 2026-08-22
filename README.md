---
title: Voice RAG
emoji: 🎙️
colorFrom: green
colorTo: yellow
sdk: gradio
app_port: 7860
pinned: false
---

# Voice-Enabled RAG System (HH Goa 2026 Task 2)

An end-to-end, ultra-low-latency voice-enabled Retrieval-Augmented Generation (RAG) system built for the `ai4bharat/MSMARCO-XI` dataset.

The system transcribes spoken input using **Sarvam Speech-to-Text**, retrieves relevant document chunks via hybrid dense (Qdrant) and lexical (SQLite FTS5) search across multi-strategy chunking, and produces grounded answers with verified citations targeting sub-200ms post-transcription latency.

---

## 1. System Architecture

```txt
┌────────────────┐
│   User Voice   │
└───────┬────────┘
        │
        v
┌──────────────────────────────┐
│  Frontend MediaRecorder UI   │
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│    FastAPI POST /ask-audio   │
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│     Sarvam STT Adapter       │  (Cloud STT Latency: Reported Separately)
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│      Input Guardrails        │  (Safety & Injection Filters)
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐  (Post-STT RAG Path < 200 ms Target)
│       Hybrid Retriever       │
├───────────────┬──────────────┤
│ Qdrant Dense  │ SQLite FTS5  │
│ Search (e5)   │ BM25 Search  │
└───────┬───────┴──────┬───────┘
        │              │
        └──────┬───────┘
               v
┌──────────────────────────────┐
│   Reciprocal Rank Fusion     │  (Diversity & Strategy Filter)
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│   Retrieval Confidence Guard │
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│ Grounded Answer Generator    │  (Fast Extractive Engine)
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│    Grounding Check Guard     │
└───────┬──────────────────────┘
        │
        v
┌──────────────────────────────┐
│   Structured JSON Response   │  (Answer, Citations, Stage Timings)
└──────────────────────────────┘
```

---

## 2. Tech Stack

- **Backend**: FastAPI (Python 3.10+)
- **STT**: Sarvam Speech-to-Text (`saarika:v2.5`)
- **Dataset**: `ai4bharat/MSMARCO-XI`
- **Embeddings**: `intfloat/multilingual-e5-small`
- **Vector Database**: Qdrant (Cosine distance, HNSW `ef_search=32`)
- **Lexical Database**: SQLite FTS5 (BM25 score)
- **RAG Fusion**: Reciprocal Rank Fusion (RRF) + Parent Doc / Strategy Diversity Filter
- **Answer Generation**: Fast Grounded Extractive Generator (< 10ms execution)
- **Benchmarking**: Custom `scripts/benchmark.py` for P50, P70, P100 measurement

---

## 3. Quickstart & Setup

### Prerequisites

- Python 3.10+
- Docker & Docker Compose (for Qdrant)

### Environment Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

Edit `.env`:

```env
SARVAM_API_KEY=your_actual_sarvam_key
QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=msmarco_xi_chunks
EMBED_MODEL=intfloat/multilingual-e5-small
SQLITE_FTS_PATH=storage/chunks.sqlite
MIN_DENSE_SCORE=0.35
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Running the System

### Step 1: Start Qdrant

```bash
docker compose up -d
```

### Step 2: Build the Hybrid Vector Index

`ai4bharat/MSMARCO-XI` ships per-language parquet files (`validation/hinval.parquet`, `train/hintrain.parquet`, etc.) rather than a working `datasets` loading script, so `build_index.py` resolves and streams those parquet files directly. `train/*` files are ~3.7GB per language; `validation/*` files are ~460MB per language and are still real MSMARCO-XI data, so that's the default.

```bash
# Set once so huggingface_hub can authenticate (needed for reliable access):
export HF_TOKEN=your_hf_token

# Default: 5 languages (hin, ben, tam, urd, mar), 500 rows each, validation split
python scripts/build_index.py

# Customize languages / row count / split:
python scripts/build_index.py --languages hin ben tam --max-rows 1000 --split validation
python scripts/build_index.py --languages all --max-rows 2000
```

This requires real bandwidth to Hugging Face's storage backend (a few MB/s is enough; a throttled connection will make even the validation split painfully slow). If it hangs indefinitely on the first row, your network to `huggingface.co` is the bottleneck, not the script.

### Step 3: Launch FastAPI Server

```bash
uvicorn app.main:app --reload --port 8000
```

Access the UI at [http://localhost:8000](http://localhost:8000).

---

## 5. Benchmarking Latency

To run the custom benchmark across test queries and generate P50 / P70 / P100 metrics:

```bash
python scripts/benchmark.py --num-queries 30
```

### Latency Summary (Empirical Benchmark Results — real MSMARCO-XI data)

Measured against the real index: 4,751 chunks from 400 rows of `ai4bharat/MSMARCO-XI` (Hindi, validation split), across 30 benchmark queries (15 real queries pulled from the indexed corpus + 15 off-topic/unsafe/prompt-injection). See `storage/benchmark_queries.json` / `storage/benchmark_results.json`.

| Stage Name | P50 (ms) | P70 (ms) | P100 (ms) | Target Met |
| :--- | :--- | :--- | :--- | :--- |
| **Input Guardrails** | 0.03 ms | 0.03 ms | 0.04 ms | ✅ Yes |
| **Dense + Lexical Search (Qdrant + SQLite FTS5)** | 37.13 ms | 49.47 ms | 108.09 ms | ✅ Yes |
| **Retrieval Guard** | 0.00 ms | 0.00 ms | 0.01 ms | ✅ Yes |
| **Grounded Answer Generator** | 0.47 ms | 0.80 ms | 1.77 ms | ✅ Yes |
| **Grounding Validator** | 0.07 ms | 0.08 ms | 0.28 ms | ✅ Yes |
| **Post-STT RAG Path Total** | **36.65 ms** | **46.75 ms** | **108.8 ms** | **✅ Under 200ms Target** |
| **Cloud STT (Sarvam)** | ~1226 ms (measured live) | — | — | *(External Cloud API — real network round trip, not an estimate)* |

**Correctness**: of the 30 queries, 29/30 (96.7%) got the correct abstain-vs-answer decision — all 15 off-topic/unsafe/prompt-injection queries correctly refused/abstained, 14/15 real corpus queries got correctly grounded answers, and the 1 miss was a false-negative abstention (safe failure mode — declining to answer rather than hallucinating), not a wrong answer.

---

## 6. Current Status / Known Issues (as of latest review)

All blocking issues found during review are now resolved and verified against the real dataset. History, for context:

| # | Issue | Resolution |
| :-- | :-- | :-- |
| 1 | `storage/` held a 10-fact placeholder dataset, not real `ai4bharat/MSMARCO-XI` data — the dataset ships per-language parquet files under a broken/legacy `datasets` loading script that hangs indefinitely instead of erroring. | ✅ **Fixed & indexed with real data.** `build_index.py`/`dataset_loader.py` rewritten to stream parquet files directly (bypassing the broken script) and parse the dataset's real schema (`passages.Translated_passages`/`is_selected`). Live run indexed **4,751 real chunks from 400 rows** of the Hindi validation split — verified by inspecting actual chunk text (real MSMARCO passages about McDonald's Corp, Rachel Carson, honesty, etc., not fabricated). Along the way, also fixed: (a) a stale-data bug where rebuilds accumulated old rows instead of replacing them in both SQLite and Qdrant's local/embedded mode, and (b) a **Devanagari tokenization bug** — stdlib `re`'s `\w+` doesn't include Unicode combining marks, so it shredded Hindi conjuncts into single-character fragments (`"कॉर्पोरेशन"` → `['क','र','प','र','शन',...]`), corrupting lexical search and answer-sentence scoring for the entire non-English corpus. Fixed with the `regex` package's `\p{L}\p{M}\p{N}` pattern. |
| 2 | `SARVAM_API_KEY` in `.env` was a placeholder, so `stt_sarvam.py` silently returned a hardcoded fake transcript. | ✅ **Fixed.** Real key added, verified with a live `200 OK` transcription call and through the full `ask_audio` harness path (`stt_ms` ≈ 1226 ms, real cloud round trip). `.env` is now git-ignored. **Still worth doing yourself**: test with real recorded speech (not a synthetic tone) via the web UI before your demo video. |
| 3 | `retrieval_guard()` failed to abstain on off-topic queries — verified live with confident, ungrounded answers to "Who won the World Cup in 2022?" etc. | ✅ **Fixed, through two iterations.** v1 (dense-score-only, fixed threshold 0.80) fixed the original bug but broke again once real data was indexed, because absolute e5 cosine similarity doesn't generalize across corpus size (the "noise floor" for unrelated queries climbed from ~0.70-0.74 on 20 chunks to ~0.75-0.84 on 4,751 chunks). Replaced with a **margin-based** guard (top score vs. mean of tail candidates), which generalizes across corpus size — see `GUARDRAILS.md` for full calibration data. Re-verified on the real index: **29/30 (96.7%)** correct abstain/answer decisions across a 30-query benchmark (15 real + 15 off-topic/unsafe/injection). |

**Before final submission**, still worth doing: index more languages/rows for a richer demo (`python scripts/build_index.py --languages all` or add more to `--languages`), and record real speech through the web UI to confirm Sarvam transcription quality.

---

## 7. Project Structure

```txt
c:/hhgoa2/
  app/
    __init__.py
    main.py
    config.py
    schemas.py
    latency.py
    stt_sarvam.py
    chunking.py
    dataset_loader.py
    retriever.py
    generator.py
    guardrails.py
    harness.py
  scripts/
    explore_dataset.py
    build_index.py
    make_benchmark_queries.py
    benchmark.py
  web/
    index.html
  storage/
    .gitkeep
  docker-compose.yml
  Dockerfile
  requirements.txt
  .env.example
  README.md
  CHUNKING.md
  LATENCY.md
  GUARDRAILS.md
```
