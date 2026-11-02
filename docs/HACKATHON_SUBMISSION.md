# Swasthya Saathi: grounded voice health answers for Bharat

Draft submission notes for **Code for a Billion: Bharat Agentic-AI Hackathon** (Health track, submissions close 15 Nov 2026).

> **Medical-safety disclaimer.** This is a hackathon prototype. It is **not** a medical device and does **not** give medical advice, diagnosis or prescriptions. The bundled demo passages were written in plain language for this repository and checked against public WHO / MoHFW / NHM / NDMA pages (each answer links its source). They have **not** been reviewed by a clinician. Users should always follow their ASHA / ANM, doctor or nearest government health centre. In an emergency, call **108** (ambulance) or **102** (pregnant women and infants).

---

## 1. Problem

Frontline health workers (ASHAs, ANMs) and families in rural India often have a simple health question (how to make ORS, what the danger signs in pregnancy are, what to do after a snakebite) and no reliable, quick, local-language source to ask. General chatbots answer confidently even when they are wrong, which is dangerous in health. Typing is also a barrier: many users would rather speak, in Hindi or another Indian language.

## 2. Solution

Swasthya Saathi ("health companion") is a voice-first question-answering service:

1. The user **speaks** (or types) a question in Hindi or English.
2. Speech is transcribed by **Sarvam STT** (`saarika:v2.5`, language auto-detected).
3. A **hybrid retriever** searches a curated health corpus (dense vectors in Qdrant plus SQLite FTS5 lexical search, fused with Reciprocal Rank Fusion).
4. The answer is **extractive**: it is made only of sentences from the retrieved passages, so it cannot invent facts. Every answer carries citations and a link to the public source.
5. If the evidence is weak, the question is off-topic, the answer would not address the question ("is a double dose of ORS harmful?" when the sources don't say), or the input is a prompt-injection attempt, the system **abstains** instead of guessing.
6. A **red-flag triage** step detects emergency phrases (snakebite, heavy bleeding, fits, chest pain, unconsciousness, heat stroke, in English and Hindi) and shows a 108/102 referral banner **in the user's script**, even when the corpus does not cover the question.

Post-transcription latency is in the single-digit milliseconds on the demo corpus (about 5 ms in the screenshot below). The real MSMARCO-XI index meets the sub-200 ms target; see the README for those numbers.

![Demo mode: Hindi snakebite question with emergency banner and cited answer](screenshots/demo_mode_hindi_snakebite.png)

## 3. Architecture

```
 voice / text
     |
     v
 [Sarvam STT] --(no key / offline demo)--> [ScriptedSTT]
     |
     v
 [Input guard]  -- unsafe / injection / empty --> abstain (blocked)
     |
     v
 [Hybrid retriever]  dense (Qdrant, e5-small or hashing embedder)
     |                + lexical (SQLite FTS5, Devanagari-aware tokens)
     |                -> RRF fusion, per-doc and per-strategy diversity caps
     v
 [Retrieval guard]  -- low score or low margin --> abstain (not in corpus)
     |
     v
 [Extractive generator] -> [Grounding check] -- fails --> abstain
     |
     v
 [Red-flag triage]  -> 108/102 advisory (Hindi or English)
     |
     v
 FastAPI JSON (answer, citations + sources, advisory, per-stage timings) -> web UI
```

On top of this pipeline sits the **agent loop** (`POST /api/agent`, section 4):

```
 question (+ answers to earlier clarifying questions)
     |
     v
 [input_guard] -- injection / unsafe --> blocked (self-harm -> Tele-MANAS 14416)
     |
     v
 [triage]  red flag? -- yes --> SHORT-CIRCUIT: 108/102 banner + emergency facility + cited first aid
     |                          (no planner, no clarifying question)
     no
     v
 planner loop (rules by default, optional local Ollama), max 6 steps, one tool per step:
     ask_clarifying   -> stop and ask (e.g. "How old is the patient?")
     facility_lookup  -> bundled facility-type / national helpline table
     retrieve_guidance-> the cited RAG pipeline above (may abstain)
     finish
     |
     v
 AgentResponse: status, answer, advisory, facility, citations, step trace (tool, args, observation, planner, ms)
```

Key files: `app/agent.py` (agent loop, tools, planners), `app/harness.py` (pipeline), `app/retriever.py`, `app/guardrails.py`, `app/generator.py`, `app/demo.py` (health corpus, triage, scripted STT), `app/main.py` (FastAPI), `web/index.html` (UI).

## 4. What is "agentic" here (honest version)

### 4a. The agent loop (built, offline, tested)

`POST /api/agent` with `{"query": "...", "slots": {"age": "..."}}` runs a small tool-using agent. Each step picks one tool, runs it and records the result. The API returns the full trace, and the web UI shows it when **Agent mode** is ticked.

| Tool | What it does |
| :-- | :-- |
| `triage` | Red-flag rules in English and Hindi. It also escalates when the age slot shows **fever in an infant under 2 months** (a danger sign in the IMNCI guidelines). |
| `ask_clarifying` | When the question is about fever or diarrhoea and no age is given, the agent stops and asks "How old is the patient?" (in Hindi for Hindi questions). The UI sends the answer back as a slot and the agent resumes. |
| `facility_lookup` | A bundled table of **facility types**, not addresses: ASHA, sub-centre / Ayushman Arogya Mandir, PHC, CHC, district hospital. It also has national helplines: 108, 102, 104, 14416 Tele-MANAS, 1098, 181. It is used for "where / whom to call" questions, and as recovery when the corpus has no guidance for a real symptom question. |
| `retrieve_guidance` | The existing cited, extractive RAG pipeline. It can still abstain. |

**Planners.**
- **Rules (default).** Deterministic routing, fully covered by tests.
- **Ollama (optional).** Set `VOICE_RAG_AGENT_LLM=ollama` (and `VOICE_RAG_AGENT_MODEL`, default `llama3.2:3b`). A local model chooses the next tool through Ollama tool-calling. Every choice is validated:
  - Unknown tools are rejected.
  - Repeated tools are rejected.
  - "finish" is rejected before retrieval.
  - The model cannot rewrite the search query.
  - The model can only ask a clarifying question that the rules agree is needed.

  Any rejection or error falls back to the rules for that step, and the trace says `(fallback: rules)`.

**Hard safety rules, enforced by the loop and not by the planner:**
1. `input_guard` and `triage` always run first.
2. A red flag **always short-circuits** to the 108/102 banner. The emergency helplines and cited first aid are added, no clarifying question is asked, and the planner is never consulted. A test uses a planner that raises if it is called.
3. Answer text only ever comes from the extractive, cited retrieval or from the fixed tables. No LLM writes medical content.

**Live check with llama3.2:3b (CPU, already pulled locally).**
- It picked a sensible first tool: `retrieve_guidance` for the anaemia question, `facility_lookup` for the TB "where" question.
- It then kept repeating that same tool, which is now rejected. The rules then finished the run.
- It did **not** ask the clarifying question for "my child has fever"; it went straight to retrieval.
- It took about 7 to 50 s per question.

So the rule planner is the default and the honest headline. The LLM planner is an experiment.

Trace screenshots:

- Facility lookup and cited guidance: ![Agent trace: TB facility lookup + cited answer](screenshots/agent_trace_tb_facility.png)
- Infant fever escalated by the age slot: ![Agent trace: infant fever escalated to 108/102](screenshots/agent_trace_infant_fever_emergency.png)
- Clarifying question: `screenshots/agent_trace_clarifying_question.png`.

### 4b. The guarded pipeline underneath

The RAG pipeline itself is a **guarded tool pipeline with decision points**:

- **Autonomous decisions on every request.** The system decides on its own whether to answer, abstain (weak evidence, off-topic) or block (injection or unsafe input). It does not always produce output.
- **Self-verification.** After generating, three checks run (`app/harness.py`). (1) The original lexical grounding check: at least 40% of answer tokens appear in the retrieved evidence. (2) Strict citation verification (`app/citations.py`): every answer sentence is checked against the passage it cites, with English and Hindi stopwords removed; at least 80% of content words must be in that passage, and every number, drug/vaccine/helpline name and dose in the sentence must appear there too ("six" and "6", "छह" and "६" count as the same number). (3) An output guard (`app/safety.py`): the answer may only contain sentences copied from the retrieved passages or fixed templates. In demo mode a fourth check asks whether the answer addresses the question at all: at least 60% of the question's content words must be in it, so an on-topic question the sources don't answer ("is a double dose of ORS harmful?") abstains instead of getting the nearest sentence (`scripts/eval_answerability.py`, held-out set written after the rule was frozen: 14/16 such questions abstain, up from 9/16, and the gate blocks none of the 14 answerable ones). Any failure makes the system abstain. On a hand-labelled set of 47 claim/passage pairs (24 supported, 23 not), the old 40% rule caught 1 of 23 bad citations (accept precision 0.52). The strict check catches 18 of 23 (precision 0.83) and still accepts all 24 good ones. The 5 it misses are negation flips and drug swaps where every word is still in the passage ("use aspirin" when the passage says "use paracetamol ... avoid aspirin"). Catching those needs an entailment model, which this offline CPU build does not have. All numbers come from `python scripts/eval_guards.py` (`storage/guard_eval.json`).
- **Prompt-injection defence.** The main protection is architectural: the generator is extractive, so no LLM reads the user's text or the retrieved passages. On top of that, `app/safety.py` normalises text first (NFKC, zero-width and bidi characters removed, Cyrillic/Greek look-alike letters mapped to Latin, leetspeak, Devanagari nukta, letter spacing). It then matches about 40 English, Hindi, Hinglish, Tamil and Tanglish pattern families, such as instruction override, keyword-free paraphrases ("set aside what you were told", "your rules don't apply"), prompt extraction, persona jailbreaks, chat-template markup, instructions addressed to the model and exfiltration. The same scan runs on the user's input and on **every retrieved passage**, and flagged chunks are quarantined before generation. Base64, hex and URL-encoded runs are decoded up to 3 nested levels (at most 16 runs, 4,096 chars each, only when a layer is readable text or another encoded run) and scanned again. Measured results: on the original 32 adversarial inputs, 32 blocked (29/32 before 31 Oct; the old 7-phrase list blocked 2/32). On the expanded 54-input set (22 new paraphrase, Tamil/Tanglish and nested-encoding attacks) 54/54 blocked (30/54 before the 31 Oct changes). All 6 poisoned-passage cases were caught. 0 false positives on the original 64 benign questions (7 demo, 3 agent chips, 28 benchmark, 26 hard negatives), 0 on 18 new hard negatives (incl. 8 Tamil/Tanglish health questions), and 0 flagged chunks among the 27 demo-corpus and 4,751 MSMARCO-XI chunks. **Caveat, measured:** patterns and those test cases were written together. A held-out set written after the patterns were frozen (10 attacks, 10 benign) gives **2/10 blocked, 0/10 false positives**. That was the honest generalisation number for patterns alone. **Update (1 Nov):** a small learned second stage now runs on input (`app/injection_clf.py`, char/word n-gram TF-IDF + logistic regression, numpy only, 115 KB, under 1 ms per query, trained reproducibly by `scripts/train_injection_clf.py`). Its threshold was chosen out-of-fold for zero validation false positives. Patterns OR classifier blocks **8/10** on that untouched held-out set and **20/30** on a second held-out set written before the training data (patterns alone: 6/30). It adds no false positives on the 64 benign, 18 hard negatives or 40 held-out benign questions; the pattern guard itself has 1 FP on the new set. It stays off retrieved passages because it would flag 3/4,751 MSMARCO chunks. Both sets share an author with the training data, so treat these as optimistic. The real defence remains that no LLM reads the text.
- **Safety escalation.** The triage step acts independently of retrieval. It escalates to an emergency referral when it sees red-flag phrases.
- **Tool use.** It calls STT, a vector store and a lexical index, and merges their results.

**Not built yet** (roadmap for the submission window):

- A real nearest-facility directory with addresses and distance, for example from the public health facility registry. Today's table gives only facility types and national helplines.
- An immunisation-schedule tool that works from date of birth.
- More clarifying slots (duration, pregnancy status) and more languages in the clarifying questions. Today there is one slot, `age`, in English and Hindi.
- A better LLM planner. The 3B model is not reliable enough to lead (see above).
- Text-to-speech replies (for example Sarvam TTS) so the full loop is voice-in, voice-out.

The LLM in any of these steps would only choose actions and phrase output. Answers stay extractive and cited.

## 5. Indic-language support: current status

| Layer | Status |
| :-- | :-- |
| Speech-to-text | Sarvam `saarika:v2.5` with `language_code: "unknown"` (auto-detect). Sarvam supports several Indian languages, but **only Hindi and English were exercised** in this project. The live key was verified with one real call. Real recorded Hindi speech has **not** been systematically tested. |
| Demo corpus | 18 passages: 10 English and 8 Hindi (Devanagari). **No other Indian languages yet.** |
| Retrieval | The lexical tokenizer handles Devanagari combining marks (a bug fix: stdlib `\w` split Hindi conjuncts). The full pipeline uses `multilingual-e5-small`. The offline demo uses a hashing embedder, so its cross-lingual matching is weak: a Hindi question matches Hindi passages, not English ones. |
| Triage | Red-flag phrases in English and Hindi only. The advisory is shown in the script of the question. |
| Full dataset | `ai4bharat/MSMARCO-XI`. The Hindi validation split was indexed (4,751 chunks). Other languages are supported by the loader but have not been indexed. |
| UI | English labels. The answer text follows the corpus language. |

Next languages to add: Tamil, Bengali, Marathi and Telugu passages, with matching red-flag lexicons. Each one needs a native speaker to review it.

## 6. How to run

```bash
# Offline demo: no API key, no Qdrant server, no model download
docker compose up --build demo          # http://localhost:8000
# or, without Docker
pip install -r requirements-demo.txt
VOICE_RAG_DEMO=1 uvicorn app.main:app --port 8000
python scripts/demo.py --check          # scripted console walkthrough
curl -X POST localhost:8000/api/agent -H "Content-Type: application/json" \
     -d '{"query": "My child has fever since yesterday"}'               # -> needs_info (age)
#    then the same query with "slots": {"age": "5 weeks"}              # -> emergency, 108/102
# optional local LLM planner (never pulls a model): VOICE_RAG_AGENT_LLM=ollama

# Tests (offline)
pip install -r requirements-dev.txt && python -m pytest -q
```

The demo panel in the UI has one-click scripted questions: an English answer, a Hindi answer, colloquial Hindi, a pregnancy red flag, a Hindi snakebite red flag, an off-topic abstain and a blocked prompt injection. `/demo/health?q=<question>` asks a question on page load. `/demo/health?agent=1&q=<question>&age=<age>` runs it through the agent. Three extra demo chips show the agent paths: a clarifying question, a facility lookup and a red-flag short-circuit.

## 7. Submission checklist (before 15 Nov)

Last reviewed 1 Nov 2026. "Done" means it was run and checked, not only written.

### Done
- [x] **Reproducible offline demo.** `bash scripts/demo.sh` (PowerShell: `scripts/demo.ps1`) runs the Health-track scenario and exits 1 if any step drifts. It covers 10 scripted questions (English and Hindi cited answers, red-flag advisories, an off-topic abstain, an on-topic unanswerable abstain, English and Tamil injections blocked), 13 guard checks and 4 agent checks. `--serve PORT` then starts the web UI. A pytest test runs each part. README "Demo" section, screenshots in `docs/screenshots/`, shot list in [`DEMO_SCRIPT.md`](DEMO_SCRIPT.md).
- [x] **Injection guard demo.** It shows both stages. The pattern list catches English, paraphrased English, Hindi, Tamil, Tanglish and base64 attacks. The learned classifier catches held-out English, Tamil and Tanglish rewordings that match no pattern. One documented miss is shown, and three benign look-alikes pass. Held-out numbers: 8/10 and 20/30 (see `GUARDRAILS.md`).
- [x] **Agentic step.** The agent loop has a clarifying question, a facility lookup and a red-flag short-circuit (rule planner; optional Ollama planner).
- [x] **Fresh-clone check of the offline quickstart (31 Oct).** New venv, `requirements-demo.txt`, server up, `/health`, a cited answer, the agent emergency path and the injection block were all verified. `pytest` passed then; it is 260 as of 1 Nov.
- [x] **Classifier weights are a Git LFS object.** CI now checks out with `lfs: true`. A pointer file logs "run `git lfs pull`" instead of silently turning off stage 2, and the demo fails loudly in that case.

### Needs you (the night run cannot or must not do these)
- [ ] **Licence.** MIT or Apache-2.0 is required, and the repo has no LICENSE file. This is your choice; also confirm that the dataset and API terms (MSMARCO-XI, Sarvam) are compatible.
- [ ] **AgentFoundry / CFI org.** Sign up on AgentFoundry.me and open-source the repo under the CFI GitHub org, as the rules say. The night run did not check the current AgentFoundry requirements (no sign-ups or registrations). Re-read the rules page for anything new, such as a listing, a form field or a required agent manifest.
- [ ] **Deploy (any cloud).** `Dockerfile.demo` is a small image with no GPU and no model download. Options and a post-deploy checklist are in [`DEPLOY.md`](DEPLOY.md). Run `git lfs pull` before building. `Dockerfile.demo` was last built on 24 Oct and has not been rebuilt since the classifier was added (no Docker on the 1 Nov light run).
- [ ] **Push with LFS.** `git lfs push` must upload `app/models/injection_clf.npz`, or CI and clones get a pointer file.
- [ ] **Demo video** with real Hindi speech through Sarvam (needs `SARVAM_API_KEY`). Without the key the mic uses a scripted transcript, so don't claim live STT.
- [ ] **Clinician or public-health review** of the corpus passages, the red-flag list, the facility table and the infant-fever escalation rule.
- [ ] **Pilot numbers**, only if a real pilot happens. The video script says to show none rather than invented ones.
