# Vouch

**Ask out loud, in any language. Get answers your documents can vouch for, or an honest "not in your documents".**

Vouch is a voice-first answer desk over an organisation's own documents. Every answer is a quote from a source you uploaded, with the source one tap away. When the documents don't say, Vouch says so, and that gap becomes a to-do for whoever owns the documents.

`voice-rag` stays as the repository and package name. Vouch is the product.

## Who it is for

The wedge is people who **don't type in English** and **can't afford a confident wrong answer**:

| Buyer | The person asking | Their documents |
|---|---|---|
| Support desks for public services, utilities, banks | callers and walk-ins speaking Hindi, Tamil, Arabic, Swahili... | policy PDFs, FAQ pages, circulars |
| NGOs and health programmes | field workers (ASHAs, community health workers) on a cheap phone | training manuals, protocol sheets |
| Universities, councils, HR teams | students, residents, staff | handbooks, regulations, forms |

They share one shape: a pile of authoritative documents, many askers who speak, not type, and a real cost when the answer is invented.

## The problem

Generic chatbots answer everything. That is the problem. They paraphrase, mix in things the model half-remembers, answer in English when asked in Marathi, and never say "I don't know". For a ration-card office or a health programme, one fluent wrong answer costs more than ten honest "not in our documents".

## Why Vouch wins

1. **Receipts, not paraphrase.** Answers are extracted sentences with citations to the exact passage. A strict per-sentence check rejects anything the cited passage doesn't support. There is no LLM between your documents and the answer by default, so there is nothing to hallucinate.
2. **Abstention is a feature.** Off-topic, unsupported or on-topic-but-unanswerable questions get a clear "not in your documents". Each one lands in the **Gaps** report: the questions people asked that your documents couldn't answer.
3. **Any language, by voice.** The question's language is detected in any script (Latin, Devanagari, Tamil, Arabic, Han, Cyrillic, Hangul and more). Speech goes through Sarvam (Indian languages), Whisper (any language), or the browser's built-in recogniser (no key). The interface flips to right-to-left for Arabic, Urdu, Persian and Hebrew, and replies can be read aloud.
4. **Safe by construction.** An injection guard (patterns plus a small learned classifier) screens questions and quarantines poisoned passages, and an emergency triage step shows a call-for-help banner in the asker's language.
5. **Drops into the site they already have.** One `<script>` tag puts a Vouch button on any web page, scoped to one collection.

## What we built (MVP, this repository)

Headline features, and why these four:

- **Collections console** (`/admin`): create named collections, upload PDF / text / Markdown / HTML or ingest a URL, see document counts and languages. Without this, Vouch is a demo; with it, a programme manager can set it up alone.
- **Embeddable widget** (`<script src="/widget.js" data-collection="...">`): the distribution channel. The buyer's users never visit our site.
- **Insights and gap report** (`/admin`, Insights tab): questions per day, abstention rate, languages used, top questions, and the unanswered ones. This is what the buyer pays for after month one: it tells them which documents to write next.
- **Source-highlighted answers**: each citation chip opens the source passage with the quoted sentence highlighted, so a sceptical user can check in one tap.

Left out on purpose: conversation memory. Follow-ups that depend on hidden context make answers harder to verify, which cuts against the product's main promise.

## Pricing sketch

| Plan | Price | For |
|---|---|---|
| Community | free, self-hosted | NGOs, students, pilots. Everything in this repo. |
| Team | $49 / month | 5 collections, 10k questions/month, hosted, widget, insights |
| Institution | $499 / month | unlimited collections, SSO, audit export, managed Sarvam/Whisper, data residency (India / EU) |
| Public sector | per-seat or per-district contract | offline appliance, IVR/phone-line add-on, procurement paperwork |

Voice minutes are passed through at cost plus 20%. Grounded text answers are cheap to serve (no LLM tokens), which is what makes the free tier and the low Team price possible.

## Roadmap

- **MVP (now):** collections plus upload, hybrid multilingual retrieval, cited extractive answers, abstention, guards, three STT options, widget, insights, light/dark, RTL.
- **v1 (next quarter):** phone-line and WhatsApp channels (the real interface for field workers), optional LLM rewrite of the cited answer into the asker's language behind the same citation and output checks, per-collection access keys, reviewer workflow on the gap report ("answer this gap" adds a document).
- **v2:** offline Android build with on-device Whisper and embeddings for low-connectivity districts; multi-tenant hosting; analytics by region; an NLI entailment check to replace the lexical citation rule.

## Honest limits today

- Answers are copied sentences, so a Spanish question over English documents gets an English quote (labelled with its language) unless a translator is configured. The default multilingual embedder (`intfloat/multilingual-e5-small`) does retrieve across languages; the offline test embedder does not.
- The injection guard has measured gaps (28/40 on the held-out set); see GUARDRAILS.md.
- No auth on the admin console yet: run it behind your own network or reverse proxy.
