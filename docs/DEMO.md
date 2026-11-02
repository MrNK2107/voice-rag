# Vouch in 60 seconds

A screen recording script. Everything runs offline: `python -m app --demo`, then open `http://localhost:8000`. Use Chrome or Edge for the browser microphone.

| Time | On screen | Say |
|---|---|---|
| 0:00 | Landing page hero, the Arabic card cycling to Spanish and Chinese | "This is Vouch. You ask your documents out loud, in any language, and every answer is a quote you can check." |
| 0:07 | Click **Ask the sample city desk**. The Lakeview Civic Help Desk opens: 14 documents in 14 languages. | "This sample is a city help desk. Each page of its handbook is in a different language." |
| 0:12 | Tap the mic, say "¿A qué hora abre la biblioteca central?" (or click the Spanish chip). Language pill flips to **Spanish**. | "I ask in Spanish..." |
| 0:17 | The answer appears with the green **Vouched** stamp. Click the **[1] Biblioteca Central** chip; the passage opens with the sentence highlighted. | "...and the answer is the exact sentence from the handbook. One tap shows where it came from." |
| 0:25 | Type "How much does a passport cost?" The ochre **Not in your documents** card. | "Ask something the handbook doesn't cover, and Vouch says so. No guessing." |
| 0:31 | Click the Arabic chip "كيف أدفع فاتورة المياه؟". The card renders right-to-left. Turn on **Read answers aloud** in settings first if you want it spoken. | "Arabic works the same way, right to left, and it can read the answer back." |
| 0:38 | Type "My neighbour is unconscious after a fall". The red emergency banner. | "If a question sounds like an emergency, Vouch tells you to call for help first, in your language." |
| 0:43 | Open **Console** (`/admin`), Collections tab: drag a PDF onto the drop zone. It appears in the document table with its language. | "Adding your own documents is a drag and drop. PDFs are cited by page." |
| 0:49 | **Insights** tab: KPIs, languages, the **gap report** listing the passport question. | "Every question you couldn't answer shows up here. That is your list of documents to write next." |
| 0:55 | **Embed** tab: copy the snippet, the widget preview on the right. | "And one script tag puts this desk on any website. That's Vouch." |

## Before recording

- Ask the questions above once before you start, so the Insights tab has data (or record Insights last).
- Deep links replay a conversation without typing: `/app?c=city-desk&q=<question>&q=<question>&cite=1` (`cite=1` opens the last answer's source).
- In demo mode the server has no speech-to-text key, so use the browser microphone (settings, Speech recognition: "In the browser"). With `SARVAM_API_KEY` or `WHISPER_API_BASE` set, the server providers appear in the same menu.

## Honesty notes for the voice-over

- Lakeview is a fictional city; the handbook was written for this demo.
- The offline demo uses a simple hashing embedder that only matches questions to passages in the same language. With the default multilingual model, cross-language retrieval works, but the quote stays in the passage's language unless translation is configured.
- Don't claim the injection guard is complete. Its measured held-out recall is 28/40 for the languages it was evaluated on (see the README).
