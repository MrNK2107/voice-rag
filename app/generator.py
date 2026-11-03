import re
import regex
from typing import List, Tuple

from app.citations import content_words
from app.schemas import Citation, RetrievedContext
from app.langid import _LATIN_STOPWORDS, detect_language
from app.textutil import has_unspaced_script, is_unspaced_token, split_sentences as _split_sentences, tokenize

# NOTE: stdlib re's Unicode \w does NOT include combining marks (Mn/Mc categories),
# so it shreds Indic-script conjuncts (Devanagari matras/virama, and equivalents in
# Bengali/Gujarati/Tamil/etc.) into single-character fragments - e.g. "कॉर्पोरेशन"
# (corporation) becomes ['क','र','प','र','शन',...] instead of one token. That broke
# term-overlap scoring for the entire (non-English) MSMARCO-XI corpus. The `regex`
# package's \p{L}\p{M}\p{N} properly keeps letter+mark+digit runs together.
_WORD_PATTERN = regex.compile(r"[\p{L}\p{M}\p{N}]+", flags=regex.UNICODE)


def split_sentences(text: str) -> List[str]:
    # Also splits on Arabic/Urdu and CJK sentence marks (app/textutil.py).
    return _split_sentences(text)


def important_terms(query: str) -> set[str]:
    terms = tokenize(query.lower())
    # Exclude common short stop-ish words (CJK bigrams are 2 characters long)
    return {t for t in terms if (len(t) > 2 or is_unspaced_token(t)) and t not in {"what", "when", "where", "which", "who", "whom", "whose", "why", "how", "that", "this", "these", "those"}}


# Question words and light verbs that say *how* something is asked, not *what*.
# They are ignored by query_coverage. EN, HI and romanised Hindi function
# words (the stopword lists in app/citations.py cover EN + Devanagari only).
_QUESTION_WORDS = set("""
much many long often take get give given need make
क्या कैसे कौन कौनसा कौनसी कौनसे सी कहाँ कहां कब कितने कितनी कितना क्यों
kya kaise kaisa kab kahan kahaan kitna kitne kitni kyun kyon kaun ka ke ki ko me mein se hai hain ho aur
bana banaye banayein banaen karein kare karna chahiye
""".split())


# Question words in other scripts, ignored by query_coverage like the EN/HI ones.
_WORLD_QUESTION_WORDS = set("""
كيف ما ماذا متى أين هل لماذا من كم هي هو في على عن
کیا کیسے کب کہاں کون کتنا کتنی کے کی کا لیے ہے ہیں چاہیے میں سے کو
چه چگونه کی کجا چرا آیا است
как что когда где почему ли кто сколько это в на с о
як що коли де чому чи
무엇 어떻게 언제 어디 왜 누가 얼마
nini vipi lini wapi nani gani kwa ya na ni
""".split())
# Chinese / Japanese function characters (question particles, "how", "what").
_CJK_FUNCTION_CHARS = set("的了吗吗呢么怎什哪谁几是在和与请吧啊呀如何为何多少哪里")


def _common_prefix(a: str, b: str) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


# Hindi light stemmer suffixes (Ramanathan & Rao, "A Lightweight Stemmer for
# Hindi", 2003), longest first. Hindi verb stems are often two letters (बच-ें,
# बच-ने), which the prefix rule below cannot match on its own.
_HI_SUFFIXES = sorted("""
ाएंगी ाएंगे ाऊंगी ाऊंगा ाइयाँ ाइयों ाइयां
ाएगी ाएगा ाओगी ाओगे एंगी एंगे ऊंगी ऊंगा ातीं नाओं नाएं ताओं ताएं ियाँ ियों ियां
ाकर ाइए ाईं ाया ेगी ेगा ोगी ोगे ाने ाना ाते ाती ाता तीं ाओं ाएं ुओं ुएं ुआं
कर ाओ िए ाई ाए ने नी ना ते ीं ती ता ाँ ां ों ें
ो े ू ु ी ि ा
""".split(), key=len, reverse=True)
_DEVANAGARI = regex.compile(r"\p{Devanagari}")


def _hi_stem(word: str) -> str:
    if not _DEVANAGARI.search(word):
        return word
    for suf in _HI_SUFFIXES:
        if word.endswith(suf) and len(word) - len(suf) >= 2:
            return word[: -len(suf)]
    return word


_ARABIC = regex.compile(r"\p{Arabic}")
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ل")
_AR_VERB_PREFIXES = ("أ", "ي", "ت", "ن")


def _ar_stem(word: str) -> str:
    """Very light Arabic-script normalisation: strip the article / conjunction
    prefixes and an imperfect-verb prefix (أدفع -> دفع). Not a real stemmer."""
    if not _ARABIC.search(word):
        return word
    for p in _AR_PREFIXES:
        if word.startswith(p) and len(word) - len(p) >= 3:
            word = word[len(p):]
            break
    if word[:1] in _AR_VERB_PREFIXES and len(word) >= 4:
        word = word[1:]
    return word


def _term_matches(term: str, answer_terms: set) -> bool:
    if term in answer_terms:
        return True
    stem = _hi_stem(term)
    if stem != term and any(_hi_stem(t) == stem for t in answer_terms):
        return True
    stem = _ar_stem(term)
    if any(_ar_stem(t) == stem for t in answer_terms if _ARABIC.search(t)):
        return True
    # Light inflection match (child/children, dose/doses): the shorter word
    # (3+ letters) is a prefix of the longer one, or, for words of 5+ letters,
    # all but its last letter is (symptom/symptoms, mosquito/mosquitoes).
    # Four-letter words need the full prefix: "heat" must not match "health".
    for t in answer_terms:
        shorter = min(len(term), len(t))
        if shorter < 3:
            continue
        cp = _common_prefix(term, t)
        if cp >= shorter or (shorter >= 5 and cp >= shorter - 1):
            return True
    return False


def query_coverage(query: str, answer: str) -> Tuple[float, List[str]]:
    """Share of the query's content words that the answer contains.

    Returns (coverage in [0, 1], query words the answer misses). Stopwords
    (EN + HI, app/citations.py) and question words are ignored. A query with
    no content words counts as fully covered.

    Extractive answers are copied sentences, so when an on-topic question asks
    for something the corpus does not contain ("is a double dose harmful?"),
    the best-overlapping sentences answer a different question. Low coverage
    is the signal for that; see VoiceRAGHarness.ask_text.
    """
    lang = detect_language(query, use_fallback=False).code
    if has_unspaced_script(query):
        # Chinese / Japanese / Thai: compare characters, not bigrams (a bigram
        # that straddles two words, like 人怎 in 老人怎么, is never in the answer).
        q_chars = {c for c in regex.findall(r"[\p{Han}\p{Hiragana}\p{Katakana}\p{Thai}]", query)
                   if c not in _CJK_FUNCTION_CHARS}
        if q_chars:
            missing = sorted(c for c in q_chars if c not in answer)
            return 1.0 - len(missing) / len(q_chars), missing
    ignore = _QUESTION_WORDS | _WORLD_QUESTION_WORDS
    if lang not in ("en", "hi", "hi-Latn"):
        # Function words of the detected language (es: qué, la, de...).
        ignore = ignore | _LATIN_STOPWORDS.get(lang, set())
    q_terms = sorted(content_words(query) - ignore)
    if not q_terms:
        return 1.0, []
    a_terms = content_words(answer)
    missing = [t for t in q_terms if not _term_matches(t, a_terms)]
    return 1.0 - len(missing) / len(q_terms), missing


def _strip_title(ctx: RetrievedContext) -> str:
    """Chunk text without the retrieval-only "Title: <title>" prefix, the
    "Intro: " label of metadata_title_intro chunks, and the dataset question of
    qa_fused chunks ("Question: ... Relevant evidence: <passage>"), which would
    otherwise be the best "answer" to that very question (app/chunking.py)."""
    text = ctx.text or ""
    if text.startswith("Question: ") and "Relevant evidence: " in text:
        return text.split("Relevant evidence: ", 1)[1]
    if ctx.title and text.startswith(f"Title: {ctx.title}"):
        text = text[len(f"Title: {ctx.title}"):].lstrip()
        if text.startswith("Intro: "):
            text = text[len("Intro: "):]
    return text


# Word-window chunks (app/chunking.py sliding_word_chunks) start and end at
# arbitrary words, so their first and last "sentences" can be fragments. Quoting
# "Children can take aspirin" from "...aspirin only when a doctor prescribes it"
# would pass every check, since the fragment is a substring of the passage.
_WINDOW_STRATEGY = regex.compile(r"^(?:micro|standard|macro)_\d+w")
_SENTENCE_END = regex.compile(r"[.!?।॥؟۔。！？][\"'”’)\]]*\s*$")


# These start at the document start but stop after a fixed number of words.
_TRUNCATED_STRATEGIES = {"metadata_title_intro", "qa_fused"}


def _whole_sentences(text: str, strategy: str | None) -> List[str]:
    sents = split_sentences(text)
    window = bool(strategy and _WINDOW_STRATEGY.match(strategy))
    if window and len(sents) > 1:
        sents = sents[1:]
    if (window or strategy in _TRUNCATED_STRATEGIES) and sents and not _SENTENCE_END.search(sents[-1]):
        sents = sents[:-1]
    return sents


class AnswerGenerator:
    def generate_extractive(
        self,
        query: str,
        contexts: List[RetrievedContext],
    ) -> Tuple[str, List[Citation]]:
        q_terms = important_terms(query)

        candidates = []

        for ctx_rank, ctx in enumerate(contexts):
            # Window/sentence-group chunks are prefixed with "Title: <title> " for
            # retrieval; don't let that prefix leak into the extracted answer.
            text = _strip_title(ctx)
            for sent in _whole_sentences(text, ctx.strategy):
                sent = sent.strip()
                if len(sent) < (8 if is_unspaced_token(sent) else 25):
                    continue

                s_terms = set(tokenize(sent.lower()))
                # Same inflection-aware matching as query_coverage, so the
                # generator and the coverage gate agree (बचें / बचने).
                matched = {t for t in q_terms if _term_matches(t, s_terms)}
                overlap = len(matched)

                # Prioritize sentence term overlap, then context rank
                score = (overlap * 3.0) + (1.0 / (ctx_rank + 1))

                candidates.append((score, sent, ctx, matched))

        if not candidates:
            return "", []

        # Stable sort keeps passage order for equal scores.
        candidates.sort(key=lambda x: x[0], reverse=True)

        _, best_sent, best_ctx, best_terms = candidates[0]
        selected = [(best_sent, best_ctx)]

        # Supporting sentence: same document as the best one (stitching two
        # documents produced answers like "TB treatment is free. Take the person
        # to a hospital...", the second from the snakebite passage), preferring
        # the sentence that adds the most query terms the first one lacks, then
        # the usual score. For "Child has diarrhoea, how do I make ORS?" the best
        # sentence matches child + diarrhoea; the support then adds the ORS step.
        support = None
        for score, sent, ctx, terms in candidates[1:]:
            if sent.lower() == best_sent.lower() or ctx.parent_doc_id != best_ctx.parent_doc_id:
                continue
            key = (len(terms - best_terms), score)
            if support is None or key > support[0]:
                support = (key, sent, ctx)
        if support is not None:
            selected.append((support[1], support[2]))

        answer_sentences = [s for s, _ in selected]
        # CJK sentences end in a full-width mark and are joined without a space.
        joiner = "" if all(has_unspaced_script(s[-1:]) or s.endswith(("。", "！", "？")) for s in answer_sentences) else " "
        answer = joiner.join(answer_sentences)

        citations = []
        for sent, ctx in selected:
            citations.append(
                Citation(
                    chunk_id=ctx.chunk_id,
                    score=ctx.score,
                    strategy=ctx.strategy,
                    language=ctx.language,
                    quote=sent[:400],
                    title=ctx.title,
                    parent_doc_id=ctx.parent_doc_id,
                    passage=_strip_title(ctx)[:2000],
                    source=ctx.source,
                    page=ctx.page,
                )
            )

        return answer, citations
