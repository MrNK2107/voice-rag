import re
import time
from typing import Dict, Optional

from app.config import settings
from app.generator import AnswerGenerator, query_coverage
from app.citations import verify_answer
from app.collection_store import CollectionNotFound
from app.guardrails import grounding_check, input_guard, retrieval_guard
from app.i18n import message
from app.langid import LANGUAGE_NAMES, base_language, detect_language, is_rtl, normalize_code
from app.safety import output_guard, scan_passages
from app.latency import timed_stage
from app.retriever import HybridRetriever
from app.schemas import RagResponse
from app.stt_sarvam import SarvamSTT
from app.triage import crisis_advisory, emergency_advisory

_LANG_TAG = re.compile(r"[A-Za-z]{2,3}(?:-[A-Za-z0-9]{2,8})*")


class VoiceRAGHarness:
    """The grounded answer pipeline:

    input guard -> retrieval (hybrid dense + FTS5) -> passage injection scan ->
    retrieval guard -> extractive answer -> grounding, coverage, citation and
    output checks -> (optional) translation -> language labels + emergency triage.

    ``collections`` (app/collection_store.CollectionStore) adds named
    collections; ``collection=None`` uses the legacy default index.
    """

    def __init__(
        self,
        stt: Optional[SarvamSTT] = None,
        retriever: Optional[HybridRetriever] = None,
        generator: Optional[AnswerGenerator] = None,
        collections=None,
        translator=None,
        question_log=None,
    ):
        # Components are injectable so tests / benchmarks can run fully offline.
        self.stt = stt or SarvamSTT()
        self.retriever = retriever or HybridRetriever()
        self.generator = generator or AnswerGenerator()
        self.collections = collections
        self.translator = translator
        self.question_log = question_log

    # ---- helpers -------------------------------------------------------------
    def _retriever_for(self, collection: Optional[str]):
        if collection:
            if self.collections is None:
                raise CollectionNotFound(collection)
            return self.collections.retriever(collection)  # raises CollectionNotFound
        return self.retriever

    def _corpus_name(self, collection: Optional[str]) -> str:
        if collection and self.collections is not None:
            return self.collections.title(collection)
        return settings.corpus_name

    @staticmethod
    def _language(text: str, hint: Optional[str]) -> str:
        # The hint comes from the client and ends up in the translator prompt,
        # so only well-formed tags for languages we know are accepted.
        if hint and hint != "auto":
            code = normalize_code(hint)
            if _LANG_TAG.fullmatch(code) and (code in LANGUAGE_NAMES or base_language(code) in LANGUAGE_NAMES):
                return code
        return detect_language(text).code

    def _finish(self, resp: RagResponse, lang: str, collection: Optional[str], channel: str) -> RagResponse:
        resp.language = lang
        resp.language_name = LANGUAGE_NAMES.get(lang, LANGUAGE_NAMES.get(base_language(lang), lang))
        resp.rtl = is_rtl(lang)
        resp.collection = collection
        if resp.citations and not resp.answer_language:
            cited = resp.citations[0]
            resp.answer_language = (normalize_code(cited.language) if cited.language
                                    else detect_language(cited.quote).code)
        elif not resp.answer_language:
            resp.answer_language = lang
        if settings.emergency_triage and resp.advisory is None and resp.transcript:
            # Crisis help is shown even when the guard blocked the question.
            resp.advisory = crisis_advisory(resp.transcript, lang)
            blocked = resp.abstain_reason and "blocked" in resp.abstain_reason.lower()
            if resp.advisory is None and not blocked:
                resp.advisory = emergency_advisory(resp.transcript, lang)
        if self.question_log is not None and settings.analytics_enabled:
            try:
                self.question_log.record_response(resp, collection, channel)
            except Exception as e:  # analytics must never break an answer
                print(f"analytics error: {e}")
        return resp

    # ---- audio ---------------------------------------------------------------
    def ask_audio(self, audio_bytes: bytes, filename: str, content_type: str,
                  collection: Optional[str] = None, language: Optional[str] = None,
                  stt=None) -> RagResponse:
        timings: Dict[str, float] = {}
        total_start = time.perf_counter()
        hint = language if language and language != "auto" else None
        stt = stt or self.stt

        if not audio_bytes:
            # Nothing to transcribe: don't spend an STT call (and don't let the
            # offline scripted STT invent a question for an empty upload).
            timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 3)
            resp = RagResponse(
                transcript="",
                answer=message("no_audio", hint),
                citations=[],
                grounded=True,
                abstained=True,
                abstain_reason="Empty audio received.",
                timings_ms=timings,
            )
            return self._finish(resp, hint or "en", collection, "voice")

        with timed_stage("stt_ms", timings):
            try:
                transcript = stt.transcribe(audio_bytes, filename, content_type)
            except Exception as e:
                timings["total_ms"] = round((time.perf_counter() - total_start) * 1000, 3)
                resp = RagResponse(
                    transcript="",
                    answer=message("stt_failed", hint),
                    citations=[],
                    grounded=False,
                    abstained=True,
                    abstain_reason=f"STT Error: {str(e)}",
                    timings_ms=timings,
                )
                return self._finish(resp, hint or "en", collection, "voice")

        # Prefer the language the STT provider heard; else detect from the text.
        stt_lang = getattr(stt, "last_language", None)
        response = self.ask_text(transcript, timings, collection=collection,
                                 language=hint or stt_lang, channel="voice")
        response.stt_provider = getattr(stt, "name", type(stt).__name__)
        response.timings_ms["total_ms"] = round(
            (time.perf_counter() - total_start) * 1000,
            3,
        )
        return response

    # ---- text ----------------------------------------------------------------
    def ask_text(self, query: str, timings: Optional[Dict[str, float]] = None,
                 collection: Optional[str] = None, language: Optional[str] = None,
                 channel: str = "text") -> RagResponse:
        timings = timings or {}
        rag_start = time.perf_counter()
        retriever = self._retriever_for(collection)
        corpus = self._corpus_name(collection)

        transcript = (query or "").strip()
        # A prefix is enough to detect the language, and stays cheap for the
        # oversized queries the input guard is about to refuse.
        lang = self._language(transcript[:2000], language)

        def done(resp: RagResponse) -> RagResponse:
            timings["post_stt_total_ms"] = round((time.perf_counter() - rag_start) * 1000, 3)
            if "total_ms" not in timings:
                timings["total_ms"] = timings["post_stt_total_ms"]
            resp.timings_ms = timings
            return self._finish(resp, lang, collection, channel)

        with timed_stage("input_guard_ms", timings):
            ok, reason = input_guard(transcript)

        if not ok:
            return done(RagResponse(
                transcript=transcript,
                answer=message("blocked", lang),
                citations=[],
                grounded=True,
                abstained=True,
                abstain_reason=reason,
            ))

        with timed_stage("retrieval_ms", timings):
            contexts, confidence = retriever.retrieve(transcript)

        quarantined = []
        if settings.scan_retrieved_passages:
            # A poisoned document in the index must not reach the generator.
            contexts, flagged = scan_passages(contexts)
            quarantined = [f"{cid}:{pat}" for cid, pat in flagged]

        with timed_stage("retrieval_guard_ms", timings):
            ok, reason = retrieval_guard(contexts, confidence)

        if not ok:
            return done(RagResponse(
                transcript=transcript,
                answer=message("no_context", lang, corpus=corpus),
                citations=[],
                grounded=True,
                abstained=True,
                abstain_reason=reason,
                quarantined_chunks=quarantined,
            ))

        with timed_stage("generation_ms", timings):
            answer, citations = self.generator.generate_extractive(transcript, contexts)

        with timed_stage("grounding_ms", timings):
            grounded = grounding_check(answer, contexts)
            reason = None if grounded else "Grounding validation failed."
            if grounded and settings.min_query_coverage > 0:
                coverage, missing = query_coverage(transcript, answer)
                if coverage < settings.min_query_coverage:
                    grounded = False
                    reason = (f"The retrieved passages do not answer this question "
                              f"(answer covers {coverage:.0%} of it; missing: {', '.join(missing[:5])}).")
            if grounded and settings.citation_check == "strict":
                check = verify_answer(answer, citations, contexts, settings.citation_min_coverage)
                if not check.ok:
                    grounded = False
                    reason = f"Citation verification failed: {check.reason}"
            if grounded and settings.output_guard_enabled:
                ok, why = output_guard(answer, contexts)
                if not ok:
                    grounded = False
                    reason = why

        translated, original = False, None
        if not grounded:
            if reason and reason.startswith("The retrieved passages do not answer"):
                answer = message("unanswerable", lang, corpus=corpus)
            else:
                answer = message("ungrounded", lang)
            citations = []
            abstained = True
        else:
            abstained = False
            # Translation happens only after every check passed on the quote.
            src_lang = normalize_code(citations[0].language) if citations and citations[0].language else None
            if (self.translator is not None and src_lang
                    and base_language(src_lang) != base_language(lang) and base_language(lang) != "und"):
                try:
                    with timed_stage("translation_ms", timings):
                        original, answer = answer, self.translator.translate(answer, lang, src_lang)
                    translated = True
                except Exception as e:
                    print(f"translation failed, returning the original quote: {e}")
                    original = None

        resp = RagResponse(
            transcript=transcript,
            answer=answer,
            citations=citations,
            grounded=grounded,
            abstained=abstained,
            abstain_reason=reason,
            quarantined_chunks=quarantined,
            translated=translated,
            original_answer=original,
        )
        if translated:
            resp.answer_language = lang
        return done(resp)
