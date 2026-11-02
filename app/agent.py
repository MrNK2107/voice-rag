"""
Swasthya Saathi agent: a small, offline tool-using loop on top of the health demo.

The agent works in steps. At each step a planner looks at what it knows so far
and picks ONE tool to call (or decides to finish). Every step is recorded in a
trace that the API returns and the web UI shows.

Tools (all offline, all deterministic):
  * triage            red-flag rules (English + Hindi) -> emergency or not
  * ask_clarifying    asks the user for a missing fact (e.g. the child's age)
  * facility_lookup   bundled table: which facility type / helpline fits the need
  * retrieve_guidance the grounded, cited RAG pipeline (abstains on weak evidence)

Planners:
  * RulePlanner (default): fixed, testable routing rules.
  * OllamaPlanner (opt-in with VOICE_RAG_AGENT_LLM=ollama): a small local model
    chooses the next tool through Ollama tool-calling. Its choice is validated;
    on any error or unknown tool it falls back to the RulePlanner.

Hard safety rules enforced by the loop itself, not by any planner:
  1. triage always runs first.
  2. A red flag short-circuits: the 108/102 advisory is set, the emergency
     helpline is looked up, cited first aid is added if the corpus has it, and
     the agent never asks a clarifying question or defers.
  3. Answers are only ever the extractive, cited text from retrieve_guidance.
     No planner writes medical content.

NOT medical advice. Demo content for a hackathon prototype.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from app.demo import ADVISORY_EN, ADVISORY_HI, red_flag_advisory
from app.guardrails import input_guard
from app.schemas import Citation

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
MAX_STEPS = 6

# --------------------------------------------------------------------------
# Bundled facility / helpline table. Facility *types* only (no addresses), so it
# is valid anywhere in India. Numbers are national helplines.
# --------------------------------------------------------------------------
HELPLINES: Dict[str, Dict[str, str]] = {
    "108": {"name": "Emergency ambulance", "name_hi": "आपातकालीन एम्बुलेंस"},
    "102": {"name": "Ambulance for pregnant women and infants (JSSK)", "name_hi": "गर्भवती महिलाओं और शिशुओं के लिए एम्बुलेंस"},
    "104": {"name": "Health information helpline (most states)", "name_hi": "स्वास्थ्य सूचना हेल्पलाइन (अधिकांश राज्य)"},
    "14416": {"name": "Tele-MANAS mental health helpline (24x7)", "name_hi": "टेली-मानस मानसिक स्वास्थ्य हेल्पलाइन (24x7)"},
    "1098": {"name": "Childline (child in distress)", "name_hi": "चाइल्डलाइन"},
    "181": {"name": "Women helpline", "name_hi": "महिला हेल्पलाइन"},
}

FACILITY_TABLE: Dict[str, Dict[str, Any]] = {
    "emergency": {
        "keywords": [],
        "facility": "Nearest government hospital with emergency care (CHC or district hospital)",
        "facility_hi": "आपातकालीन सेवा वाला नज़दीकी सरकारी अस्पताल (सीएचसी या ज़िला अस्पताल)",
        "helplines": ["108", "102"],
    },
    "mental_health": {
        "keywords": [r"suicid", r"kill myself", r"depress", r"anxiety", r"आत्महत्या", r"तनाव", r"डिप्रेशन"],
        "facility": "Tele-MANAS counsellor (phone), then the district mental health programme at the district hospital",
        "facility_hi": "टेली-मानस परामर्शदाता (फ़ोन), फिर ज़िला अस्पताल में ज़िला मानसिक स्वास्थ्य कार्यक्रम",
        "helplines": ["14416", "108"],
    },
    "tb": {
        "keywords": [r"\btb\b", r"tuberculosis", r"cough", r"टीबी", r"खाँसी", r"खांसी", r"बलगम"],
        "facility": "PHC / Ayushman Arogya Mandir for a free sputum test; TB treatment is free under the national TB programme",
        "facility_hi": "मुफ़्त बलगम जाँच के लिए पीएचसी / आयुष्मान आरोग्य मंदिर; टीबी का इलाज राष्ट्रीय टीबी कार्यक्रम में मुफ़्त है",
        "helplines": ["104"],
    },
    "pregnancy": {
        "keywords": [r"pregnan", r"delivery", r"antenatal", r"गर्भ", r"प्रसव"],
        "facility": "Sub-centre / ANM for antenatal check-ups; PHC or CHC with 24x7 delivery for childbirth",
        "facility_hi": "प्रसवपूर्व जाँच के लिए उप-केंद्र / एएनएम; प्रसव के लिए 24x7 सेवा वाला पीएचसी या सीएचसी",
        "helplines": ["102", "104"],
    },
    "vaccination": {
        "keywords": [r"vaccin", r"immuni[sz]", r"टीका", r"टीकाकरण"],
        "facility": "Sub-centre or the monthly Village Health and Nutrition Day (VHND) with the ANM",
        "facility_hi": "उप-केंद्र या मासिक ग्राम स्वास्थ्य एवं पोषण दिवस (वीएचएनडी) पर एएनएम",
        "helplines": ["104"],
    },
    "snakebite": {
        "keywords": [r"snake", r"साँप", r"सांप"],
        "facility": "CHC or district hospital (anti-snake venom is stocked at government hospitals)",
        "facility_hi": "सीएचसी या ज़िला अस्पताल (सरकारी अस्पतालों में एंटी-स्नेक वेनम उपलब्ध)",
        "helplines": ["108"],
    },
    "general": {
        "keywords": [],
        "facility": "ASHA worker first; then the sub-centre / Ayushman Arogya Mandir or PHC",
        "facility_hi": "पहले आशा कार्यकर्ता; फिर उप-केंद्र / आयुष्मान आरोग्य मंदिर या पीएचसी",
        "helplines": ["104"],
    },
}

_WHERE_RE = re.compile(
    r"\bwhere\b|\bhelpline\b|\bnumber\b|\bhospital\b|\bclinic\b|\bcall\b|कहाँ|कहां|हेल्पलाइन|नंबर|अस्पताल",
    re.IGNORECASE,
)

# Symptoms where the next safe step depends on the patient's age.
_AGE_DEPENDENT_RE = re.compile(r"\bfever\b|\bdiarrh?o?ea\b|\bloose motion|बुखार|दस्त", re.IGNORECASE)
_AGE_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|mo|year|years|yr|yrs|y|दिन|हफ़्ते|हफ्ते|महीने|महीना|साल|वर्ष)\b",
    re.IGNORECASE,
)
_ADULT_HINT_RE = re.compile(r"\b(i have|i've|i am|i'm|me|myself|adult)\b|मुझे", re.IGNORECASE)


def _lang(text: str) -> str:
    return "hi" if _DEVANAGARI.search(text or "") else "en"


def parse_age_months(text: str) -> Optional[float]:
    m = _AGE_RE.search(text or "")
    if not m:
        bare = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*", text or "")
        return float(bare.group(1)) * 12 if bare else None  # a bare number = years
    n, unit = float(m.group(1)), m.group(2).lower()
    if unit in {"day", "days", "दिन"}:
        return n / 30
    if unit in {"week", "weeks", "हफ़्ते", "हफ्ते"}:
        return n / 4.3
    if unit in {"month", "months", "mo", "महीने", "महीना"}:
        return n
    return n * 12


# --------------------------------------------------------------------------
# API models
# --------------------------------------------------------------------------
class AgentRequest(BaseModel):
    query: str
    # Answers to earlier clarifying questions, e.g. {"age": "8 months"}.
    slots: Dict[str, str] = Field(default_factory=dict)


class AgentStep(BaseModel):
    n: int
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    observation: str
    planner: str  # "rules" | "ollama:<model>" | "safety" (forced by the loop)
    ms: float


class AgentResponse(BaseModel):
    query: str
    status: str  # "answered" | "emergency" | "needs_info" | "abstained" | "blocked"
    answer: str
    advisory: Optional[str] = None
    clarifying_question: Optional[str] = None
    missing_slot: Optional[str] = None
    facility: Optional[Dict[str, Any]] = None
    citations: List[Citation] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    planner: str = "rules"
    total_ms: float = 0.0


# --------------------------------------------------------------------------
# Agent state + tools
# --------------------------------------------------------------------------
@dataclass
class AgentState:
    query: str
    slots: Dict[str, str]
    lang: str
    triaged: bool = False
    emergency: bool = False
    advisory: Optional[str] = None
    facility: Optional[Dict[str, Any]] = None
    rag: Any = None  # RagResponse from retrieve_guidance
    clarifying_question: Optional[str] = None
    missing_slot: Optional[str] = None
    blocked_reason: Optional[str] = None
    crisis: bool = False
    called: List[str] = field(default_factory=list)


def facility_lookup(need: str, lang: str = "en") -> Dict[str, Any]:
    row = FACILITY_TABLE.get(need) or FACILITY_TABLE["general"]
    hi = lang == "hi"
    return {
        "need": need if need in FACILITY_TABLE else "general",
        "facility": row["facility_hi"] if hi else row["facility"],
        "helplines": [{"number": n, "name": HELPLINES[n]["name_hi" if hi else "name"]} for n in row["helplines"]],
    }


def classify_need(text: str) -> str:
    for need, row in FACILITY_TABLE.items():
        if any(re.search(k, text, re.IGNORECASE) for k in row["keywords"]):
            return need
    return "general"


def triage(query: str, slots: Dict[str, str]) -> Dict[str, Any]:
    """Red-flag rules. Also: any fever in a young infant (< 2 months) is a danger
    sign in the national IMNCI guidelines, so the age slot can escalate."""
    advisory = red_flag_advisory(query)
    reason = "red-flag phrase" if advisory else None
    age = parse_age_months(slots.get("age", "")) if slots.get("age") else None
    if not advisory and age is not None and age < 2 and re.search(r"fever|बुखार", query, re.IGNORECASE):
        advisory = ADVISORY_HI if _lang(query) == "hi" else ADVISORY_EN
        reason = "fever in an infant under 2 months"
    return {"emergency": bool(advisory), "advisory": advisory, "reason": reason}


def clarifying_question_for(state: AgentState) -> Optional[Dict[str, str]]:
    """Return {"slot", "question"} when an age-dependent symptom lacks an age."""
    q = state.query
    if "age" in state.slots or not _AGE_DEPENDENT_RE.search(q):
        return None
    if _AGE_RE.search(q) or _ADULT_HINT_RE.search(q):
        return None
    if state.lang == "hi":
        return {"slot": "age", "question": "मरीज़ की उम्र कितनी है? (जैसे 6 महीने, 3 साल, 30 साल)"}
    return {"slot": "age", "question": "How old is the patient? (for example 6 months, 3 years, 30 years)"}


# --------------------------------------------------------------------------
# Planners
# --------------------------------------------------------------------------
TOOL_NAMES = ("ask_clarifying", "facility_lookup", "retrieve_guidance", "finish")


class RulePlanner:
    name = "rules"

    def next_action(self, state: AgentState) -> Dict[str, Any]:
        if state.rag is None and "ask_clarifying" not in state.called:
            cq = clarifying_question_for(state)
            if cq:
                return {"tool": "ask_clarifying", "args": cq}
        if state.facility is None and _WHERE_RE.search(state.query):
            return {"tool": "facility_lookup", "args": {"need": classify_need(state.query)}}
        if state.rag is None:
            return {"tool": "retrieve_guidance", "args": {"query": state.query}}
        # Recovery: the corpus has no guidance for a real symptom question, so
        # point the user to the right first contact instead of a bare abstain.
        if (state.rag.abstained and state.facility is None
                and (state.slots or _AGE_DEPENDENT_RE.search(state.query))):
            return {"tool": "facility_lookup", "args": {"need": classify_need(state.query)}}
        return {"tool": "finish", "args": {}}


class OllamaPlanner:
    """Lets a small local model choose the next tool via Ollama tool-calling.

    Only used when VOICE_RAG_AGENT_LLM=ollama. Never pulls a model. Any failure
    (server down, model missing, malformed or unknown tool) falls back to the
    RulePlanner for that step, and the trace says so.
    """

    def __init__(self, model: str, host: str = "http://localhost:11434", timeout: float = 20.0,
                 post: Optional[Callable[[str, Dict[str, Any]], Dict[str, Any]]] = None):
        self.model = model
        self.host = host.rstrip("/")
        self.timeout = timeout
        self.name = f"ollama:{model}"
        self._post = post or self._http_post
        self._fallback = RulePlanner()

    def _http_post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:  # noqa: S310 (local Ollama only)
            return json.loads(r.read().decode())

    @staticmethod
    def tool_schemas() -> List[Dict[str, Any]]:
        def fn(name, desc, props, required):
            return {"type": "function", "function": {"name": name, "description": desc,
                    "parameters": {"type": "object", "properties": props, "required": required}}}

        return [
            fn("ask_clarifying", "Ask the user ONE question when a needed fact (the patient's age) is missing.",
               {"slot": {"type": "string", "enum": ["age"]}, "question": {"type": "string"}}, ["slot", "question"]),
            fn("facility_lookup", "Find which facility type / helpline fits the need.",
               {"need": {"type": "string", "enum": [k for k in FACILITY_TABLE if k != "emergency"]}}, ["need"]),
            fn("retrieve_guidance", "Search the cited health corpus for guidance.",
               {"query": {"type": "string"}}, ["query"]),
            fn("finish", "Stop: enough information has been gathered.", {}, []),
        ]

    def next_action(self, state: AgentState) -> Dict[str, Any]:
        done = ", ".join(t for t in state.called if t != "triage") or "none"
        prompt = (
            "You route a health helpline assistant. Pick exactly one tool. Rules: ask_clarifying only if the "
            "question is about fever or diarrhoea and the age is unknown; facility_lookup if the user asks where "
            "to go or whom to call; retrieve_guidance once before finishing; finish when guidance was retrieved.\n"
            f"Question: {state.query}\nKnown answers: {json.dumps(state.slots, ensure_ascii=False)}\n"
            f"Tools already called: {done}"
        )
        try:
            out = self._post(f"{self.host}/api/chat", {
                "model": self.model, "stream": False, "options": {"temperature": 0},
                "messages": [{"role": "user", "content": prompt}], "tools": self.tool_schemas(),
            })
            calls = (out.get("message") or {}).get("tool_calls") or []
            if not calls:
                raise ValueError("no tool call")
            fn = calls[0]["function"]
            tool, args = fn["name"], fn.get("arguments") or {}
            if isinstance(args, str):
                args = json.loads(args)
            if tool not in TOOL_NAMES:
                raise ValueError(f"unknown tool {tool!r}")
            # Small models tend to repeat the same call; each tool runs at most once.
            if tool != "finish" and tool in state.called:
                raise ValueError(f"repeated tool {tool!r}")
            if tool == "ask_clarifying":
                # The model may only ask for a slot the rules agree is missing.
                cq = clarifying_question_for(state)
                if not cq or "ask_clarifying" in state.called:
                    raise ValueError("clarification not needed")
                args = cq if state.lang == "hi" else {"slot": cq["slot"], "question": args.get("question") or cq["question"]}
            if tool == "facility_lookup":
                if args.get("need") not in FACILITY_TABLE:
                    args = {"need": classify_need(state.query)}
            if tool == "retrieve_guidance":
                args = {"query": state.query}  # the model does not rewrite the query
            if tool == "finish" and state.rag is None:
                raise ValueError("finish before retrieval")
            return {"tool": tool, "args": args}
        except Exception as e:  # noqa: BLE001
            action = self._fallback.next_action(state)
            action["fallback"] = f"{type(e).__name__}: {e}"[:120]
            return action


def planner_from_env() -> Any:
    if os.environ.get("VOICE_RAG_AGENT_LLM", "").strip().lower() == "ollama":
        return OllamaPlanner(model=os.environ.get("VOICE_RAG_AGENT_MODEL", "llama3.2:3b"),
                             host=os.environ.get("OLLAMA_HOST", "http://localhost:11434"))
    return RulePlanner()


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------
class HealthAgent:
    def __init__(self, harness, planner=None, max_steps: int = MAX_STEPS):
        self.harness = harness
        self.planner = planner or RulePlanner()
        self.max_steps = max_steps

    def run(self, query: str, slots: Optional[Dict[str, str]] = None) -> AgentResponse:
        t0 = time.perf_counter()
        query = (query or "").strip()
        state = AgentState(query=query, slots=dict(slots or {}), lang=_lang(query))
        steps: List[AgentStep] = []

        def record(tool, args, obs, planner, started):
            steps.append(AgentStep(n=len(steps) + 1, tool=tool, args=args, observation=obs, planner=planner,
                                   ms=round((time.perf_counter() - started) * 1000, 3)))
            state.called.append(tool)

        # Step 1 (forced): input guard + triage. Not delegated to any planner.
        s = time.perf_counter()
        ok, reason = input_guard(query)
        if not ok:
            state.blocked_reason = reason
            crisis = re.search(r"suicid|kill myself|आत्महत्या", query, re.IGNORECASE)
            if crisis:
                state.crisis = True
                state.facility = facility_lookup("mental_health", state.lang)
            record("input_guard", {}, f"blocked: {reason}" + (" -> crisis helpline" if crisis else ""), "safety", s)
            return self._finish(state, steps, t0)

        s = time.perf_counter()
        tri = triage(query, state.slots)
        state.triaged = True
        state.emergency, state.advisory = tri["emergency"], tri["advisory"]
        record("triage", {"slots": state.slots} if state.slots else {},
               f"EMERGENCY ({tri['reason']})" if tri["emergency"] else "no red flag", "safety", s)

        if state.emergency:
            # Hard short-circuit: helpline + cited first aid, no questions, no planner.
            s = time.perf_counter()
            state.facility = facility_lookup("emergency", state.lang)
            record("facility_lookup", {"need": "emergency"}, "108 / 102, nearest government hospital", "safety", s)
            s = time.perf_counter()
            state.rag = self.harness.ask_text(query)
            record("retrieve_guidance", {"query": query}, self._rag_obs(state.rag), "safety", s)
            return self._finish(state, steps, t0)

        for _ in range(self.max_steps):
            action = self.planner.next_action(state)
            tool, args = action["tool"], action.get("args", {})
            planner_name = self.planner.name + (" (fallback: rules)" if action.get("fallback") else "")
            s = time.perf_counter()
            if tool == "finish":
                record("finish", {}, "done", planner_name, s)
                break
            if tool == "ask_clarifying":
                state.clarifying_question, state.missing_slot = args["question"], args["slot"]
                record(tool, args, f"waiting for user: {args['slot']}", planner_name, s)
                break
            if tool == "facility_lookup":
                state.facility = facility_lookup(args.get("need", "general"), state.lang)
                nums = ", ".join(h["number"] for h in state.facility["helplines"])
                record(tool, args, f"{state.facility['need']}: helplines {nums}", planner_name, s)
                continue
            if tool == "retrieve_guidance":
                state.rag = self.harness.ask_text(query)
                record(tool, {"query": query}, self._rag_obs(state.rag), planner_name, s)
                continue
            record(tool, args, "ignored: unknown tool", planner_name, s)
            break
        return self._finish(state, steps, t0)

    @staticmethod
    def _rag_obs(rag) -> str:
        if rag.abstained:
            return f"abstained: {rag.abstain_reason}"
        return f"{len(rag.citations)} cited passage(s): " + ", ".join(
            sorted({c.parent_doc_id or c.chunk_id for c in rag.citations}))

    def _finish(self, state: AgentState, steps: List[AgentStep], t0: float) -> AgentResponse:
        rag = state.rag
        citations = list(rag.citations) if rag is not None and not rag.abstained else []
        if state.blocked_reason:
            status = "blocked"
            answer = ("Please talk to someone now. Call Tele-MANAS 14416 (free, 24x7) or 108 in an emergency."
                      if state.crisis and state.lang == "en" else
                      "कृपया अभी किसी से बात करें। टेली-मानस 14416 (मुफ़्त, 24x7) या आपातकाल में 108 पर फ़ोन करें।"
                      if state.crisis else "I cannot process that request.")
        elif state.emergency:
            status = "emergency"
            answer = rag.answer if rag is not None and not rag.abstained else state.advisory
        elif state.clarifying_question:
            status, answer = "needs_info", state.clarifying_question
        elif rag is not None and not rag.abstained:
            status, answer = "answered", rag.answer
        elif state.facility:
            status, answer = "answered", state.facility["facility"]
        else:
            status = "abstained"
            answer = rag.answer if rag is not None else "I could not find guidance for this."
        return AgentResponse(
            query=state.query, status=status, answer=answer, advisory=state.advisory,
            clarifying_question=state.clarifying_question, missing_slot=state.missing_slot,
            facility=state.facility, citations=citations, steps=steps,
            planner=getattr(self.planner, "name", "rules"),
            total_ms=round((time.perf_counter() - t0) * 1000, 3),
        )
