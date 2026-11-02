"""Prompt-injection detection for user input AND retrieved passages, plus an
output-side check.

What this is: a deterministic, pattern-based filter with text normalisation.
What it is not: a classifier. A paraphrase that avoids every pattern below will
get through: on a held-out set written after these patterns were frozen it
blocks 2/10 (see `heldout_attacks` in tests/data/adversarial_prompts.json and
storage/guard_eval.json). For user input, a small learned second stage
(app/injection_clf.py) runs after these patterns: pattern OR classifier blocks
8/10 there. The main defence in this codebase is still
architectural: the answer generator is extractive, so no LLM ever reads the
retrieved text or the user's instructions. These checks are defence in depth,
and the output check is what would keep a future LLM generator honest.

Normalisation (applied before matching):
  * Unicode NFKC (folds full-width letters, ligatures, compatibility forms)
  * zero-width / bidi / soft-hyphen characters removed
  * common Cyrillic / Greek look-alike letters mapped to Latin ("іgnоrе")
  * Devanagari nukta removed (ज़ == ज) so spelling variants match
  * casefold + whitespace collapsed
  * a second "leet" view (1->i, 0->o, 3->e, 4->a, 5->s, 7->t, @->a, $->s)
  * a third "squashed" view with every non-letter removed, matched only against
    long multi-word phrases ("i g n o r e  p r e v i o u s ...")
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, List, Optional, Sequence, Tuple

_ZERO_WIDTH = dict.fromkeys(map(ord, "​‌‍‎‏⁠⁡⁢⁣⁤"
                                     "﻿­͏᠎‪‫‬‭‮"
                                     "⁦⁧⁨⁩"), None)

# Look-alike letters most often used to dodge ASCII keyword filters.
_CONFUSABLES = str.maketrans({
    # Cyrillic
    "а": "a", "в": "b", "е": "e", "ё": "e", "к": "k", "м": "m", "н": "h", "о": "o", "р": "p",
    "с": "c", "т": "t", "у": "y", "х": "x", "і": "i", "ї": "i", "ј": "j", "ѕ": "s", "ԁ": "d",
    "ɡ": "g", "ԛ": "q", "ԝ": "w", "һ": "h", "ӏ": "l",
    # Greek
    "α": "a", "β": "b", "ε": "e", "η": "n", "ι": "i", "κ": "k", "ν": "v", "ο": "o", "ρ": "p",
    "τ": "t", "υ": "u", "χ": "x", "γ": "y", "ω": "w",
    # Latin variants
    "ı": "i", "ł": "l", "ø": "o", "ß": "ss",
})
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"})
_NUKTA = "़"
_NON_LETTERS = re.compile(r"[^a-zऀ-ॿ]+")


def normalize_text(text: str) -> str:
    """Canonical form used for all pattern matching (never shown to users)."""
    t = unicodedata.normalize("NFKC", text or "")
    t = t.translate(_ZERO_WIDTH)
    t = t.casefold().translate(_CONFUSABLES)
    # Strip combining accents on Latin letters (e.g. "ïgnöre"), keep Indic marks.
    t = "".join(ch for ch in unicodedata.normalize("NFD", t)
                if not (unicodedata.combining(ch) and not ("ऀ" <= ch <= "෿")))
    t = unicodedata.normalize("NFC", t).replace(_NUKTA, "")
    return re.sub(r"\s+", " ", t).strip()


def _views(text: str) -> Tuple[str, str, str]:
    norm = normalize_text(text)
    leet = norm.translate(_LEET)
    squashed = _NON_LETTERS.sub("", leet)
    return norm, leet, squashed


_SEP = r"[\s\W_]*"  # separators allowed between words (also lets squashed text match)

# Each entry: (name, regex). Patterns are matched against the normalised and
# leet views; entries in _SQUASH_SAFE also against the squashed view.
_VERBS_IGNORE = r"(?:ignore|disregard|forget|override|overrule|bypass|skip|drop|discard|abandon|neglect)"
_QUAL = r"(?:(?:all|any|the|your|my|these|those|previous|prior|preceding|earlier|above|old|original|initial|existing|system|safety|other)" + _SEP + r"){0,4}"
_TARGET = r"(?:instructions?|rules?|prompts?|guidelines?|directions?|directives?|constraints?|polic(?:y|ies)|guardrails?|restrictions?|filters?|programming|context|commands?)"

INJECTION_PATTERNS: List[Tuple[str, str]] = [
    ("ignore_instructions", _VERBS_IGNORE + _SEP + _QUAL + _TARGET),
    ("instructions_ignore", r"(?:previous|prior|above|earlier|system)" + _SEP + _TARGET + _SEP + r"(?:are|is)?" + _SEP + r"(?:void|cancell?ed|obsolete|no longer apply)"),
    ("reveal_prompt", r"(?:reveal|show|print|display|repeat|leak|output|dump|tell me|give me|what (?:is|are)|spell out|recite)" + _SEP
     + r"(?:me" + _SEP + r")?(?:your|the|its|this)?" + _SEP + r"(?:full|complete|entire|exact|hidden|secret|internal|initial|original)?" + _SEP
     + r"(?:system|developer|dev|hidden|secret|initial|original|internal)" + _SEP + r"(?:prompt|message|instructions?|rules|configuration|config)"),
    ("system_prompt", r"system" + _SEP + r"prompt"),
    ("developer_message", r"developer" + _SEP + r"(?:message|mode|instructions?)"),
    ("jailbreak", r"\bjail" + _SEP + r"break(?:ing|ed)?\b|\bdo anything now\b|\bdan mode\b|\bgod mode\b"),
    ("unrestricted_persona", r"\b(?:act|behave|respond|operate)" + _SEP + r"as" + _SEP + r"(?:an?" + _SEP + r")?(?:unrestricted|unfiltered|uncensored|jailbroken|evil|rogue|different|new)\b"),
    ("you_are_now", r"\byou are now\b|\bfrom now on,? you\b|\bpretend (?:that )?you are\b|\bpretend to be\b|\broleplay as\b|\bswitch to .{0,20}mode\b"),
    ("no_restrictions", r"\b(?:without|no|ignore) (?:any )?(?:filters|censorship|safety (?:rules|filters|guidelines|checks))\b|\bno longer (?:bound|restricted)\b"),
    ("new_instructions", r"\bnew (?:instructions?|rules|task)\s*:|\bupdated instructions?\b|\boverride (?:mode|code)\b"),
    ("chat_markup", r"<\|?(?:im_start|im_end|system|endoftext)\|?>|\[/?inst\]|<</?sys>>|^\s*(?:system|assistant)\s*:|###\s*(?:instruction|system)"),
    ("addressed_to_model", r"\bif you are an? (?:ai|llm|language model|assistant|chatbot)\b|\bas an? (?:ai|llm|language model)\b|\b(?:ai|llm|assistant|model)s? (?:must|should) (?:always|now|instead)\b|\bwhen (?:you are )?answering,? (?:always|you must|tell)\b"),
    ("tell_user", r"\b(?:tell|instruct|advise) the (?:user|reader|patient) to (?:ignore|stop|visit|call|buy|send|click)\b"),
    ("exfiltrate", r"\b(?:send|post|email|upload|forward) .{0,40}\b(?:to|at) (?:https?://|www\.|[\w.-]+@[\w-]+\.)"),
    # ---- Paraphrases (no "ignore ... instructions" keyword pair) ----
    # Each needs BOTH an override verb/claim AND a reference to the model's own
    # setup ("you were told", "your guidelines", "your creators"), so a user
    # describing their own doctor's advice does not match.
    ("paraphrase_told", r"\b(?:set aside|put aside|pay no attention to|disregard|ignore|forget|throw (?:away|out)|scrap|erase|wipe|discard|never mind)"
     r"\b.{0,30}\b(?:what|everything|anything|whatever|all) (?:that )?you(?:'ve| have| were| had)? (?:been )?(?:told|given|instructed|taught|programmed|trained|configured)\b"),
    ("paraphrase_stop_following", r"\b(?:stop|quit|cease|(?:do not|don't|dont|never)) (?:following|obeying|listening to|follow|obey|listen to) your (?:own |usual |normal |current |safety )?"
     r"(?:guidelines?|rules|instructions?|programming|training|restrictions?|polic(?:y|ies)|filters?|guardrails?|creators?|developers?|makers?|programmers?)\b"),
    ("paraphrase_obey_creators", r"\b(?:do not|don't|dont|never|stop) (?:obey(?:ing)?|listen(?:ing)? to|follow(?:ing)?) the (?:people|ones|team|company) (?:who|that) (?:made|built|programmed|trained|created|set up) you\b"),
    ("paraphrase_rules_void", r"\byour (?:usual |normal |own |current |earlier |original )?(?:rules|guidelines|restrictions|limits|instructions|policies|filters|guardrails|programming|setup|training) (?:do not|don't|dont|no longer|won't|will not|shouldn't) (?:apply|matter|count|exist|bind)"),
    ("paraphrase_before_void", r"\b(?:everything|anything|all|whatever|the text|the messages?) (?:before|above|prior to|earlier than) (?:this|here|now)\b.{0,25}\b(?:no longer|doesn't|does not|don't|do not|won't|is not|isn't)\b.{0,6}\b(?:matters?|count|apply|valid|relevant)"),
    ("paraphrase_permission", r"\byou(?:'re| are) (?:now )?(?:free|allowed|permitted|authori[sz]ed) to (?:break|ignore|bypass|skip|drop|disregard|forget) (?:your|the|all|any)\b"),
    ("paraphrase_as_if", r"\b(?:answer|respond|reply|act|behave|talk) as (?:if|though) (?:you|there) (?:had|have|has|were|are|was) (?:no|zero|without) (?:safety |content |usual )?(?:rules|limits|restrictions|filters|guidelines|guardrails|instructions)"),
    ("paraphrase_reset", r"\b(?:throw away|discard|erase|wipe|scrap|reset|delete|drop) (?:your|all your) (?:earlier |previous |original |old |initial )?(?:setup|set-up|configuration|programming|training|persona|role|memory|identity)\b"),
    ("paraphrase_identity", r"\byou(?:'re| are) (?:now )?no longer (?:an? |the )?(?:\w+ ){0,2}(?:assistant|bot|ai|chatbot|model|bound|restricted|limited)\b"),
    ("paraphrase_creators", r"\b(?:whatever|what|everything) your (?:creators?|developers?|makers?|programmers?|owners?) (?:told|taught|instructed|asked|want(?:ed)?) you\b"),
    # ---- Hindi (Devanagari; nukta already stripped by normalisation) ----
    ("hi_ignore_instructions", r"(?:पिछले|पहले|पुराने|सभी|सारे|ऊपर|अपने)?\s*(?:के\s*)?(?:सभी\s*|सारे\s*)?(?:निर्देश|निर्देशों|नियम|नियमों|हिदायत|हिदायतों|आदेश|आदेशों)\s*(?:को\s*)?(?:अनदेखा|नजरअंदाज|नजर अंदाज|भूल|भुला|मत मानो|न मानो|छोड़|छोड|तोड़|तोड|हटा)"),
    ("hi_reveal_prompt", r"(?:सिस्टम|छिपे|गुप्त|डेवलपर)\s*(?:हुए\s*)?(?:प्रॉम्प्ट|प्रोम्प्ट|प्रांप्ट|संदेश|मैसेज|निर्देश)"),
    ("hi_persona", r"अब\s*(?:से\s*)?(?:तुम|आप)\s*(?:एक\s*)?(?:बिना\s*(?:किसी\s*)?(?:नियम|रोक|पाबंदी))|बिना\s*(?:किसी\s*)?(?:नियम|रोक|पाबंदी|सेंसर)\s*के\s*(?:जवाब|उत्तर)"),
    # ---- Hinglish (romanised Hindi) ----
    ("hinglish_ignore", r"\b(?:pichhle|pichle|purane|sabhi|saare|sare|apne)?\s*(?:instructions?|nirdesh|niyam|rules?)\s*(?:ko\s*)?(?:ignore|bhool|bhul|nazarandaz|najarandaz|mat maano|mat mano|chhod|chod)\b"),
    ("hinglish_reveal", r"\b(?:system|secret|hidden)\s*prompt\s*(?:dikhao|batao|bata|dikha)\b"),
    # ---- Tamil (script U+0B80-U+0BFF); target noun stem then verb (SOV) ----
    ("ta_ignore_instructions", r"(?:வழிமுறை|அறிவுறுத்தல்|அறிவுறுத்த|விதி|கட்டளை|நெறிமுறை|கட்டுப்பாடு)\S*\s*(?:\S+\s*){0,2}?"
     r"(?:புறக்கணி|மற(?:ந்து|க்க|ந்)|கவனிக்கா|கவனிக்க\s*வேண்டாம்|பின்பற்றா|பின்பற்ற\s*வேண்டாம்|பின்பற்றாதே|மீறு|மீற|உதாசீன|விட்டுவிடு|தூக்கி\s*எறி)"),
    ("ta_reveal_prompt", r"(?:சிஸ்டம்|சிஸ்டெம்|கணினி|மறைக்கப்பட்ட|மறைந்த|ரகசிய|டெவலப்பர்)\s*(?:\S+\s*)?(?:ப்ராம்ப்ட்|ப்ராம்ட்|பிராம்ப்ட்|ப்ரொம்ப்ட்|அறிவுறுத்தல்|வழிமுறை|கட்டளை|செய்தி)"),
    ("ta_persona", r"(?:இனிமேல்|இனி|இப்போதிலிருந்து)\s*(?:\S+\s*){0,4}?(?:விதி|கட்டுப்பாடு|வரம்பு|தடை)\S*\s*(?:\S+\s*)?(?:இல்லாமல்|இன்றி)"),
    # ---- Tanglish (romanised Tamil) ----
    ("tanglish_ignore", r"\b(?:instructions?|rules?|niyamam|vidhi(?:gal)?|kattalai)\s*(?:ellam|ellathaiyum|ellathayum|ellaam|ah|a|-a)?\s*(?:ignore|marandhu|marandu|maranthu|marakka|vidu|vittudu|follow panna vendam|kandukadha|kandukatha)"),
]

# ---- Other world languages --------------------------------------------------
# Matched against NFKC + casefold text only (no look-alike folding, which would
# turn genuine Cyrillic/Greek words into Latin noise). Each needs an override
# verb plus an "instructions/rules/prompt" noun, or a system-prompt request.
# Coverage is deliberately narrow: the common "ignore previous instructions"
# and "reveal your system prompt" forms per language. Not measured on a
# held-out set; see GUARDRAILS.md.
WORLD_PATTERNS: List[Tuple[str, str]] = [
    ("es_ignore", r"\b(?:ignora|ignore|olvida|olvide|omite|descarta|desobedece)\w*\b.{0,40}\b(?:instrucciones|reglas|indicaciones|órdenes|prompt)"),
    ("pt_ignore", r"\b(?:ignore|ignora|esqueça|esqueca|desconsidere|descarte)\w*\b.{0,40}\b(?:instruções|instrucoes|regras|ordens|prompt)"),
    ("fr_ignore", r"\b(?:ignore|ignorez|oublie|oubliez|néglige|negligez|ne tiens pas compte)\w*\b.{0,40}\b(?:instructions|règles|regles|consignes|directives|prompt)"),
    ("de_ignore", r"\b(?:ignorier|vergiss|vergessen|missachte|übergehe)\w*\b.{0,40}\b(?:anweisungen|regeln|vorgaben|befehle|prompt)"),
    ("it_ignore", r"\b(?:ignora|ignori|dimentica|dimentichi|trascura)\w*\b.{0,40}\b(?:istruzioni|regole|direttive|prompt)"),
    ("id_ignore", r"\b(?:abaikan|lupakan|acuhkan)\b.{0,40}\b(?:instruksi|aturan|perintah|prompt)"),
    ("tr_ignore", r"\b(?:talimatları|kuralları|komutları)\b.{0,30}\b(?:yok say|görmezden gel|unut)"),
    ("sw_ignore", r"\b(?:puuza|sahau|acha)\b.{0,40}\b(?:maagizo|maelekezo|sheria)"),
    ("ru_ignore", r"(?:игнорируй|игнорируйте|забудь|забудьте|проигнорируй|не обращай внимания на)\w*.{0,40}(?:инструкци|правил|указани|промпт)"),
    ("uk_ignore", r"(?:ігноруй|забудь|не зважай на)\w*.{0,40}(?:інструкці|правил|вказівк|промпт)"),
    ("zh_ignore", r"(?:忽略|无视|無視|忘记|忘記|不要理会|不要遵守)\S{0,12}(?:指令|指示|规则|規則|提示词|提示詞|设定|設定)"),
    ("ja_ignore", r"(?:指示|命令|ルール|規則|プロンプト)\S{0,12}(?:無視|忘れ)"),
    ("ko_ignore", r"(?:지시|명령|규칙|프롬프트)\S{0,8}\s?(?:\S+\s)?(?:무시|잊어)"),
    ("ar_ignore", r"(?:تجاهل|انس|إنس|اترك|لا تتبع)\S*\s+(?:\S+\s+){0,3}?(?:التعليمات|تعليمات|القواعد|الأوامر|الاوامر)"),
    ("ur_ignore", r"(?:ہدایات|قواعد|اصول|احکامات)\S*\s+(?:\S+\s+){0,3}?(?:نظر انداز|بھول|مت مانو)"),
    ("fa_ignore", r"(?:دستورالعمل|دستورات|قوانین)\S*\s+(?:\S+\s+){0,3}?(?:نادیده|فراموش)"),
    ("world_reveal_prompt", r"(?:prompt del sistema|prompt système|prompt systeme|systemprompt|system-prompt|prompt do sistema|системн\w* (?:промпт|подсказк|инструкци)|系统提示|系統提示|システムプロンプト|시스템 프롬프트|موجه النظام|تعليمات النظام)"),
]
# Latin-script languages also need a "previous / all / your / system" word, so
# "can I ignore the parking rules on Sundays?" is not an attack.
_WORLD_QUALIFIERS = {
    "es_ignore": r"\b(?:anteriores|previas|tus|todas|sistema|originales|iniciales)\b",
    "pt_ignore": r"\b(?:anteriores|prévias|previas|suas|todas|sistema|originais|iniciais)\b",
    "fr_ignore": r"\b(?:précédentes|precedentes|antérieures|anterieures|tes|vos|toutes|système|systeme|initiales)\b",
    "de_ignore": r"\b(?:vorherigen|bisherigen|deine|ihre|alle|system|ursprünglichen)\b",
    "it_ignore": r"\b(?:precedenti|tue|sue|tutte|sistema|originali|iniziali)\b",
    "id_ignore": r"\b(?:sebelumnya|semua|kamu|anda|sistem)\b",
    "tr_ignore": r"\b(?:önceki|tüm|bütün|senin|sistem)\b",
    "sw_ignore": r"\b(?:yote|awali|zako|yako|mfumo)\b",
}
_COMPILED_WORLD = [(name, re.compile(p, flags=re.IGNORECASE),
                    re.compile(_WORLD_QUALIFIERS[name], flags=re.IGNORECASE) if name in _WORLD_QUALIFIERS else None)
                   for name, p in WORLD_PATTERNS]

# Long, specific phrases that are safe to match with all separators removed.
_SQUASH_SAFE = {
    "ignore_instructions", "reveal_prompt", "system_prompt", "developer_message",
}
_COMPILED = [(name, re.compile(p, flags=re.IGNORECASE | re.MULTILINE)) for name, p in INJECTION_PATTERNS]
# For the squashed view use separator-free versions of the safe patterns.
_COMPILED_SQUASHED = [(name, re.compile(p.replace(_SEP, "").replace(r"\b", ""), flags=re.IGNORECASE))
                      for name, p in INJECTION_PATTERNS if name in _SQUASH_SAFE]


# ---- Encoded payloads (base64 / hex / URL-encoding) ---------------------------
# Decoded up to _MAX_DECODE_DEPTH nested levels, bounded in size and run count,
# and kept only if the result is mostly printable text (or is itself another
# encoded run); the decoded text then goes through the same pattern scan.
_B64_RUN = re.compile(r"(?<![A-Za-z0-9+/_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_RUN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){8,}(?![0-9A-Fa-f])")
_URLENC = re.compile(r"%[0-9A-Fa-f]{2}")
_MAX_ENCODED_RUNS = 16
_MAX_DECODE_DEPTH = 3
_MAX_DECODE_CHARS = 4096


def _printable_text(raw: bytes) -> Optional[str]:
    try:
        s = raw.decode("utf-8")
    except UnicodeDecodeError:
        return None
    if not s:
        return None
    ok = sum(1 for ch in s if ch.isprintable() or ch.isspace())
    # Count combining marks too: Indic vowel signs (matras) are category M*, not
    # alphabetic, so Hindi/Tamil text would otherwise fall under the 50% bar.
    letters = sum(1 for ch in s if ch.isalpha() or unicodedata.category(ch).startswith("M"))
    # Mostly printable, with real words in it (random IDs decode to binary).
    if ok / len(s) < 0.9 or letters / len(s) < 0.5 or " " not in s.strip():
        return None
    return s


def _encoded_layer(raw: bytes) -> Optional[str]:
    """An intermediate layer: ASCII that is itself one encoded run (base64 / hex)
    or URL-encoded text. Such a layer is decoded again (up to _MAX_DECODE_DEPTH)."""
    try:
        s = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if len(s) < 16 or not s.isprintable():
        return None
    if _B64_RUN.fullmatch(s) or _HEX_RUN.fullmatch(s) or len(_URLENC.findall(s)) >= 2:
        return s
    return None


def _decode_level(text: str, budget: List[int]) -> List[str]:
    """Decode every encoded run in `text` once. `budget` is a shared one-item
    counter of runs still allowed (keeps nested decoding bounded)."""
    import base64
    import binascii
    from urllib.parse import unquote

    out: List[str] = []

    def keep(raw: bytes) -> None:
        dec = _printable_text(raw) or _encoded_layer(raw)
        if dec:
            out.append(dec)

    for m in _HEX_RUN.finditer(text):
        if budget[0] <= 0:
            return out
        budget[0] -= 1
        run = m.group(0)[: _MAX_DECODE_CHARS]
        keep(bytes.fromhex(run[: len(run) // 2 * 2]))
    for m in _B64_RUN.finditer(text):
        if budget[0] <= 0:
            return out
        run = m.group(0)
        # Needs mixed case or digits (plain words like "acetaminophen" are not base64).
        if run.isalpha() and (run.islower() or run.isupper()):
            continue
        # A pure hex run was already handled above.
        if _HEX_RUN.fullmatch(run):
            continue
        budget[0] -= 1
        run = run[: _MAX_DECODE_CHARS].rstrip("=")
        run = run.replace("-", "+").replace("_", "/")
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=True)
        except (binascii.Error, ValueError):
            continue
        keep(raw)
    if len(_URLENC.findall(text[:_MAX_DECODE_CHARS])) >= 2 and budget[0] > 0:
        budget[0] -= 1
        dec = unquote(text[: _MAX_DECODE_CHARS])
        if dec != text:
            out.append(dec)
    return out


def decode_candidates(text: str, max_depth: int = None) -> List[str]:
    """Plausible decoded payloads hidden in `text`.

    Nested encodings (base64 of base64, hex of base64, base64 of URL-encoding)
    are unwrapped up to `max_depth` levels (default _MAX_DECODE_DEPTH = 3), with
    at most _MAX_ENCODED_RUNS runs decoded in total. An intermediate layer is
    only followed if it is itself a single encoded run, so ordinary IDs (which
    decode to binary) stop at the first level.
    """
    depth = _MAX_DECODE_DEPTH if max_depth is None else max_depth
    budget = [_MAX_ENCODED_RUNS]
    out: List[str] = []
    level = [text]
    for _ in range(depth):
        nxt: List[str] = []
        for s in level:
            nxt.extend(_decode_level(s, budget))
        if not nxt:
            break
        out.extend(nxt)
        level = nxt
    return out


def detect_injection(text: str) -> Optional[str]:
    """Return the name of the first matching injection pattern, else None.

    Also scans base64 / hex / URL-encoded runs (up to 3 nested levels); a hit
    there is reported as "encoded_<pattern>".
    """
    hit = _detect_plain(text)
    if hit or not text:
        return hit
    for decoded in decode_candidates(text):
        hit = _detect_plain(decoded)
        if hit:
            return "encoded_" + hit
    return None


def _detect_plain(text: str) -> Optional[str]:
    if not text:
        return None
    norm, leet, squashed = _views(text)
    for name, rx in _COMPILED:
        if rx.search(norm) or rx.search(leet):
            return name
    if len(squashed) >= 12:
        for name, rx in _COMPILED_SQUASHED:
            if rx.search(squashed):
                return name
    raw = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH).casefold()
    for name, rx, qual in _COMPILED_WORLD:
        if rx.search(raw) and (qual is None or qual.search(raw)):
            return name
    return None


def scan_passages(contexts: Sequence) -> Tuple[list, List[Tuple[str, str]]]:
    """Split retrieved contexts into (clean, quarantined) by injection scan.

    A poisoned document in the index should never reach the generator (or a
    future LLM). Returns the clean contexts in their original order and a list
    of (chunk_id, pattern_name) for the quarantined ones.
    """
    clean, flagged = [], []
    for ctx in contexts:
        hit = detect_injection(getattr(ctx, "text", "") or "")
        if hit:
            flagged.append((ctx.chunk_id, hit))
        else:
            clean.append(ctx)
    return clean, flagged


# ---- Output-side check --------------------------------------------------------
from app.textutil import SENT_SPLIT_RE as _SENT_SPLIT  # noqa: E402


def _squash_ws(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s or "")).strip()


def output_guard(answer: str, contexts: Sequence, allowed_templates: Iterable[str] = ()) -> Tuple[bool, Optional[str]]:
    """The answer may only contain (a) sentences copied from the retrieved
    passages it was given or (b) one of the system's fixed templates.

    The extractive generator satisfies this by construction; the check exists so
    that swapping in an LLM generator cannot silently add unsupported text, a
    leaked prompt or an injected instruction to a medical answer.
    """
    ans = _squash_ws(answer)
    if not ans:
        return False, "Output guard: empty answer."
    templates = {_squash_ws(t) for t in allowed_templates}
    if ans in templates:
        return True, None
    hit = detect_injection(ans)
    if hit:
        return False, f"Output guard: answer matches injection pattern '{hit}'."
    passages = [_squash_ws(getattr(c, "text", "") or "") for c in contexts]
    for sent in _SENT_SPLIT.split(ans):
        sent = sent.strip()
        if not sent or sent in templates:
            continue
        if not any(sent in p for p in passages):
            return False, "Output guard: answer contains text not found in the retrieved passages."
    return True, None
