/* Vouch ask app (/app) and widget frame (/embed/<collection>). No build step. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const RTL = new Set(["ar", "ur", "fa", "he", "ps", "sd", "ug", "yi", "dv", "ckb"]);
  const EMBED = document.body.classList.contains("embed-page");
  const base = (code) => (code || "und").split("-")[0].toLowerCase();
  const isRtl = (code) => RTL.has(base(code));
  const store = {
    get(k, d) { try { const v = localStorage.getItem("vouch." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("vouch." + k, v); } catch { /* private mode */ } },
  };
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  const state = { config: null, collections: [], current: null, lang: store.get("lang", "auto"),
                  stt: store.get("stt", ""), tts: store.get("tts", "0") === "1", recording: null };

  // ---------- theme ----------
  const theme = store.get("theme", "");
  if (theme) document.documentElement.dataset.theme = theme;
  $("#theme-btn")?.addEventListener("click", () => {
    const dark = document.documentElement.dataset.theme
      ? document.documentElement.dataset.theme === "dark"
      : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = dark ? "light" : "dark";
    store.set("theme", document.documentElement.dataset.theme);
  });

  function toast(msg, ms = 3200) {
    const t = document.createElement("div");
    t.className = "toast"; t.textContent = msg; t.setAttribute("role", "status");
    document.body.appendChild(t); setTimeout(() => t.remove(), ms);
  }

  async function api(path, opts = {}) {
    const res = await fetch(path, opts);
    let body = null;
    try { body = await res.json(); } catch { /* empty */ }
    if (!res.ok) throw new Error((body && (body.detail || (body.errors && body.errors[0]?.error))) || `HTTP ${res.status}`);
    return body;
  }

  const langName = (code) => (state.config?.languages || {})[code] || (state.config?.languages || {})[base(code)] || code || "";

  // ---------- collections ----------
  function renderCollections() {
    const nav = $("#collections");
    if (!nav) return;
    nav.innerHTML = state.collections.map((c) => {
      const langs = Object.keys(c.languages || {}).filter((l) => l !== "und").slice(0, 6);
      return `<button class="col-item ${c.name === state.current ? "active" : ""}" data-name="${esc(c.name)}">
        <span class="t">${esc(c.title)}</span>
        <span class="s">${c.documents} doc${c.documents === 1 ? "" : "s"}${langs.length ? " · " + langs.map((l) => `<span class="lang-pill">${esc(l)}</span>`).join(" ") : ""}</span>
      </button>`;
    }).join("") || `<p class="muted">No collections yet. Create one below.</p>`;
    nav.querySelectorAll(".col-item").forEach((b) => b.addEventListener("click", () => { selectCollection(b.dataset.name); closeSide(); }));
  }

  function selectCollection(name) {
    const c = state.collections.find((x) => x.name === name);
    state.current = c ? c.name : null;
    if (!EMBED) {
      store.set("collection", state.current || "");
      const url = new URL(location.href); url.searchParams.set("c", state.current || ""); history.replaceState(null, "", url);
    }
    $("#col-title").textContent = c ? c.title : "No collection";
    const langs = c ? Object.keys(c.languages || {}).filter((l) => l !== "und").map(langName) : [];
    $("#col-meta").textContent = c ? `${c.documents} document${c.documents === 1 ? "" : "s"}${langs.length ? " in " + langs.slice(0, 4).join(", ") + (langs.length > 4 ? ` +${langs.length - 4}` : "") : ""}` : "";
    renderCollections();
    renderSuggestions();
  }

  function renderSuggestions() {
    const box = $("#suggestions");
    if (!box) return;
    const ex = (state.config?.examples || {})[state.current] || [];
    box.innerHTML = ex.slice(0, 9).map((q) => `<button class="chip" type="button" dir="auto" data-q="${esc(q)}">${esc(q)}</button>`).join("");
    box.querySelectorAll(".chip").forEach((b) => b.addEventListener("click", () => ask(b.dataset.q)));
  }

  // ---------- rendering ----------
  const ICON_ALERT = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 3l10 18H2L12 3z"/><path d="M12 10v5M12 18h0"/></svg>`;
  const ICON_SPEAK = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M4 9v6h4l5 4V5L8 9H4z"/><path d="M16 9a4 4 0 0 1 0 6M18.5 6.5a8 8 0 0 1 0 11"/></svg>`;
  const STAMP = `<svg class="stamp" viewBox="0 0 64 64" aria-hidden="true"><circle cx="32" cy="32" r="29" fill="none" stroke="var(--verified)" stroke-width="2.5"/><circle cx="32" cy="32" r="23" fill="none" stroke="var(--verified)" stroke-width="1" stroke-dasharray="2 2.5"/><path d="M21 33l7 7 15-16" fill="none" stroke="var(--verified)" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/></svg>`;

  function highlight(passage, quote) {
    const p = esc(passage || ""), q = esc((quote || "").trim());
    if (!q) return p;
    const i = p.indexOf(q);
    if (i >= 0) return p.slice(0, i) + "<mark>" + q + "</mark>" + p.slice(i + q.length);
    const head = q.slice(0, 40), j = head ? p.indexOf(head) : -1;
    return j >= 0 ? p.slice(0, j) + "<mark>" + p.slice(j, j + q.length) + "</mark>" + p.slice(j + q.length) : p;
  }

  function highlightAll(passage, quotes) {
    // Mark each quoted sentence; quotes are sentences copied from the passage.
    let out = esc(passage || "");
    for (const q of quotes) {
      const e = esc((q || "").trim());
      if (e && out.includes(e)) out = out.replace(e, "<mark>" + e + "</mark>");
    }
    return out.includes("<mark>") ? out : highlight(passage, quotes[0]);
  }

  function addUser(text, lang) {
    const el = document.createElement("div");
    el.className = "msg user"; el.setAttribute("dir", "auto");
    el.innerHTML = `<div class="q"></div><div class="meta"></div>`;
    el.querySelector(".q").textContent = text;
    if (lang) el.lang = lang;
    $("#thread").appendChild(el);
    $("#welcome")?.remove();
    return el;
  }

  function addTyping() {
    const el = document.createElement("div");
    el.className = "msg card"; el.innerHTML = `<span class="typing" aria-label="Looking through the documents"><i></i><i></i><i></i></span>`;
    $("#thread").appendChild(el); scrollDown(); return el;
  }

  function scrollDown() { const t = $("#thread"); t.scrollTop = t.scrollHeight; }

  // Card status labels in the asker's language (English fallback).
  const LABELS = {
    en: ["Vouched", "Not in your documents", "Blocked by safety check"],
    hi: ["प्रमाणित", "आपके दस्तावेज़ों में नहीं", "सुरक्षा जाँच ने रोका"],
    bn: ["প্রমাণিত", "আপনার নথিতে নেই", "নিরাপত্তা যাচাইয়ে আটকানো"],
    ta: ["ஆதாரம் உள்ளது", "உங்கள் ஆவணங்களில் இல்லை", "பாதுகாப்புச் சோதனையால் தடுக்கப்பட்டது"],
    es: ["Respaldado", "No está en tus documentos", "Bloqueado por seguridad"],
    fr: ["Attesté", "Absent de vos documents", "Bloqué par la sécurité"],
    de: ["Belegt", "Nicht in Ihren Dokumenten", "Aus Sicherheitsgründen blockiert"],
    pt: ["Comprovado", "Não está nos seus documentos", "Bloqueado por segurança"],
    it: ["Verificato", "Non è nei tuoi documenti", "Bloccato per sicurezza"],
    ar: ["موثّق", "غير موجود في مستنداتك", "محظور لأسباب أمنية"],
    ur: ["مصدقہ", "آپ کی دستاویزات میں نہیں", "حفاظتی جانچ نے روک دیا"],
    fa: ["مستند", "در اسناد شما نیست", "به دلایل امنیتی مسدود شد"],
    zh: ["有据可查", "文档中没有", "已被安全检查拦截"],
    ja: ["出典あり", "文書にありません", "安全チェックでブロック"],
    ko: ["출처 확인", "문서에 없음", "보안 검사로 차단됨"],
    ru: ["Подтверждено", "Нет в ваших документах", "Заблокировано проверкой безопасности"],
    uk: ["Підтверджено", "Немає у ваших документах", "Заблоковано перевіркою безпеки"],
    sw: ["Imethibitishwa", "Haipo kwenye nyaraka zako", "Imezuiwa na ukaguzi wa usalama"],
    id: ["Terverifikasi", "Tidak ada di dokumen Anda", "Diblokir pemeriksaan keamanan"],
    tr: ["Doğrulandı", "Belgelerinizde yok", "Güvenlik kontrolü engelledi"],
  };
  const labelsFor = (code) => (code === "hi-Latn" ? LABELS.en : LABELS[code] || LABELS[base(code)] || LABELS.en);

  function statusOf(r) {
    if (!r.abstained) return "ok";
    return /blocked|unsafe/i.test(r.abstain_reason || "") ? "blocked" : "gap";
  }

  function renderAnswer(r) {
    const kind = statusOf(r);
    const el = document.createElement("article");
    el.className = `msg card answer ${kind === "ok" ? "verified" : kind}`;
    const aLang = r.answer_language || r.language;
    // One chip per source passage; several quoted sentences from the same
    // passage are all highlighted when it opens.
    const cites = [];
    for (const c of r.citations || []) {
      const prev = cites.find((x) => x.chunk_id === c.chunk_id);
      if (prev) prev.quotes.push(c.quote); else cites.push({ ...c, quotes: [c.quote] });
    }
    const L = labelsFor(r.language);
    const label = { ok: `${L[0]} · ${cites.length}`, gap: L[1], blocked: L[2] }[kind];
    const sameLang = base(aLang) === base(r.language);
    el.innerHTML = `
      ${r.advisory ? `<div class="emergency" role="alert" dir="${isRtl(r.language) ? "rtl" : "ltr"}" lang="${esc(r.language || "")}">${ICON_ALERT}<span>${esc(r.advisory)}</span></div>` : ""}
      ${kind === "ok" ? STAMP : ""}
      <div class="answer-head"><span class="status ${kind}" lang="${esc(L === LABELS.en ? "en" : r.language || "en")}" title="${kind === "ok" ? `${cites.length} source${cites.length === 1 ? "" : "s"}` : ""}">${esc(label)}</span>
        ${r.language ? `<span class="lang-pill" title="Language of your question">${esc(langName(r.language))}</span>` : ""}</div>
      <p class="answer-text" dir="${isRtl(aLang) ? "rtl" : "ltr"}" lang="${esc(aLang || "")}"></p>
      ${cites.length ? `<div class="cites">${cites.map((c, i) => `<button class="cite-chip" type="button" aria-expanded="false" data-i="${i}"><b>[${i + 1}]</b><span>${esc(c.title || c.parent_doc_id || "source")}${c.page ? ` · p. ${c.page}` : ""}</span></button>`).join("")}</div><div class="passages"></div>` : ""}
      <div class="answer-foot">
        <span>${kind === "ok" ? (r.translated ? `Translated from a ${esc(langName(cites[0]?.language))} source` : sameLang ? "Quoted word for word from the source" : `Quoted from a ${esc(langName(aLang))} source`) : esc(kind === "gap" ? "Nothing was guessed." : "")}</span>
        <span>${r.timings_ms?.total_ms != null ? `<span class="mono">${Math.round(r.timings_ms.total_ms)} ms</span>` : ""}
        ${"speechSynthesis" in window ? `<button class="speak-btn" type="button" aria-label="Read aloud">${ICON_SPEAK}</button>` : ""}</span>
      </div>`;
    el.querySelector(".answer-text").textContent = r.answer;
    // Right-to-left card when the quoted answer (or, for abstentions, the
    // question) is in Arabic, Urdu, Persian or Hebrew.
    if (kind === "ok" ? isRtl(aLang) : r.rtl) el.setAttribute("dir", "rtl");
    el.querySelectorAll(".cite-chip").forEach((b) => b.addEventListener("click", () => togglePassage(el, b, cites[+b.dataset.i])));
    el.querySelector(".speak-btn")?.addEventListener("click", () => speak(r.answer, aLang));
    $("#thread").appendChild(el);
    if (state.tts) speak(r.advisory ? r.advisory + " " + r.answer : r.answer, aLang);
    scrollDown();
    return el;
  }

  function togglePassage(card, chip, c) {
    const box = card.querySelector(".passages");
    const open = chip.getAttribute("aria-expanded") === "true";
    card.querySelectorAll(".cite-chip").forEach((x) => x.setAttribute("aria-expanded", "false"));
    box.innerHTML = "";
    if (open) return;
    chip.setAttribute("aria-expanded", "true");
    const dir = isRtl(c.language) ? "rtl" : "ltr";
    const src = c.source && /^https?:\/\//.test(c.source) ? `<a href="${esc(c.source)}" target="_blank" rel="noopener">${esc(c.source)}</a>` : esc(c.source || "");
    box.innerHTML = `<div class="passage" dir="${dir}" lang="${esc(c.language || "")}">${highlightAll(c.passage || c.quote, c.quotes || [c.quote])}
      <div class="src" dir="ltr"><span>${esc(c.title || "")}${c.page ? ` · page ${c.page}` : ""}${src ? " · " + src : ""}</span><span class="mono">${esc((c.chunk_id || "").slice(0, 8))}</span></div></div>`;
  }

  function updateLanguage(r) {
    const ind = $("#lang-indicator"); if (!ind || !r.language) return;
    $("#lang-name").textContent = r.language_name || langName(r.language);
    ind.dir = r.rtl ? "rtl" : "ltr";
    ind.classList.remove("flash"); void ind.offsetWidth; ind.classList.add("flash");
  }

  // ---------- TTS ----------
  function speak(text, lang) {
    if (!("speechSynthesis" in window) || !text) return;
    speechSynthesis.cancel();
    const u = new SpeechSynthesisUtterance(text);
    if (lang && lang !== "und") {
      u.lang = base(lang) === "hi" && lang.endsWith("Latn") ? "en-IN" : lang;
      const v = speechSynthesis.getVoices().find((x) => x.lang && x.lang.toLowerCase().startsWith(base(lang)));
      if (v) u.voice = v;
    }
    speechSynthesis.speak(u);
  }

  // ---------- asking ----------
  async function ask(text, opts = {}) {
    text = (text || "").trim();
    if (!text && !opts.audio) return;
    if (!state.current && state.collections.length) selectCollection(state.collections[0].name);
    const userEl = opts.audio ? addUser("Voice question…") : addUser(text);
    const typing = addTyping();
    try {
      let r;
      if (opts.audio) {
        const fd = new FormData();
        fd.append("file", opts.audio, "question.webm");
        if (state.current) fd.append("collection", state.current);
        if (state.lang !== "auto") fd.append("language", state.lang);
        if (state.stt && state.stt !== "browser" && (state.config.stt_providers || []).includes(state.stt)) fd.append("stt", state.stt);
        r = await api("/api/ask-audio", { method: "POST", body: fd });
        userEl.querySelector(".q").textContent = r.transcript || "(no speech recognised)";
      } else {
        r = await api("/api/ask-text", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: text, collection: state.current, language: opts.lang || (state.lang === "auto" ? null : state.lang),
                                 channel: EMBED ? "widget" : (opts.voice ? "voice" : "text") }) });
      }
      typing.remove();
      if (r.language) { userEl.lang = r.language; userEl.setAttribute("dir", r.rtl ? "rtl" : "ltr"); }
      userEl.querySelector(".meta").innerHTML = r.language ? `<span>${esc(r.language_name || r.language)}</span>${r.stt_provider ? `<span>· ${esc(r.stt_provider)}</span>` : ""}` : "";
      updateLanguage(r);
      renderAnswer(r);
    } catch (e) {
      typing.remove();
      toast(e.message || "Something went wrong");
    }
  }

  $("#composer").addEventListener("submit", (e) => {
    e.preventDefault();
    const q = $("#query"); const text = q.value; q.value = ""; autosize(); ask(text);
  });
  const qEl = $("#query");
  function autosize() { qEl.style.height = "auto"; qEl.style.height = Math.min(qEl.scrollHeight, 160) + "px"; }
  qEl.addEventListener("input", autosize);
  qEl.addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });

  // ---------- voice ----------
  const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  const mic = $("#mic");
  let timerId = null;

  function setRecording(on, text) {
    mic.classList.toggle("recording", on);
    mic.setAttribute("aria-pressed", String(on));
    mic.setAttribute("aria-label", on ? "Stop recording" : "Ask by voice");
    $("#rec-status").hidden = !on;
    if (text) $("#rec-text").textContent = text;
    clearInterval(timerId);
    if (on) {
      const t0 = Date.now();
      const tick = () => { const s = Math.floor((Date.now() - t0) / 1000); $("#rec-timer").textContent = `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`; };
      tick(); timerId = setInterval(tick, 500);
    }
  }

  function browserRecognition() {
    if (!SR) { toast("This browser has no built-in speech recognition. Try Chrome or Edge, or ask your admin to enable Whisper."); return; }
    const rec = new SR();
    rec.lang = state.lang !== "auto" ? state.lang : (navigator.language || "en-US");
    rec.interimResults = true; rec.continuous = false;
    let finalText = "";
    rec.onresult = (ev) => {
      let interim = "";
      for (const res of ev.results) { if (res.isFinal) finalText = res[0].transcript; else interim += res[0].transcript; }
      qEl.value = finalText || interim; autosize();
    };
    rec.onerror = (ev) => { if (ev.error !== "no-speech" && ev.error !== "aborted") toast(`Microphone: ${ev.error}`); };
    rec.onend = () => { setRecording(false); state.recording = null; const t = (finalText || qEl.value).trim(); if (t) { qEl.value = ""; autosize(); ask(t, { voice: true }); } };
    state.recording = { stop: () => rec.stop() };
    setRecording(true, "Listening… speak in any language");
    rec.start();
  }

  async function serverRecording() {
    if (!navigator.mediaDevices?.getUserMedia) { toast("Microphone not available here."); return; }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mr = new MediaRecorder(stream);
      const chunks = [];
      mr.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      mr.onstop = () => { stream.getTracks().forEach((t) => t.stop()); setRecording(false); state.recording = null;
                          ask("", { audio: new Blob(chunks, { type: mr.mimeType || "audio/webm" }) }); };
      state.recording = { stop: () => mr.state !== "inactive" && mr.stop() };
      mr.start();
      setRecording(true, "Recording… tap to stop");
      setTimeout(() => state.recording?.stop(), 30000);
    } catch (e) { toast("Microphone permission was denied."); }
  }

  mic.addEventListener("click", () => {
    if (state.recording) { state.recording.stop(); return; }
    (state.stt === "browser" ? browserRecognition : serverRecording)();
  });

  // ---------- settings ----------
  const STT_LABELS = { browser: "In the browser (no key)", sarvam: "Sarvam (Indian languages)", whisper_api: "Whisper API (any language)",
                       faster_whisper: "Local Whisper (any language)" };
  function initSettings() {
    const btn = $("#settings-btn"), panel = $("#settings");
    btn?.addEventListener("click", () => { panel.hidden = !panel.hidden; btn.setAttribute("aria-expanded", String(!panel.hidden)); });
    document.addEventListener("click", (e) => { if (panel && !panel.hidden && !panel.contains(e.target) && !btn.contains(e.target)) { panel.hidden = true; btn.setAttribute("aria-expanded", "false"); } });
    const ls = $("#lang-select");
    const langs = Object.entries(state.config.languages || {}).filter(([c]) => c !== "und" && c !== "hi-Latn").sort((a, b) => a[1].localeCompare(b[1]));
    ls.innerHTML = `<option value="auto">Detect automatically</option>` + langs.map(([c, n]) => `<option value="${esc(c)}">${esc(n)}</option>`).join("");
    ls.value = state.lang; ls.addEventListener("change", () => { state.lang = ls.value; store.set("lang", ls.value); });
    const ss = $("#stt-select");
    const provs = state.config.stt_providers || ["browser"];
    ss.innerHTML = provs.map((p) => `<option value="${esc(p)}">${esc(STT_LABELS[p] || p)}</option>`).join("");
    const preferred = provs.includes(state.stt) ? state.stt : (state.config.stt_default !== "browser" && provs.includes(state.config.stt_default) ? state.config.stt_default : "browser");
    // Offline demo: the server's scripted recogniser is more reliable than nothing when the browser has none.
    state.stt = !SR && preferred === "browser" && state.config.demo_mode ? "server-demo" : preferred;
    ss.value = provs.includes(state.stt) ? state.stt : "browser";
    ss.addEventListener("change", () => { state.stt = ss.value; store.set("stt", ss.value); });
    const tt = $("#tts-toggle"); tt.checked = state.tts;
    tt.addEventListener("change", () => { state.tts = tt.checked; store.set("tts", tt.checked ? "1" : "0"); });
  }

  // ---------- new collection ----------
  function initNewCollection() {
    const form = $("#new-col-form"); if (!form) return;
    const drop = $("#drop"), files = $("#new-col-files");
    ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
    ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
    drop.addEventListener("drop", (e) => { files.files = e.dataTransfer.files; showFiles(); });
    files.addEventListener("change", showFiles);
    function showFiles() { drop.querySelector("span").innerHTML = files.files.length ? [...files.files].map((f) => esc(f.name)).join("<br>") : "Drop PDF, TXT, MD or HTML"; }
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const title = $("#new-col-title").value.trim();
      const name = title.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || `collection-${Date.now() % 100000}`;
      const btn = form.querySelector("button[type=submit]"); btn.disabled = true; btn.textContent = "Ingesting…";
      try {
        await api("/api/collections", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name, title }) });
        if (files.files.length) {
          const fd = new FormData(); [...files.files].forEach((f) => fd.append("files", f));
          const r = await api(`/api/collections/${name}/files`, { method: "POST", body: fd });
          if (r.errors?.length) toast(`${r.errors.length} file(s) skipped: ${r.errors[0].error}`);
        }
        const url = $("#new-col-url").value.trim();
        if (url) await api(`/api/collections/${name}/url`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ url }) });
        await loadCollections(name);
        form.reset(); showFiles(); $("#new-col").open = false;
        toast(`“${title}” is ready. Ask away.`);
      } catch (err) { toast(err.message); }
      finally { btn.disabled = false; btn.textContent = "Create & ingest"; }
    });
  }

  // ---------- mobile sidebar ----------
  function closeSide() { $("#sidebar")?.classList.remove("open"); }
  $("#open-side")?.addEventListener("click", () => $("#sidebar").classList.add("open"));
  $("#close-side")?.addEventListener("click", closeSide);

  async function loadCollections(select) {
    const data = await api("/api/collections");
    state.collections = data.collections || [];
    const params = new URLSearchParams(location.search);
    const fromPath = EMBED ? decodeURIComponent(location.pathname.split("/").filter(Boolean)[1] || "") : null;
    const want = select || fromPath || params.get("c") || store.get("collection", "");
    const first = state.collections.find((c) => !c.read_only && c.preset === "sample") || state.collections[0];
    selectCollection(state.collections.some((c) => c.name === want) ? want : first?.name);
  }

  async function init() {
    try {
      state.config = await api("/api/config");
    } catch { state.config = { languages: {}, stt_providers: ["browser"], examples: {} }; }
    initSettings();
    initNewCollection();
    try { await loadCollections(); }
    catch { $("#col-title").textContent = "Default index"; state.collections = []; }
    // Deep links: /app?c=<collection>&q=first&q=second&cite=1 replays a short
    // conversation (shareable demos); cite=1 opens the last answer's first source.
    const params = new URLSearchParams(location.search);
    for (const q of params.getAll("q")) { if (q.trim()) await ask(q); }
    if (params.get("cite") === "1") {
      const withSources = [...document.querySelectorAll(".answer")].filter((a) => a.querySelector(".cite-chip"));
      withSources[withSources.length - 1]?.querySelector(".cite-chip")?.click();
    }
  }
  init();
})();
