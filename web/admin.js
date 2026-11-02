/* Vouch console: collections, insights (gap report), embed snippet. */
(() => {
  "use strict";
  const $ = (s, el = document) => el.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const store = {
    get(k, d) { try { const v = localStorage.getItem("vouch." + k); return v === null ? d : v; } catch { return d; } },
    set(k, v) { try { localStorage.setItem("vouch." + k, v); } catch { /* private mode */ } },
  };
  const state = { config: null, collections: [], selected: null, token: store.get("token", "") };

  const theme = store.get("theme", "");
  if (theme) document.documentElement.dataset.theme = theme;
  $("#theme-btn").addEventListener("click", () => {
    const d = document.documentElement.dataset.theme ? document.documentElement.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    document.documentElement.dataset.theme = d ? "light" : "dark";
    store.set("theme", document.documentElement.dataset.theme);
  });

  function toast(msg) {
    const t = document.createElement("div"); t.className = "toast"; t.textContent = msg; t.setAttribute("role", "status");
    document.body.appendChild(t); setTimeout(() => t.remove(), 3200);
  }
  async function api(path, opts = {}) {
    opts.headers = { ...(opts.headers || {}) };
    if (state.token) opts.headers["X-Admin-Token"] = state.token;
    const res = await fetch(path, opts);
    let body = null; try { body = await res.json(); } catch { /* empty */ }
    if (res.status === 401) { $("#token-bar").hidden = false; }
    if (!res.ok) throw new Error((body && (body.detail || body.errors?.[0]?.error)) || `HTTP ${res.status}`);
    return body;
  }
  const json = (method, body) => ({ method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const langName = (c) => (state.config?.languages || {})[c] || c;

  // ---------- tabs ----------
  function showTab(name) {
    document.querySelectorAll(".tabs button").forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    document.querySelectorAll(".tab").forEach((t) => { t.hidden = t.id !== `tab-${name}`; });
    if (name === "insights") loadInsights();
    if (name === "embed") renderEmbed();
    history.replaceState(null, "", `#${name}`);
  }
  document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));
  $("#token-save").addEventListener("click", () => { state.token = $("#token").value.trim(); store.set("token", state.token); $("#token-bar").hidden = true; refresh(); });

  // ---------- collections ----------
  async function refresh(select) {
    const data = await api("/api/collections");
    state.collections = data.collections || [];
    renderGrid();
    const opts = state.collections.map((c) => `<option value="${esc(c.name)}">${esc(c.title)}</option>`).join("");
    $("#i-col").innerHTML = `<option value="">All collections</option>` + opts;
    const ec = $("#e-col"); const prev = ec.value; ec.innerHTML = opts;
    ec.value = prev || (state.collections.find((c) => !c.read_only) || state.collections[0] || {}).name || "";
    const want = select || state.selected;
    if (want && state.collections.some((c) => c.name === want)) openDetail(want);
  }

  function renderGrid() {
    $("#col-grid").innerHTML = state.collections.map((c) => {
      const langs = Object.keys(c.languages || {}).filter((l) => l !== "und");
      return `<button class="col-card card ${c.name === state.selected ? "active" : ""}" data-name="${esc(c.name)}">
        <span class="badge ${c.preset === "sample" || c.read_only ? "sample" : ""}">${c.read_only ? "built-in" : c.preset === "sample" ? "sample" : "yours"}</span>
        <h3>${esc(c.title)}</h3><p>${esc(c.description || "No description yet.")}</p>
        <span class="stats">${c.documents} docs · ${c.chunks} chunks ${langs.slice(0, 5).map((l) => `<span class="lang-pill">${esc(l)}</span>`).join("")}${langs.length > 5 ? ` +${langs.length - 5}` : ""}</span>
      </button>`;
    }).join("") || `<p class="empty">No collections yet. Create your first one.</p>`;
    document.querySelectorAll(".col-card").forEach((b) => b.addEventListener("click", () => openDetail(b.dataset.name)));
  }

  async function openDetail(name) {
    state.selected = name; renderGrid();
    const d = await api(`/api/collections/${encodeURIComponent(name)}`);
    $("#detail").hidden = false;
    $("#d-name").textContent = name; $("#d-title").textContent = d.title; $("#d-desc").textContent = d.description || "";
    $("#d-open").href = `/app?c=${encodeURIComponent(name)}`;
    $("#d-delete").hidden = d.read_only;
    $("#ingest").hidden = d.read_only;
    $("#d-docs").innerHTML = (d.document_list || []).map((x) => `<tr><td>${esc(x.title)}${x.page ? ` <span class="muted">p. ${x.page}</span>` : ""}${d.read_only ? "" : ` <button class="doc-del" type="button" data-id="${esc(x.doc_id)}" aria-label="Remove ${esc(x.title)}">remove</button>`}</td>
      <td><span class="lang-pill">${esc(x.language || "und")}</span> ${esc(langName(x.language))}</td><td>${x.chunks ?? ""}</td>
      <td>${/^https?:/.test(x.source || "") ? `<a href="${esc(x.source)}" target="_blank" rel="noopener">${esc(x.source)}</a>` : esc(x.source || "")}</td></tr>`).join("")
      || `<tr><td colspan="4" class="empty">No documents yet. Drop a file above.</td></tr>`;
    document.querySelectorAll(".doc-del").forEach((b) => b.addEventListener("click", async () => {
      try { await api(`/api/collections/${encodeURIComponent(name)}/documents/${encodeURIComponent(b.dataset.id)}`, { method: "DELETE" });
            toast("Document removed."); await refresh(); } catch (err) { toast(err.message); }
    }));
  }

  $("#new-btn").addEventListener("click", () => { const f = $("#new-form"); f.hidden = !f.hidden; if (!f.hidden) $("#nf-title").focus(); });
  $("#new-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const title = $("#nf-title").value.trim();
    const name = title.toLowerCase().normalize("NFKD").replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 48) || `collection-${Date.now() % 100000}`;
    try {
      await api("/api/collections", json("POST", { name, title, description: $("#nf-desc").value.trim() }));
      e.target.reset(); e.target.hidden = true; await refresh(name); toast(`Created “${title}”. Now add documents.`);
    } catch (err) { toast(err.message); }
  });

  const drop = $("#d-drop"), files = $("#d-files");
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("over"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
  drop.addEventListener("drop", (e) => upload(e.dataTransfer.files));
  files.addEventListener("change", () => upload(files.files));
  async function upload(list) {
    if (!list || !list.length || !state.selected) return;
    const fd = new FormData(); [...list].forEach((f) => fd.append("files", f));
    drop.querySelector("strong").textContent = `Ingesting ${list.length} file(s)…`;
    try {
      const r = await api(`/api/collections/${encodeURIComponent(state.selected)}/files`, { method: "POST", body: fd });
      toast(`Added ${r.added.length} document(s)${r.errors?.length ? `, ${r.errors.length} skipped` : ""}.`);
      await refresh();
    } catch (err) { toast(err.message); }
    finally { drop.querySelector("strong").textContent = "Drop files to add them"; files.value = ""; }
  }
  $("#d-url-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try { await api(`/api/collections/${encodeURIComponent(state.selected)}/url`, json("POST", { url: $("#d-url").value.trim() }));
          e.target.reset(); toast("Page added."); await refresh(); } catch (err) { toast(err.message); }
  });
  $("#d-text-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    try { await api(`/api/collections/${encodeURIComponent(state.selected)}/text`, json("POST", { title: $("#d-text-title").value.trim() || "Pasted text", text: $("#d-text").value }));
          e.target.reset(); toast("Text added."); await refresh(); } catch (err) { toast(err.message); }
  });
  $("#d-delete").addEventListener("click", async () => {
    if (!state.selected || !confirm(`Delete “${state.selected}” and all its documents?`)) return;
    try { await api(`/api/collections/${encodeURIComponent(state.selected)}`, { method: "DELETE" });
          state.selected = null; $("#detail").hidden = true; await refresh(); toast("Deleted."); } catch (err) { toast(err.message); }
  });

  // ---------- insights ----------
  $("#i-col").addEventListener("change", loadInsights);
  async function loadInsights() {
    let s;
    try { s = await api(`/api/insights?collection=${encodeURIComponent($("#i-col").value)}`); }
    catch (err) { $("#kpis").innerHTML = `<p class="empty">${esc(err.message)}</p>`; return; }
    if (!s.enabled) { $("#kpis").innerHTML = `<p class="empty">Insights are turned off on this server (ANALYTICS_ENABLED=false).</p>`; return; }
    $("#i-days").textContent = s.days;
    const pct = Math.round((s.abstention_rate || 0) * 100);
    $("#kpis").innerHTML = [
      ["Questions", s.total, ""], ["Answered with a quote", s.answered, ""],
      ["Not in the documents", `${pct}%`, pct > 30 ? "warn" : ""], ["Emergencies flagged", s.emergencies, s.emergencies ? "alert" : ""],
    ].map(([l, v, cls]) => `<div class="kpi card ${cls}"><div class="v">${esc(v)}</div><div class="l">${esc(l)}</div></div>`).join("");
    renderChart(s.daily || []);
    const max = Math.max(1, ...(s.languages || []).map((x) => x.count));
    $("#langs").innerHTML = (s.languages || []).map((x) => `<div class="lang-row"><span>${esc(langName(x.code))}</span><span class="track"><span class="fill" style="width:${(x.count / max) * 100}%"></span></span><span class="n">${x.count}</span></div>`).join("") || `<p class="empty">No questions yet.</p>`;
    const writable = new Set(state.collections.filter((c) => !c.read_only).map((c) => c.name));
    $("#gaps").innerHTML = (s.gaps || []).map((g, i) => `<li><span dir="auto">${esc(g.question)}</span><span class="count">×${g.count}${g.language ? " · " + esc(g.language) : ""}${!$("#i-col").value && g.collection ? " · " + esc(g.collection) : ""}</span>
      ${writable.has(g.collection) ? `<button class="fill-gap" type="button" data-i="${i}">Answer this gap →</button>` : ""}</li>`).join("") || `<li class="empty">No gaps. Every question was answered.</li>`;
    document.querySelectorAll(".fill-gap").forEach((b) => b.addEventListener("click", () => fillGap(s.gaps[+b.dataset.i])));
    $("#top").innerHTML = (s.top_questions || []).map((g) => `<li dir="auto">${esc(g.question)}<span class="count">×${g.count}</span></li>`).join("") || `<li class="empty">No questions yet.</li>`;
  }

  // Close the loop: a gap becomes a pasted answer in the right collection.
  async function fillGap(g) {
    showTab("collections");
    await openDetail(g.collection);
    const det = document.querySelector("#ingest details"); det.open = true;
    $("#d-text-title").value = g.question;
    $("#d-text").placeholder = `Write the answer to “${g.question}” as it should be quoted…`;
    $("#d-text").focus();
    det.scrollIntoView({ behavior: "smooth", block: "center" });
  }

  function renderChart(days) {
    const el = $("#chart");
    if (!days.length) { el.innerHTML = `<p class="empty">Ask a few questions in the desk and they show up here.</p>`; return; }
    const W = 520, H = 170, pad = 22, n = days.length, bw = Math.max(6, Math.min(40, (W - pad * 2) / n - 6));
    const max = Math.max(1, ...days.map((d) => d.total));
    const bars = days.map((d, i) => {
      const x = pad + i * ((W - pad * 2) / n) + 3, h = (d.total / max) * (H - 44), hg = (d.abstained / max) * (H - 44);
      return `<rect class="bar" x="${x}" y="${H - 22 - h}" width="${bw}" height="${h}" rx="3"><title>${esc(d.date)}: ${d.total} questions, ${d.abstained} not answered</title></rect>
              <rect class="bar-gap" x="${x}" y="${H - 22 - hg}" width="${bw}" height="${hg}" rx="3"/>
              ${n <= 14 || i % Math.ceil(n / 7) === 0 ? `<text x="${x + bw / 2}" y="${H - 6}" text-anchor="middle">${esc(d.date.slice(5))}</text>` : ""}`;
    }).join("");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Questions per day; ochre is the share not answered">${bars}</svg>`;
  }

  // ---------- embed ----------
  ["#e-col", "#e-label", "#e-color", "#e-pos"].forEach((s) => $(s).addEventListener("input", renderEmbed));
  function renderEmbed() {
    const col = $("#e-col").value || state.collections[0]?.name || "";
    const origin = location.origin;
    const snip = `<script src="${origin}/widget.js"\n        data-collection="${col}"\n        data-label="${$("#e-label").value.replace(/"/g, "")}"\n        data-color="${$("#e-color").value.replace(/"/g, "")}"\n        data-position="${$("#e-pos").value}"\n        defer></` + `script>`;
    $("#e-snippet").textContent = snip;
    $("#e-share").href = `${origin}/embed/${encodeURIComponent(col)}`;
    const f = $("#e-frame"); const src = `/embed/${encodeURIComponent(col)}`;
    if (col && f.getAttribute("src") !== src) f.setAttribute("src", src);
  }
  $("#e-copy").addEventListener("click", async () => {
    try { await navigator.clipboard.writeText($("#e-snippet").textContent); toast("Snippet copied."); } catch { toast("Select the snippet and copy it."); }
  });

  (async () => {
    try { state.config = await api("/api/config"); } catch { state.config = { languages: {} }; }
    if (state.config.admin_token_required && !state.token) $("#token-bar").hidden = false;
    try { await refresh(); } catch (err) { toast(err.message); }
    const tab = (location.hash || "#collections").slice(1);
    showTab(["collections", "insights", "embed"].includes(tab) ? tab : "collections");
    if (!state.selected && state.collections.length && tab === "collections") openDetail((state.collections.find((c) => !c.read_only) || state.collections[0]).name);
  })();
})();
