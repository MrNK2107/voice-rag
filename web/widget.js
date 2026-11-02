/* Vouch embeddable widget.
 *
 *   <script src="https://YOUR-VOUCH-HOST/widget.js"
 *           data-collection="city-desk"
 *           data-label="Ask us anything"
 *           data-color="#c8432b"
 *           data-position="right" defer></script>
 *
 * Adds a floating "ask" button that opens /embed/<collection> in an iframe.
 * The host page shares nothing with the frame; answers are served by Vouch.
 */
(function () {
  "use strict";
  var script = document.currentScript || document.querySelector('script[src*="widget.js"][data-collection]');
  if (!script || window.__vouchWidget) return;
  window.__vouchWidget = true;
  var origin = new URL(script.src, location.href).origin;
  var collection = script.getAttribute("data-collection") || "";
  var label = script.getAttribute("data-label") || "Ask a question";
  var color = script.getAttribute("data-color") || "#c8432b";
  var side = script.getAttribute("data-position") === "left" ? "left" : "right";
  if (!/^#[0-9a-fA-F]{3,8}$/.test(color)) color = "#c8432b";

  var css = "" +
    ".vouch-btn{position:fixed;bottom:20px;" + side + ":20px;z-index:2147483000;display:flex;align-items:center;gap:10px;" +
    "padding:12px 18px 12px 12px;border:none;border-radius:999px;background:" + color + ";color:#fff;cursor:pointer;" +
    "font:600 15px/1.2 system-ui,-apple-system,'Segoe UI',sans-serif;box-shadow:0 10px 30px -8px rgba(0,0,0,.45);transition:transform .15s}" +
    ".vouch-btn:hover{transform:translateY(-2px)}" +
    ".vouch-btn svg{width:30px;height:30px;flex:none}" +
    ".vouch-frame{position:fixed;bottom:86px;" + side + ":20px;z-index:2147483000;width:min(400px,calc(100vw - 24px));" +
    "height:min(620px,calc(100vh - 110px));border:1px solid rgba(0,0,0,.12);border-radius:18px;background:#f5efe3;" +
    "box-shadow:0 24px 60px -18px rgba(0,0,0,.5);display:none}" +
    ".vouch-frame.open{display:block;animation:vouchIn .2s ease}" +
    "@keyframes vouchIn{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:none}}" +
    "@media (max-width:520px){.vouch-frame{" + side + ":12px;bottom:80px;width:calc(100vw - 24px);height:calc(100vh - 100px)}}";
  var style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  var btn = document.createElement("button");
  btn.className = "vouch-btn";
  btn.type = "button";
  btn.setAttribute("aria-expanded", "false");
  btn.innerHTML = '<svg viewBox="0 0 32 32" aria-hidden="true"><circle cx="16" cy="16" r="15" fill="rgba(255,255,255,.18)"/>' +
    '<rect x="12.5" y="7" width="7" height="12" rx="3.5" fill="#fff"/><path d="M9 15a7 7 0 0 0 14 0M16 22v3" stroke="#fff" stroke-width="2" fill="none" stroke-linecap="round"/></svg>';
  var span = document.createElement("span");
  span.textContent = label;
  btn.appendChild(span);

  var frame = null;
  btn.addEventListener("click", function () {
    if (!frame) {
      frame = document.createElement("iframe");
      frame.className = "vouch-frame";
      frame.title = label;
      frame.allow = "microphone; autoplay";
      frame.src = origin + "/embed/" + encodeURIComponent(collection);
      document.body.appendChild(frame);
    }
    var open = !frame.classList.contains("open");
    frame.classList.toggle("open", open);
    btn.setAttribute("aria-expanded", String(open));
  });
  (document.body ? Promise.resolve() : new Promise(function (r) { document.addEventListener("DOMContentLoaded", r); }))
    .then(function () { document.body.appendChild(btn); });
})();
