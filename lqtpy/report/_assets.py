"""Stylesheet, script and glossary shared by the standalone reports.

Plain strings (no f-string brace escaping). The palette is checked for
colour-vision deficiencies by ``tests/test_accessibility.py``.
"""
from __future__ import annotations

CSS = r"""
:root {
  color-scheme: dark;
  --bg:#0d1015; --panel:#151a21; --panel-2:#1b212a; --line:rgba(255,255,255,.08);
  --line-2:rgba(255,255,255,.16); --ink:#e9edf2; --ink-2:#b3bcc8; --ink-3:#8f98a6;
  --lesion:#f07a45; --lesion-track:rgba(240,122,69,.14); --comp:#8f9aab; --comp-track:rgba(255,255,255,.06);
  --left:#3d70ee; --right:#22c0a8; --mid:#6b7480;
  --extra:#9d93ee; --extra-soft:rgba(144,133,233,.10);
  --ok:#35c04f; --warn:#fab219; --err:#ff6b63; --info:#8f98a6;
  --ok-bg:rgba(12,163,12,.14); --warn-bg:rgba(250,178,25,.13); --err-bg:rgba(208,59,59,.18); --info-bg:rgba(255,255,255,.05);
  --focus:#7ab3ff; --hl:#7ab3ff; --shadow:0 1px 0 rgba(255,255,255,.03), 0 8px 24px rgba(0,0,0,.25);
  --plate:#000; --fm-na:#3a414c; --fm-edge:#0b0d11;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  --mono:ui-monospace,"SF Mono",Consolas,"Liberation Mono",monospace;
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg:#f4f5f7; --panel:#ffffff; --panel-2:#f1f3f6; --line:#e3e6eb; --line-2:#cfd4dc;
  --ink:#121820; --ink-2:#4a5361; --ink-3:#5f6877;
  --lesion:#d9582a; --lesion-track:rgba(224,96,42,.12); --comp:#8a94a3; --comp-track:#edf0f3;
  --left:#2a5bd6; --right:#12a594; --mid:#9aa3ae;
  --extra:#4a3aa7; --extra-soft:rgba(74,58,167,.06);
  --ok:#0b7a0b; --warn:#8a5d00; --err:#c0342f; --info:#5f6877;
  --ok-bg:#e8f5e8; --warn-bg:#fff4d6; --err-bg:#fde8e6; --info-bg:#f1f3f6;
  --focus:#1f6fd1; --hl:#1f6fd1; --shadow:0 1px 2px rgba(16,24,40,.05);
  --plate:#fff; --fm-na:#d9dce1; --fm-edge:#2b2f36;
}
* { box-sizing:border-box; }
html, body { margin:0; }
body { background:var(--bg); color:var(--ink); font:14px/1.55 var(--sans); -webkit-font-smoothing:antialiased; }
a { color:var(--focus); }
:focus-visible { outline:2px solid var(--focus); outline-offset:2px; border-radius:4px; }
.num, td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.muted { color:var(--ink-3); }
.skip { position:absolute; left:-9999px; top:8px; z-index:50; background:var(--panel); color:var(--ink);
  padding:8px 12px; border-radius:8px; border:1px solid var(--line-2); }
.skip:focus { left:12px; }
.sr { position:absolute; width:1px; height:1px; overflow:hidden; clip:rect(0 0 0 0); white-space:nowrap; }

/* ---------- toolbar ---------- */
.bar { position:sticky; top:0; z-index:10; display:flex; align-items:center; gap:12px; flex-wrap:wrap;
  padding:10px 20px; background:color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter:blur(8px); border-bottom:1px solid var(--line); }
.brand { font-weight:700; letter-spacing:.02em; }
.brand small { font-weight:500; color:var(--ink-3); margin-left:8px; letter-spacing:0; }
.spacer { flex:1; }
.seg { display:inline-flex; border:1px solid var(--line-2); border-radius:8px; overflow:hidden; }
.seg button, .btn { font:inherit; font-size:13px; color:var(--ink-2); background:transparent; border:0;
  padding:6px 12px; cursor:pointer; }
.seg button[aria-pressed="true"] { background:var(--panel-2); color:var(--ink); font-weight:600; }
.btn { border:1px solid var(--line-2); border-radius:8px; }
.btn.small { font-size:12px; padding:3px 9px; }
.btn:hover, .seg button:hover { color:var(--ink); }
.btn.primary { background:var(--ink); color:var(--bg); border-color:var(--ink); font-weight:600; }
html[data-view="clinical"] .theme-only { display:none; }

/* ---------- views ---------- */
.view { display:none; }
html[data-view="dashboard"] .view-dash { display:block; }
html[data-view="clinical"] .view-clin { display:block; }

/* ---------- dashboard ---------- */
.dash { max-width:1180px; margin:0 auto; padding:24px 20px 72px; display:grid; gap:16px; }
.hero { display:flex; flex-wrap:wrap; align-items:flex-end; gap:8px 16px; }
.hero h1 { margin:0; font-size:24px; letter-spacing:-.01em; text-wrap:balance; }
.hero .meta { color:var(--ink-3); font-size:12.5px; }
.back { font-size:13px; }
.pill { display:inline-flex; align-items:center; gap:6px; padding:2px 10px; border-radius:999px;
  font-size:12px; font-weight:600; border:1px solid var(--line-2); color:var(--ink-2); }
.pill.extra { color:var(--extra); border-color:var(--extra); background:var(--extra-soft); }
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:12px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; min-width:0; }
.kpi .k { font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3); }
.kpi .v { font-size:22px; font-weight:650; margin-top:2px; line-height:1.2; white-space:nowrap;
  overflow:hidden; text-overflow:ellipsis; }
.kpi .v.text { font-size:16px; white-space:normal; display:-webkit-box; -webkit-line-clamp:2;
  -webkit-box-orient:vertical; overflow-wrap:anywhere; line-height:1.25; }
.kpi .s { font-size:12.5px; color:var(--ink-2); margin-top:2px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px;
  box-shadow:var(--shadow); min-width:0; }
.card > h2 { margin:0 0 12px; font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); font-weight:650;
  display:flex; align-items:center; gap:8px; }
.card > h2 .tools { margin-left:auto; text-transform:none; letter-spacing:0; font-weight:500; }
.cols { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:16px; }
.cols3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.cols2 { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; }
@media (max-width:900px){ .cols, .cols3, .cols2 { grid-template-columns:1fr; } }
.desc p { margin:0 0 10px; max-width:72ch; color:var(--ink-2); }
.desc p strong { color:var(--ink); font-weight:650; }
.desc .h { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); margin-bottom:1px; }
.plate { background:var(--plate); border-radius:10px; overflow:hidden; padding:6px; }
:root[data-theme="light"] .plate { border:1px solid var(--line); }
.plate img { display:block; width:100%; height:auto; }
svg.bfig { display:block; width:100%; height:auto; }
svg.bfig text { fill:#8f98a6; font:12px var(--sans); }
figure { margin:0; }
figcaption { font-size:12px; color:var(--ink-3); margin-top:6px; }
.figs { display:grid; gap:12px; }

/* vector region maps */
svg.fm { display:block; width:100%; height:auto; }
svg.fm path { fill:var(--fd, var(--fm-na)); stroke:var(--fm-edge); stroke-width:.6; transition:stroke-width .1s; }
:root[data-theme="light"] svg.fm path { fill:var(--fl, var(--fm-na)); }
svg.fm path.na, svg.fm path.shell { fill:var(--fm-na); }
svg.fm.aseg path { stroke-width:var(--sw, 1); }
svg.fm text { fill:var(--ink-3); font:13px var(--sans); }
svg.fm.aseg text { font-size:var(--fs, 13px); }
svg.fm .cb-light { display:none; }
:root[data-theme="light"] svg.fm .cb-light { display:inline; }
:root[data-theme="light"] svg.fm .cb-dark { display:none; }
svg.fm path.hl, svg.fm path[data-hl]:hover { stroke:var(--hl); stroke-width:3; }
svg.fm.aseg path.hl, svg.fm.aseg path[data-hl]:hover { stroke-width:calc(var(--sw, 1) * 4); }
tr.hl td { background:color-mix(in srgb, var(--hl) 14%, transparent); }
.row.hl .name { color:var(--hl); }

/* bars */
.rows { display:grid; gap:7px; }
.row { display:grid; grid-template-columns:minmax(0,1fr) 72px 44px; gap:10px; align-items:center; font-size:13px; }
.row .name { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; line-height:1.3; }
.row .name small { color:var(--ink-3); margin-left:6px; }
.track { height:8px; border-radius:4px; background:var(--comp-track); overflow:hidden; }
.track i { display:block; height:100%; border-radius:4px; background:var(--comp); }
.track.dmg { background:var(--lesion-track); }
.track.dmg i { background:var(--lesion); }
.chip { display:inline-block; font-size:11px; padding:0 7px; border-radius:999px; border:1px solid var(--line-2);
  color:var(--ink-2); margin-left:6px; white-space:nowrap; vertical-align:1px; }
.chip.warn { color:var(--warn); border-color:var(--warn); }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--ink-3); margin:2px 0 10px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.side { display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--comp-track); margin-top:8px; gap:2px; }
.side span { display:block; height:100%; }
.l { background:var(--left); } .r { background:var(--right); } .m { background:var(--mid); }
.note { font-size:12.5px; color:var(--ink-3); }
.dist { display:grid; gap:6px; font-size:13px; }
.dist div { display:flex; justify-content:space-between; gap:12px; border-bottom:1px solid var(--line); padding-bottom:4px; }

/* quality */
.flags { display:grid; gap:8px; }
.flag { display:grid; grid-template-columns:22px 1fr; gap:10px; padding:10px 12px; border-radius:10px;
  background:var(--info-bg); border:1px solid var(--line); }
.flag .ic { width:22px; height:22px; border-radius:50%; display:grid; place-items:center; font-weight:800;
  font-size:13px; color:#0d1015; background:var(--info); }
.flag.warn { background:var(--warn-bg); } .flag.warn .ic { background:var(--warn); }
.flag.error { background:var(--err-bg); } .flag.error .ic { background:var(--err); color:#fff; }
.flag b { display:block; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
.flag .sug { color:var(--ink-3); font-size:12.5px; }
.flags .verdict { margin:0 0 2px; font-size:14px; }
.status { display:inline-flex; align-items:center; gap:6px; font-weight:650; }
.status .dot { width:9px; height:9px; border-radius:50%; background:var(--ok); }
.status.warn .dot { background:var(--warn); } .status.error .dot { background:var(--err); }

/* details */
.section-title { margin:10px 2px 0; font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); font-weight:650; }
details.card { padding:0; }
details.card > summary { list-style:none; cursor:pointer; padding:14px 20px; display:flex; gap:10px; align-items:baseline; flex-wrap:wrap; }
details.card > summary::-webkit-details-marker { display:none; }
details.card > summary::before { content:"\25B8"; color:var(--ink-3); transition:transform .15s; }
details.card[open] > summary::before { transform:rotate(90deg); }
details.card > summary .t { font-weight:650; }
details.card > summary .s { color:var(--ink-3); font-size:12.5px; }
details.card > .inner { padding:0 20px 18px; }
table { border-collapse:collapse; width:100%; font-size:13px; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:middle; }
th { font-size:11.5px; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-3); font-weight:650; }
td .track { width:120px; display:inline-block; vertical-align:middle; margin-left:8px; }
.tablewrap { overflow-x:auto; }
th.sortable { cursor:pointer; user-select:none; white-space:nowrap; }
th.sortable:hover { color:var(--ink); }
th.sortable::after { content:"\2195"; margin-left:4px; opacity:.35; font-size:10px; }
th.sortable[aria-sort="ascending"]::after { content:"\25B2"; opacity:.9; }
th.sortable[aria-sort="descending"]::after { content:"\25BC"; opacity:.9; }
.tablewrap.scroll { max-height:min(60vh, 560px); overflow:auto; }
.tablewrap.scroll thead th { position:sticky; top:0; background:var(--panel); z-index:1;
  box-shadow:inset 0 -1px 0 var(--line); }
.card .count { font-weight:500; color:var(--ink-3); text-transform:none; letter-spacing:0; margin-left:6px; }
.theme-img.dark-img { display:block; } .theme-img.light-img { display:none; }
:root[data-theme="light"] .theme-img.dark-img { display:none; }
:root[data-theme="light"] .theme-img.light-img { display:block; }
.flatmap { margin-bottom:10px; }
.extras { border:1px solid var(--extra); background:linear-gradient(var(--extra-soft), var(--panel) 140px); }
.extras > h2 { color:var(--extra); }
.tier { font-size:10px; font-weight:800; letter-spacing:.06em; color:var(--extra); border:1px solid var(--extra);
  border-radius:4px; padding:0 5px; margin-left:6px; vertical-align:2px; }
.words { font-size:15px; color:var(--ink); margin:0 0 8px; }
.mods { list-style:none; padding:0; margin:0; display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:4px 18px; font-size:13px; }
.mods li { display:flex; gap:8px; }
.st { font-size:10.5px; font-weight:800; letter-spacing:.05em; text-transform:uppercase; min-width:84px; }
.st.ran { color:var(--ok); } .st.skipped, .st.unavailable { color:var(--ink-3); } .st.failed { color:var(--err); }
.qc-badge { font-size:11px; font-weight:700; border-radius:999px; padding:1px 8px; white-space:nowrap; }
.qc-badge.ok { background:var(--ok-bg); color:var(--ok); }
.qc-badge.info { background:var(--info-bg); color:var(--info); }
.qc-badge.warn { background:var(--warn-bg); color:var(--warn); }
.qc-badge.error { background:var(--err-bg); color:var(--err); }
.mini { display:flex; height:6px; width:70px; border-radius:3px; overflow:hidden; background:var(--comp-track); gap:1px; }
.mini span { height:100%; }
.thumb { width:46px; height:auto; display:block; background:var(--plate); border-radius:4px; }
code, .mono { font-family:var(--mono); font-size:12.5px; }
.kv { display:grid; grid-template-columns:max-content 1fr; gap:3px 16px; font-size:12.5px; margin:8px 0; }
.kv dt { color:var(--ink-3); } .kv dd { margin:0; word-break:break-all; }
.cmd { display:flex; gap:8px; align-items:center; background:var(--panel-2); border-radius:8px; padding:6px 10px; margin:6px 0; }
.cmd code { flex:1; overflow-x:auto; white-space:nowrap; }
.dl { display:flex; flex-wrap:wrap; gap:8px; }

/* filters */
.filters { display:flex; flex-wrap:wrap; gap:6px 10px; align-items:center; margin:0 0 12px; }
.filters input[type=search] { font:inherit; font-size:13px; color:var(--ink); background:var(--panel-2);
  border:1px solid var(--line-2); border-radius:8px; padding:5px 10px; min-width:220px; }
.fgroup { display:flex; flex-wrap:wrap; gap:4px; align-items:center; }
.fgroup .lab { font-size:11px; text-transform:uppercase; letter-spacing:.05em; color:var(--ink-3); margin-right:2px; }
.fchip { font:inherit; font-size:12px; color:var(--ink-2); background:transparent; border:1px solid var(--line-2);
  border-radius:999px; padding:1px 9px; cursor:pointer; }
.fchip[aria-pressed="true"] { background:var(--ink); color:var(--bg); border-color:var(--ink); }
.shown { font-size:12px; color:var(--ink-3); }

/* glossary */
.gl { border-bottom:1px dotted var(--ink-3); cursor:help; position:relative; }
.gl:hover::after, .gl:focus::after { content:attr(data-def); position:absolute; left:0; top:calc(100% + 6px); z-index:20;
  width:max-content; max-width:320px; white-space:normal; background:var(--panel-2); color:var(--ink);
  border:1px solid var(--line-2); border-radius:8px; padding:8px 10px; font:12.5px/1.45 var(--sans);
  font-weight:400; box-shadow:0 8px 24px rgba(0,0,0,.25); }

/* slice viewer */
.viewer { position:relative; background:var(--plate); border-radius:10px; overflow:hidden; }
.viewer canvas { display:block; width:100%; }
.viewer img { display:none; }

/* ---------- clinical report ---------- */
html[data-view="clinical"] body { background:#e9ebee; }
.doc { background:#fff; color:#15181d; max-width:210mm; margin:24px auto 64px; padding:18mm 18mm 16mm;
  box-shadow:0 1px 3px rgba(0,0,0,.12), 0 12px 32px rgba(0,0,0,.10); font:10.5pt/1.5 var(--serif);
  --plate:#fff; --fm-na:#d9dce1; --fm-edge:#2b2f36; --ink-3:#5b6370; }
.doc h1 { font:700 17pt/1.2 var(--sans); margin:0 0 2mm; letter-spacing:-.01em; }
.doc .sub { font:9pt/1.4 var(--sans); color:#5b6370; margin-bottom:6mm; }
.doc h2 { font:700 11pt/1.3 var(--sans); margin:7mm 0 2.5mm; padding-bottom:1.5mm; border-bottom:1.5px solid #15181d;
  break-after:avoid; }
.doc h2 .no { color:#8a8f98; margin-right:2mm; font-weight:600; }
.doc p { margin:0 0 2.5mm; }
.doc strong { font-weight:700; }
.doc table { font:9pt/1.4 var(--sans); break-inside:auto; }
.doc th, .doc td { border-bottom:1px solid #d9dde2; padding:1.4mm 2mm; color:#15181d; }
.doc th { color:#5b6370; font-size:8pt; }
.doc .meta td:first-child { color:#5b6370; width:34%; }
.doc .flagline { font:9.5pt/1.45 var(--sans); margin:0 0 1.8mm; padding-left:5mm; position:relative; }
.doc .flagline::before { content:"\25B2"; position:absolute; left:0; top:0; font-size:8pt; color:#8a5d00; }
.doc .flagline.error::before { color:#c0342f; }
.doc .ok { font:9.5pt var(--sans); color:#0b6b0b; }
.doc figure { break-inside:avoid; margin:3mm 0 4mm; }
.doc img { display:block; max-width:100%; height:auto; }
.doc .plate { max-width:170mm; background:#fff; border:1px solid #d9dde2; border-radius:2mm; }
.doc svg.bfig text, .doc svg.fm text { fill:#5b6370; }
.doc svg.fm path { fill:var(--fl, #d9dce1); stroke:#2b2f36; }
.doc svg.fm path.na, .doc svg.fm path.shell { fill:#d9dce1; }
.doc svg.fm .cb-light { display:inline; } .doc svg.fm .cb-dark { display:none; }
.doc figcaption { font:8.5pt/1.4 var(--sans); color:#5b6370; }
.doc .extra-tag { font:700 7.5pt var(--sans); letter-spacing:.06em; color:#4a3aa7; border:1px solid #4a3aa7;
  border-radius:1mm; padding:0 1.2mm; margin-left:2mm; vertical-align:1pt; }
.doc .fine { font:8pt/1.45 var(--sans); color:#5b6370; }
.doc .two { display:grid; grid-template-columns:1fr 1fr; gap:5mm; }
.doc .gl { border-bottom:0; cursor:auto; }
.doc .gl::after { content:none !important; }

/* ---------- print: always the clinical document, on white ---------- */
@page { size:A4; margin:12mm; }
@media print {
  :root { color-scheme:light; }
  body, html[data-view="clinical"] body { background:#fff !important; }
  .bar, .view-dash, .skip { display:none !important; }
  .view-clin { display:block !important; }
  .doc { box-shadow:none; margin:0; padding:0; max-width:none; }
  th.sortable::after { content:none !important; }
  a { color:inherit; text-decoration:none; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

JS = r"""
(function () {
  var root = document.documentElement;
  function get(k, d) { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }
  function put(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function each(sel, fn, scope) { Array.prototype.forEach.call((scope || document).querySelectorAll(sel), fn); }
  function setView(v) {
    root.setAttribute("data-view", v); put("lqtpy.view", v);
    each("[data-set-view]", function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-set-view") === v));
    });
  }
  function setTheme(t) {
    root.setAttribute("data-theme", t); put("lqtpy.theme", t);
    var b = document.getElementById("theme-btn");
    if (b) { b.textContent = t === "dark" ? "Light mode" : "Dark mode"; }
    document.dispatchEvent(new CustomEvent("lqtpy-theme", { detail: t }));
  }
  // Figures are embedded once; the clinical view copies them by id.
  each("[data-copy-from]", function (el) {
    var src = document.getElementById(el.getAttribute("data-copy-from"));
    if (src) { el.innerHTML = src.innerHTML; }
  });
  each("img[data-src-from]", function (img) {
    var src = document.getElementById(img.getAttribute("data-src-from"));
    if (src) { img.src = src.src; }
  });
  // A link can pick the view/theme: report.html#clinical, #dashboard, #light, #dark.
  var h = (location.hash || "").toLowerCase();
  setTheme(/light/.test(h) ? "light" : /dark/.test(h) ? "dark" : get("lqtpy.theme", "dark"));
  setView(/clinical/.test(h) ? "clinical" : /dashboard/.test(h) ? "dashboard"
          : get("lqtpy.view", "dashboard"));
  each("[data-set-view]", function (b) {
    b.addEventListener("click", function () { setView(b.getAttribute("data-set-view")); });
  });
  var tb = document.getElementById("theme-btn");
  if (tb) tb.addEventListener("click", function () {
    setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });

  // ---- sortable tables, remembered per table id ----
  function key(td) {
    if (!td) return "";
    var v = td.hasAttribute("data-sort") ? td.getAttribute("data-sort") : td.textContent.trim();
    var m = v.replace(/,/g, "").match(/^[<>~]?\s*(-?\d+(\.\d+)?)/);
    return m ? parseFloat(m[1]) : v.toLowerCase();
  }
  function sortTable(table, col, dir) {
    var head = table.tHead, body = table.tBodies[0], ths = head.rows[0].cells;
    var rows = Array.prototype.slice.call(body.rows);
    rows.sort(function (a, b) {
      var x = key(a.cells[col]), y = key(b.cells[col]);
      var xn = typeof x === "number", yn = typeof y === "number";
      var c = xn && yn ? x - y : xn ? -1 : yn ? 1 : String(x).localeCompare(String(y));
      if (xn !== yn) return c;  // blanks last in both directions
      return dir === "ascending" ? c : -c;
    });
    rows.forEach(function (r) { body.appendChild(r); });
    Array.prototype.forEach.call(ths, function (o) { o.removeAttribute("aria-sort"); });
    ths[col].setAttribute("aria-sort", dir);
  }
  each("table", function (table) {
    var head = table.tHead, body = table.tBodies[0];
    if (!head || !body || body.rows.length < 2) return;
    var tid = table.getAttribute("data-table");
    var ths = head.rows[0].cells;
    Array.prototype.forEach.call(ths, function (th, col) {
      th.classList.add("sortable");
      th.setAttribute("tabindex", "0");
      th.setAttribute("title", "Sort by this column");
      function sort() {
        var cur = th.getAttribute("aria-sort");
        var numeric = Array.prototype.every.call(body.rows, function (r) {
          var k = key(r.cells[col]); return typeof k === "number" || k === "" || k === "—";
        });
        var dir = cur ? (cur === "ascending" ? "descending" : "ascending")
                      : (numeric ? "descending" : "ascending");
        sortTable(table, col, dir);
        if (tid) put("lqtpy.sort." + tid, col + ":" + dir);
      }
      th.addEventListener("click", sort);
      th.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sort(); }
      });
    });
    if (tid) {
      var saved = get("lqtpy.sort." + tid, "");
      var m = saved.match(/^(\d+):(ascending|descending)$/);
      if (m && +m[1] < ths.length) sortTable(table, +m[1], m[2]);
    }
  });

  // ---- link highlighting: table rows, bars and map regions share data-hl ----
  function hl(k, on) {
    if (!k) return;
    each('[data-hl="' + k.replace(/"/g, '\\"') + '"]', function (el) { el.classList.toggle("hl", on); });
  }
  each("[data-hl]", function (el) {
    var k = el.getAttribute("data-hl");
    el.addEventListener("mouseenter", function () { hl(k, true); });
    el.addEventListener("mouseleave", function () { hl(k, false); });
    el.addEventListener("focusin", function () { hl(k, true); });
    el.addEventListener("focusout", function () { hl(k, false); });
  });

  // ---- copy buttons ----
  function copyText(text, btn) {
    function done() { var t = btn.textContent; btn.textContent = "Copied"; setTimeout(function () { btn.textContent = t; }, 1400); }
    if (navigator.clipboard && window.isSecureContext) {
      navigator.clipboard.writeText(text).then(done, fallback);
    } else { fallback(); }
    function fallback() {
      var ta = document.createElement("textarea"); ta.value = text; ta.style.position = "fixed"; ta.style.opacity = "0";
      document.body.appendChild(ta); ta.select();
      try { document.execCommand("copy"); done(); } catch (e) {}
      document.body.removeChild(ta);
    }
  }
  each("[data-copy]", function (btn) {
    btn.addEventListener("click", function () {
      var src = document.getElementById(btn.getAttribute("data-copy"));
      if (src) copyText(src.textContent.trim(), btn);
    });
  });

  // ---- downloads of the report's own tables ----
  each("[data-dl]", function (btn) {
    btn.addEventListener("click", function () {
      var src = document.getElementById(btn.getAttribute("data-dl"));
      if (!src) return;
      var bin = atob(src.textContent.trim()), bytes = new Uint8Array(bin.length);
      for (var i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
      var url = URL.createObjectURL(new Blob([bytes], { type: "text/csv" }));
      var a = document.createElement("a"); a.href = url; a.download = btn.getAttribute("data-name");
      document.body.appendChild(a); a.click(); document.body.removeChild(a);
      setTimeout(function () { URL.revokeObjectURL(url); }, 2000);
    });
  });

  // ---- group filters: search + chips; cohort bars recomputed ----
  var table = document.querySelector("table[data-table='subjects']");
  var dataEl = document.getElementById("lqtpy-subjects");
  if (table && dataEl) {
    var data = JSON.parse(dataEl.textContent);
    var search = document.getElementById("subj-search");
    var active = {};
    function apply() {
      var q = (search && search.value || "").toLowerCase().trim();
      var shown = [];
      Array.prototype.forEach.call(table.tBodies[0].rows, function (r) {
        var ok = !q || r.textContent.toLowerCase().indexOf(q) >= 0;
        Object.keys(active).forEach(function (f) {
          var vals = active[f]; if (!vals.length) return;
          if (vals.indexOf(r.getAttribute("data-f-" + f) || "") < 0) ok = false;
        });
        r.hidden = !ok;
        if (ok) shown.push(r.getAttribute("data-subject"));
      });
      var cnt = document.getElementById("subj-shown");
      var total = table.tBodies[0].rows.length;
      if (cnt) cnt.textContent = shown.length === total ? "" :
        "Showing " + shown.length + " of " + total + (data.subjects.length ? "; cohort bars recomputed" : "");
      if (data.subjects.length) recompute(shown);
    }
    function recompute(names) {
      var set = {}; names.forEach(function (n) { set[n] = 1; });
      var subs = data.subjects.filter(function (s) { return set[s.name]; });
      each("[data-rollup]", function (box) {
        var key = box.getAttribute("data-rollup"), skip = (box.getAttribute("data-skip") || "").split("|");
        var agg = {};
        subs.forEach(function (s) {
          var r = s.rollups[key] || {};
          Object.keys(r).forEach(function (k) {
            if (skip.indexOf(k) >= 0) return;
            agg[k] = agg[k] || { sum: 0, n: 0 }; agg[k].sum += r[k]; agg[k].n += 1;
          });
        });
        var n = subs.length || 1;
        var rows = Object.keys(agg).map(function (k) { return { k: k, mean: agg[k].sum / n, n: agg[k].n }; });
        rows.sort(function (a, b) { return b.n - a.n || b.mean - a.mean; });
        rows = rows.slice(0, 10);
        var top = rows.reduce(function (m, r) { return Math.max(m, r.mean); }, 0) || 1;
        var names_ = data.labels[key] || {};
        box.innerHTML = rows.length ? rows.map(function (r) {
          var nm = names_[r.k] || r.k, w = (100 * r.mean / top).toFixed(1);
          var v = r.mean > 0 && r.mean < 1 ? "<1%" : r.mean.toFixed(1) + "%";
          return '<div class="row"><span class="name" title="' + esc(nm) + '">' + esc(nm) +
            '<span class="chip">' + r.n + "/" + subs.length + '</span></span><span class="track dmg" aria-hidden="true"><i style="width:' +
            w + '%"></i></span><span class="num">' + v + "</span></div>";
        }).join("") : '<p class="note">No subjects in this selection.</p>';
      });
    }
    function esc(s) { return String(s).replace(/[&<>"]/g, function (c) { return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]; }); }
    if (search) search.addEventListener("input", apply);
    each(".fchip", function (b) {
      b.addEventListener("click", function () {
        var f = b.getAttribute("data-f"), v = b.getAttribute("data-v");
        active[f] = active[f] || [];
        var i = active[f].indexOf(v);
        if (i >= 0) active[f].splice(i, 1); else active[f].push(v);
        b.setAttribute("aria-pressed", String(i < 0));
        apply();
      });
    });
    var clear = document.getElementById("subj-clear");
    if (clear) clear.addEventListener("click", function () {
      active = {}; if (search) search.value = "";
      each(".fchip", function (b) { b.setAttribute("aria-pressed", "false"); });
      apply();
    });
  }

  // ---- scroll-through slice viewer (brainsprite) ----
  var vbox = document.getElementById("viewer");
  if (vbox && window.brainsprite) {
    var params = JSON.parse(document.getElementById("viewer-params").textContent);
    var started = false;
    function draw() {
      var old = document.getElementById("viewer-canvas");
      var c = old.cloneNode(false);  // a fresh canvas drops old listeners
      old.parentNode.replaceChild(c, old);
      var dark = root.getAttribute("data-theme") !== "light";
      var p = JSON.parse(JSON.stringify(params));
      p.canvas = "viewer-canvas"; p.sprite = "viewer-bg"; p.overlay.sprite = "viewer-ov";
      p.colorBackground = dark ? "#000000" : "#ffffff";
      p.colorFont = dark ? "#e9edf2" : "#121820";
      try { window.brainsprite(p); } catch (e) {}
    }
    function start() {
      if (started) return; started = true;
      var imgs = [document.getElementById("viewer-bg"), document.getElementById("viewer-ov")];
      var left = imgs.length;
      imgs.forEach(function (im) {
        if (im.complete) { if (--left === 0) draw(); }
        else im.addEventListener("load", function () { if (--left === 0) draw(); });
      });
    }
    var det = vbox.closest("details");
    if (det && !det.open) det.addEventListener("toggle", function () { if (det.open) start(); });
    else start();
    document.addEventListener("lqtpy-theme", function () { if (started) draw(); });
  }

  var pb = document.getElementById("pdf-btn");
  if (pb) pb.addEventListener("click", function () { window.print(); });
})();
"""

#: term -> one-line definition and the rule LQTpy used. Matched
#: case-insensitively, first occurrence per paragraph.
GLOSSARY = {
    "lacunar-sized": "A single small subcortical lesion at most 20 mm across (axial) and at most "
                     "1.5 cm³: the size of a lacunar infarct.",
    "periventricular": "Reaching the lateral ventricles: within 3 mm of them in the MNI template.",
    "juxtacortical": "White matter reaching the cortex: within 3 mm of it in the MNI template.",
    "infratentorial": "Mostly (50%+) in the brainstem or cerebellum.",
    "cortico-subcortical": "At least 10% of the lesion's brain tissue is cortex and at least 10% is "
                           "white or deep grey matter.",
    "deep grey matter": "The basal ganglia and thalamus.",
    "deep perforator": "Territory of small deep arteries (lenticulostriate, thalamoperforator, "
                       "basilar perforators) rather than the cortical branches.",
    "border zone": "Where two arterial territories meet (a watershed area). LQTpy's zones are the "
                   "6 mm either side of the atlas's territory boundaries: approximate.",
    "watershed": "A border zone between arterial territories, where infarcts are usually put down "
                 "to low flow rather than one blocked artery.",
    "lenticulostriate": "Small deep branches of the middle (lateral) and anterior (medial) cerebral "
                        "arteries supplying the basal ganglia and internal capsule.",
    "Yeo-7": "Seven large-scale functional networks (Yeo et al. 2011), here from the Schaefer-100 "
             "parcels.",
    "homologue": "The same place in the other hemisphere (the lesion's mirror image).",
    "laterality index": "(left - right) / (left + right) volume: +1 fully left, -1 fully right.",
    "cross-section": "A 1 mm slab across a tract's main axis; the load is the share of that "
                     "slab the lesion covers.",
    "functional network": "Yeo-7 network from the Schaefer-100 parcels.",
    "location labels": "Functional names describe where a region is, not what symptoms to expect.",
}
