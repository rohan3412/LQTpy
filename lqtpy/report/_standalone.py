"""LQTpy's self-contained HTML reports: a dashboard and a clinical report in
one file.

``render_single`` (one lesion) and ``render_group`` (a cohort) turn an
``analyze_single``/``analyze_group`` result into a single HTML file with
every image embedded. The file holds two views, switched from its toolbar:

- **Dashboard** -- dark by default (light on request), summary first: key
  numbers, the written description, quality flags, figures, then where the
  lesion is. Atlas tables and extras sit below, collapsed.
- **Clinical report** -- a paper-like document (serif text, numbered
  sections, figure captions). This is also what "Export PDF" prints, and
  what :func:`lqtpy.report.to_pdf` produces.

Colour is informative and used the same way everywhere:

=================  ========================================================
lesion orange      how much of a structure is damaged (bars, flat maps)
neutral ink        what share of the lesion lies somewhere
blue / teal        left / right hemisphere
violet             extras (anything not core)
status colours     QC only, always with an icon and a word
=================  ========================================================

Every data value is escaped: region labels can come from host-registered
atlases and are treated as untrusted.
"""
from __future__ import annotations

import datetime
import html

from .style import DEFAULT_STYLE, ReportStyle

__all__ = ["render_single", "render_group"]


def _esc(x) -> str:
    return html.escape(str(x))


def _records(x) -> "list[dict]":
    if x is None:
        return []
    if hasattr(x, "to_dict") and hasattr(x, "columns"):
        return x.to_dict("records")
    return list(x)


def _basename(path) -> str:
    import os
    return os.path.basename(str(path).replace("\\", "/")) if path else ""


def _pct(v, digits=0) -> str:
    v = float(v)
    if 0 < v < 1 and digits == 0:
        return "<1%"
    return f"{v:.{digits}f}%"


# =========================================================================== #
# Stylesheet and behaviour (plain strings: no f-string brace escaping)
# =========================================================================== #
_CSS = r"""
:root {
  color-scheme: dark;
  --bg:#0d1015; --panel:#151a21; --panel-2:#1b212a; --line:rgba(255,255,255,.08);
  --line-2:rgba(255,255,255,.16); --ink:#e9edf2; --ink-2:#b3bcc8; --ink-3:#8a93a1;
  --lesion:#f07a45; --lesion-track:rgba(240,122,69,.14); --comp:#8f9aab; --comp-track:rgba(255,255,255,.06);
  --left:#3987e5; --right:#199e70; --mid:#6b7480;
  --extra:#9085e9; --extra-soft:rgba(144,133,233,.10);
  --ok:#35c04f; --warn:#fab219; --err:#e5534b; --info:#8a93a1;
  --ok-bg:rgba(12,163,12,.14); --warn-bg:rgba(250,178,25,.13); --err-bg:rgba(208,59,59,.16); --info-bg:rgba(255,255,255,.05);
  --focus:#7ab3ff; --shadow:0 1px 0 rgba(255,255,255,.03), 0 8px 24px rgba(0,0,0,.25);
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
  --serif:Charter,"Bitstream Charter","Sitka Text",Cambria,Georgia,serif;
  --mono:ui-monospace,"SF Mono",Consolas,"Liberation Mono",monospace;
}
:root[data-theme="light"] {
  color-scheme: light;
  --bg:#f4f5f7; --panel:#ffffff; --panel-2:#f1f3f6; --line:#e3e6eb; --line-2:#cfd4dc;
  --ink:#121820; --ink-2:#4a5361; --ink-3:#6c7584;
  --lesion:#e0602a; --lesion-track:rgba(224,96,42,.12); --comp:#8a94a3; --comp-track:#edf0f3;
  --left:#2a78d6; --right:#1a9e6f; --mid:#9aa3ae;
  --extra:#4a3aa7; --extra-soft:rgba(74,58,167,.06);
  --ok:#0b8a0b; --warn:#9a6700; --err:#c0342f; --info:#6c7584;
  --ok-bg:#e8f5e8; --warn-bg:#fff4d6; --err-bg:#fde8e6; --info-bg:#f1f3f6;
  --focus:#2a78d6; --shadow:0 1px 2px rgba(16,24,40,.05);
}
* { box-sizing:border-box; }
html, body { margin:0; }
body { background:var(--bg); color:var(--ink); font:14px/1.55 var(--sans); -webkit-font-smoothing:antialiased; }
a { color:var(--focus); }
:focus-visible { outline:2px solid var(--focus); outline-offset:2px; border-radius:4px; }
.num, td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; }
.muted { color:var(--ink-3); }

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
.kpis { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.kpi { background:var(--panel); border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.kpi .k { font-size:11.5px; text-transform:uppercase; letter-spacing:.06em; color:var(--ink-3); }
.kpi .v { font-size:22px; font-weight:650; margin-top:2px; line-height:1.2; }
.kpi .v.text { font-size:17px; }
.kpi .s { font-size:12.5px; color:var(--ink-2); margin-top:2px; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:14px; padding:18px 20px;
  box-shadow:var(--shadow); min-width:0; }
.card > h2 { margin:0 0 12px; font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); font-weight:650; }
.cols { display:grid; grid-template-columns:minmax(0,1.15fr) minmax(0,1fr); gap:16px; }
.cols3 { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
@media (max-width:900px){ .cols, .cols3 { grid-template-columns:1fr; } }
.desc p { margin:0 0 10px; max-width:72ch; color:var(--ink-2); }
.desc p strong { color:var(--ink); font-weight:650; }
.desc .h { display:block; font-size:11px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); margin-bottom:1px; }
.plate { background:#000; border-radius:10px; overflow:hidden; }
.plate img { display:block; width:100%; height:auto; }
figure { margin:0; }
figcaption { font-size:12px; color:var(--ink-3); margin-top:6px; }
.figs { display:grid; gap:12px; }

/* bars */
.rows { display:grid; gap:7px; }
.row { display:grid; grid-template-columns:minmax(0,1fr) 72px 40px; gap:10px; align-items:center; font-size:13px; }
.row .name { display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; line-height:1.3; }
.row .name small { color:var(--ink-3); margin-left:6px; }
.track { height:8px; border-radius:4px; background:var(--comp-track); overflow:hidden; }
.track i { display:block; height:100%; border-radius:4px; background:var(--comp); }
.track.dmg { background:var(--lesion-track); }
.track.dmg i { background:var(--lesion); }
.chip { display:inline-block; font-size:11px; padding:0 7px; border-radius:999px; border:1px solid var(--line-2);
  color:var(--ink-2); margin-left:6px; white-space:nowrap; vertical-align:1px; }
.legend { display:flex; gap:14px; flex-wrap:wrap; font-size:12px; color:var(--ink-3); margin:2px 0 10px; }
.legend i { display:inline-block; width:10px; height:10px; border-radius:2px; margin-right:6px; vertical-align:-1px; }
.side { display:flex; height:10px; border-radius:5px; overflow:hidden; background:var(--comp-track); margin-top:8px; gap:2px; }
.side span { display:block; height:100%; }
.l { background:var(--left); } .r { background:var(--right); } .m { background:var(--mid); }

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
.status { display:inline-flex; align-items:center; gap:6px; font-weight:650; }
.status .dot { width:9px; height:9px; border-radius:50%; background:var(--ok); }
.status.warn .dot { background:var(--warn); } .status.error .dot { background:var(--err); }

/* details */
.section-title { margin:10px 2px 0; font-size:13px; text-transform:uppercase; letter-spacing:.07em; color:var(--ink-3); font-weight:650; }
details.card { padding:0; }
details.card > summary { list-style:none; cursor:pointer; padding:14px 20px; display:flex; gap:10px; align-items:baseline; }
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
.theme-img.dark-img { display:block; } .theme-img.light-img { display:none; }
:root[data-theme="light"] .theme-img.dark-img { display:none; }
:root[data-theme="light"] .theme-img.light-img { display:block; }
.flatmap img { width:100%; height:auto; border-radius:8px; }
.note { font-size:12.5px; color:var(--ink-3); }
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
.qc-badge.warn { background:var(--warn-bg); color:var(--warn); }
.qc-badge.error { background:var(--err-bg); color:var(--err); }
.mini { display:flex; height:6px; width:70px; border-radius:3px; overflow:hidden; background:var(--comp-track); gap:1px; }
.mini span { height:100%; }

/* ---------- clinical report ---------- */
html[data-view="clinical"] body { background:#e9ebee; }
.doc { background:#fff; color:#15181d; max-width:210mm; margin:24px auto 64px; padding:18mm 18mm 16mm;
  box-shadow:0 1px 3px rgba(0,0,0,.12), 0 12px 32px rgba(0,0,0,.10); font:10.5pt/1.5 var(--serif); }
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
.doc .flagline::before { content:"\25B2"; position:absolute; left:0; top:0; font-size:8pt; color:#9a6700; }
.doc .flagline.error::before { color:#c0342f; }
.doc .ok { font:9.5pt var(--sans); color:#0b6b0b; }
.doc figure { break-inside:avoid; margin:3mm 0 4mm; }
.doc img { display:block; max-width:100%; height:auto; }
.doc .plate { max-width:150mm; }
.doc figure .plate { border-radius:2mm; }
.doc figcaption { font:8.5pt/1.4 var(--sans); color:#5b6370; }
.doc .extra-tag { font:700 7.5pt var(--sans); letter-spacing:.06em; color:#4a3aa7; border:1px solid #4a3aa7;
  border-radius:1mm; padding:0 1.2mm; margin-left:2mm; vertical-align:1pt; }
.doc .fine { font:8pt/1.45 var(--sans); color:#5b6370; }
.doc .two { display:grid; grid-template-columns:1fr 1fr; gap:5mm; }

/* ---------- print: always the clinical document, on white ---------- */
@page { size:A4; margin:12mm; }
@media print {
  :root { color-scheme:light; }
  body, html[data-view="clinical"] body { background:#fff !important; }
  .bar, .view-dash { display:none !important; }
  .view-clin { display:block !important; }
  .doc { box-shadow:none; margin:0; padding:0; max-width:none; }
  a { color:inherit; text-decoration:none; }
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

_JS = r"""
(function () {
  var root = document.documentElement;
  function get(k, d) { try { return localStorage.getItem(k) || d; } catch (e) { return d; } }
  function put(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  function setView(v) {
    root.setAttribute("data-view", v); put("lqtpy.view", v);
    document.querySelectorAll("[data-set-view]").forEach(function (b) {
      b.setAttribute("aria-pressed", String(b.getAttribute("data-set-view") === v));
    });
  }
  function setTheme(t) {
    root.setAttribute("data-theme", t); put("lqtpy.theme", t);
    var b = document.getElementById("theme-btn");
    if (b) { b.textContent = t === "dark" ? "Light mode" : "Dark mode"; }
  }
  // Images are embedded once; the clinical view borrows them by id.
  document.querySelectorAll("img[data-src-from]").forEach(function (img) {
    var src = document.getElementById(img.getAttribute("data-src-from"));
    if (src) { img.src = src.src; }
  });
  // A link can pick the view/theme: report.html#clinical, #dashboard, #light, #dark.
  var h = (location.hash || "").toLowerCase();
  setTheme(/light/.test(h) ? "light" : /dark/.test(h) ? "dark" : get("lqtpy.theme", "dark"));
  setView(/clinical/.test(h) ? "clinical" : /dashboard/.test(h) ? "dashboard"
          : get("lqtpy.view", "dashboard"));
  document.querySelectorAll("[data-set-view]").forEach(function (b) {
    b.addEventListener("click", function () { setView(b.getAttribute("data-set-view")); });
  });
  var tb = document.getElementById("theme-btn");
  if (tb) tb.addEventListener("click", function () {
    setTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
  });
  var pb = document.getElementById("pdf-btn");
  if (pb) pb.addEventListener("click", function () { window.print(); });
})();
"""


def _page(title: str, dash: str, clin: str, *, subtitle: str = "") -> str:
    return (
        '<!doctype html><html lang="en" data-theme="dark" data-view="dashboard"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{_esc(title)}</title><style>{_CSS}</style></head><body>"
        '<div class="bar" role="toolbar" aria-label="Report view">'
        f'<span class="brand">LQTpy<small>{_esc(subtitle)}</small></span><span class="spacer"></span>'
        '<span class="seg" role="group" aria-label="View">'
        '<button type="button" data-set-view="dashboard" aria-pressed="true">Dashboard</button>'
        '<button type="button" data-set-view="clinical" aria-pressed="false">Clinical report</button>'
        '</span>'
        '<button type="button" class="btn theme-only" id="theme-btn">Light mode</button>'
        '<button type="button" class="btn primary" id="pdf-btn" title="Opens the print dialog; '
        'choose Save as PDF. Prints the clinical report.">Export PDF</button>'
        "</div>"
        f'<main class="view view-dash">{dash}</main>'
        f'<main class="view view-clin">{clin}</main>'
        f"<script>{_JS}</script></body></html>"
    )


# =========================================================================== #
# Shared pieces
# =========================================================================== #
_KIND = {"single": "Single lesion", "group": "Group", "group-member": "Single lesion (from a group)"}
_TERR = {"ACA": "anterior cerebral artery", "MCA": "middle cerebral artery",
         "PCA": "posterior cerebral artery", "VB": "vertebrobasilar"}
_SHORT = {"aal": "AAL", "schaefer": "Schaefer", "dk": "DKT", "jhu": "JHU", "arterial": "Arterial"}


def _bar(value, cls="", max_value=100.0) -> str:
    w = 0.0 if not max_value else max(0.0, min(100.0, 100.0 * float(value) / float(max_value)))
    return f'<span class="track {cls}" aria-hidden="true"><i style="width:{w:.1f}%"></i></span>'


def _row(name, value, cls, extra="", max_value=100.0, digits=0) -> str:
    return (f'<div class="row"><span class="name" title="{_esc(name)}">{_esc(name)}{extra}</span>'
            f'{_bar(value, cls, max_value)}<span class="num">{_pct(value, digits)}</span></div>')


def _emph(text: str) -> str:
    """Escape, turning **...** spans (key findings) into <strong>."""
    parts = str(text).split("**")
    return "".join(_esc(p) if i % 2 == 0 else f"<strong>{_esc(p)}</strong>"
                   for i, p in enumerate(parts))


def _desc_sections(res) -> "list[tuple[str, str]]":
    d = res.get("description") or {}
    return [(h, p) for h, p in (d.get("marked") or d.get("sections") or []) if h != "Quality"]


def _qc_worst(flags) -> str:
    sev = {f["severity"] for f in flags or []}
    return "error" if "error" in sev else "warn" if "warn" in sev else "ok"


def _flags_html(flags) -> str:
    items = [f for f in flags or [] if f["severity"] != "info"] + \
            [f for f in flags or [] if f["severity"] == "info"]
    out = []
    for f in items:
        icon = {"error": "×", "warn": "!", "info": "i"}[f["severity"]]
        word = {"error": "Problem", "warn": "Check", "info": "Note"}[f["severity"]]
        sug = f'<div class="sug">{_esc(f["suggestion"])}</div>' if f.get("suggestion") else ""
        out.append(f'<div class="flag {f["severity"]}"><span class="ic" aria-hidden="true">{icon}</span>'
                   f'<div><b>{word}</b>{_esc(f["message"])}{sug}</div></div>')
    return f'<div class="flags">{"".join(out)}</div>'


def _modules(manifest) -> str:
    items = []
    for m in manifest.get("modules", []):
        tier = '<span class="tier">EXTRA</span>' if m.get("tier") == "extra" else ""
        why = f' <span class="muted">— {_esc(m["reason"])}</span>' if m.get("reason") else ""
        items.append(f'<li><span class="st {_esc(m["status"])}">{_esc(m["status"])}</span>'
                     f'<span>{_esc(m["label"])}{tier}{why}</span></li>')
    return f'<ul class="mods">{"".join(items)}</ul>'


def _scope_pill(manifest) -> str:
    if manifest.get("scope", "core") != "core":
        return '<span class="pill extra">Core + extras</span>'
    return '<span class="pill">Core only</span>'


def _about(manifest, extra_rows="") -> str:
    s = manifest.get("settings", {})
    sets = [f"lqtpy {_esc(manifest.get('lqtpy_version', ''))}",
            f"threshold &gt; {_esc(s.get('threshold', 0.5))}"]
    if s.get("profile"):
        sets.append(f"description profile: {_esc(s['profile'])}")
    return (f'<details class="card"><summary><span class="t">About this report</span>'
            f'<span class="s">{" · ".join(sets)}</span></summary><div class="inner">'
            f"{_modules(manifest)}{extra_rows}"
            '<p class="note">Physical measures use the image affine. Region overlap resamples the '
            'mask onto each atlas grid (nearest neighbour). Functional area labels describe '
            'location, not symptoms. Not for clinical decision-making on its own.</p>'
            "</div></details>")


def _img(uri, alt, id_=None) -> str:
    if not uri:
        return ""
    idattr = f' id="{_esc(id_)}"' if id_ else ""
    return f'<img{idattr} src="{uri}" alt="{_esc(alt)}" loading="lazy">'


def _borrow(id_, alt) -> str:
    return f'<img data-src-from="{_esc(id_)}" alt="{_esc(alt)}">'


def _flatmaps(a, id_prefix) -> str:
    out = []
    for i, (cap, uris) in enumerate(a.get("flat_uris") or []):
        if isinstance(uris, str):
            uris = {"light": uris, "dark": uris}
        light = _img(uris.get("light"), cap, f"{id_prefix}-flat{i}")
        dark = _img(uris.get("dark"), cap)
        out.append(f'<figure class="flatmap"><div class="theme-img dark-img">{dark}</div>'
                   f'<div class="theme-img light-img">{light}</div>'
                   f"<figcaption>{_esc(cap)}</figcaption></figure>")
    return "".join(out)


# =========================================================================== #
# Single lesion
# =========================================================================== #
def _main_structure(facts) -> "tuple[str, float] | None":
    best = None
    for h in facts.get("hemispheres", []):
        side = h["hemisphere"] if h["hemisphere"] in ("left", "right") else ""
        for lb in h["lobes"]:
            name = f"{side} {lb['lobe'].lower()}".strip()
            if best is None or lb["pct_of_lesion"] > best[1]:
                best = (name, lb["pct_of_lesion"])
        if h.get("wm_pct", 0) and (best is None or h["wm_pct"] > best[1]):
            best = (f"{side} white matter".strip(), h["wm_pct"])
    return best


def _composition(facts) -> "list[tuple[str, float]]":
    rows = []
    for h in facts.get("hemispheres", []):
        side = h["hemisphere"] if h["hemisphere"] in ("left", "right") else "midline"
        for lb in h["lobes"]:
            rows.append((f"{side} {lb['lobe'].lower()}", lb["pct_of_lesion"]))
        if h.get("wm_pct"):
            rows.append((f"{side} white matter (named tracts)", h["wm_pct"]))
        if h.get("unlabelled_pct"):
            rows.append((f"{side} unnamed tissue", h["unlabelled_pct"]))
    return sorted([r for r in rows if r[1] >= 0.5], key=lambda r: -r[1])


def _damaged(facts, limit=8) -> "list[dict]":
    rows = []
    for h in facts.get("hemispheres", []):
        side = h["hemisphere"] if h["hemisphere"] in ("left", "right") else ""
        for lb in h["lobes"]:
            for r in lb["regions"]:
                rows.append({"name": f"{side} {r['name']}".strip(), "dmg": r["pct_damaged"],
                             "share": r["pct_of_lesion"], "function": r.get("function")})
        for w in h.get("wm", []):
            rows.append({"name": f"{side} {w['name']}".strip(), "dmg": w["pct_damaged"],
                         "share": w["pct_of_lesion"], "function": None, "wm": True})
    rows = [r for r in rows if r["share"] >= 0.5 or r["dmg"] >= 5]
    return sorted(rows, key=lambda r: -r["dmg"])[:limit]


def _territories(facts, short: bool = False) -> "list[tuple[str, float, bool]]":
    """Territory parts. ``short``: 'L MCA · frontal part' / 'L lenticulostriate'."""
    rows = []
    for t in facts.get("territories", []):
        side = t["hemisphere"] if t["hemisphere"] in ("left", "right") else ""
        for pp in t["parts"]:
            label = pp["name"].replace(" territory", "")
            if short:
                s1 = {"left": "L ", "right": "R "}.get(side, "")
                if ", " in label:
                    label = f"{s1}{t['territory']} · {label.split(', ', 1)[1]}"
                elif pp["deep"]:
                    label = f"{s1}{label.replace('lateral ', 'lat. ').replace('medial ', 'med. ')}"
                else:
                    label = f"{s1}{t['territory']}"
                rows.append((label, pp["pct_of_lesion"], pp["deep"]))
            else:
                rows.append((f"{side} {label}".strip(), pp["pct_of_lesion"], pp["deep"]))
    return sorted([r for r in rows if r[1] >= 1], key=lambda r: -r[1])[:8]


def _atlas_details(a, prefix) -> str:
    recs = _records(a["df"])
    cov = a.get("coverage") or {}
    head_s = (f"{len(recs)} regions involved · {cov.get('PercentLabelled', 0):.0f}% of the lesion "
              f"is in {_SHORT.get(a['key'], a['key'])} regions") if cov else f"{len(recs)} regions"
    tier = '<span class="tier">EXTRA</span>' if a.get("tier") == "extra" else ""
    rows = "".join(
        f'<tr><td>{_esc(r["RegionName"])}</td><td class="n">{r["PercentDamage"]:.1f}%'
        f'{_bar(r["PercentDamage"], "dmg")}</td><td class="n">{r["PercentOfLesion"]:.1f}%'
        f'{_bar(r["PercentOfLesion"])}</td><td class="n">{r["LesionVoxels"]:,}</td></tr>'
        for r in recs[:40])
    more = (f'<p class="note">Showing 40 of {len(recs)} regions; every region is in the CSV.</p>'
            if len(recs) > 40 else "")
    rollups = ""
    for key, label in (("lobe_df", "Lobe"), ("network_df", "Network")):
        rr = _records(a.get(key))
        if rr:
            col = "Lobe" if key == "lobe_df" else "Network"
            name = {"schaefer": "Yeo-7 network", "arterial": "Territory"}.get(a["key"], label)
            rollups += (f'<h3 class="section-title">{_esc(name)}</h3><div class="tablewrap"><table>'
                        f'<thead><tr><th>{_esc(name)}</th><th class="n">% damaged</th>'
                        f'<th class="n">% of lesion</th></tr></thead><tbody>' + "".join(
                            f'<tr><td>{_esc(r[col])}</td><td class="n">{r["PercentDamage"]:.1f}%'
                            f'{_bar(r["PercentDamage"], "dmg")}</td><td class="n">{r["PercentOfLesion"]:.1f}%'
                            f'{_bar(r["PercentOfLesion"])}</td></tr>' for r in rr) + "</tbody></table></div>")
    table = ("" if not recs else
             '<div class="tablewrap"><table><thead><tr><th>Region</th><th class="n">% of region damaged</th>'
             f'<th class="n">% of lesion</th><th class="n">Voxels</th></tr></thead><tbody>{rows}'
             f"</tbody></table></div>{more}")
    return (f'<details class="card"><summary><span class="t">{_esc(a["name"])}{tier}</span>'
            f'<span class="s">{_esc(head_s)}</span></summary><div class="inner">'
            f'{_flatmaps(a, prefix)}{rollups}{table}</div></details>')


def _extras_card(res, manifest) -> str:
    measures = res.get("measures") or {}
    extra_mods = [m for m in manifest.get("modules", []) if m.get("tier") == "extra"
                  and m["id"].startswith("measure:")]
    if not measures:
        body = ('<p class="note">No extra measures ran for this report. Run with '
                '<code>extras="all"</code> (or <code>--extras all</code>) to add shape and other '
                'extra measures.</p>')
    else:
        blocks = []
        for name, m in measures.items():
            words = "; ".join(m.get("descriptors") or []) or "No interpretation available."
            vals = {k: v for k, v in (m.get("values") or {}).items() if k != "per_component"}
            rows = "".join(f'<tr><td>{_esc(k.replace("_", " "))}</td><td class="n">{_esc(v)}</td></tr>'
                           for k, v in vals.items())
            blocks.append(f'<h3 class="section-title">{_esc(name.capitalize())}</h3>'
                          f'<p class="words">{_esc(words)}</p><details><summary class="note">'
                          f'Numbers</summary><table><tbody>{rows}</tbody></table></details>')
        body = "".join(blocks)
    missing = "".join(f'<p class="note">{_esc(m["label"])}: {_esc(m["status"])}'
                      f'{" — " + _esc(m["reason"]) if m.get("reason") else ""}</p>'
                      for m in extra_mods if m["status"] not in ("ran",)
                      and not (m["status"] == "skipped" and not measures))
    return f'<section class="card extras"><h2>Extras</h2>{body}{missing}</section>'


def render_single(res: dict, style: "ReportStyle | None" = None, *,
                  back_link: "str | None" = None) -> str:
    """The single-lesion report (dashboard + clinical views in one file)."""
    style = style or DEFAULT_STYLE
    manifest = res.get("manifest") or {}
    m = res["morphometry"]
    facts = (res.get("description") or {}).get("facts") or {}
    flags = res.get("qc") or []
    ts = res.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    name = res["name"]
    src = res.get("source_path") or ""
    worst = _qc_worst(flags)
    n_warn = sum(1 for f in flags if f["severity"] in ("warn", "error"))
    main = _main_structure(facts)
    terrs = facts.get("territories") or []
    parts = facts.get("n_clusters", m["n_components"])

    # ---------------- dashboard ----------------
    back = f'<a class="back" href="{_esc(back_link)}">← Group report</a>' if back_link else ""
    hero = (f'<div class="hero"><div>{back}<h1>{_esc(name)}</h1>'
            f'<div class="meta" title="{_esc(src)}">{_esc(_basename(src))}{" · " if src else ""}{_esc(ts)} · analysed in '
            f'{_esc(res.get("elapsed", "?"))}</div></div><span class="spacer"></span>'
            f'{_scope_pill(manifest)}<span class="pill">{_esc(_KIND.get(manifest.get("kind"), "Report"))}</span></div>')
    side = (f'<div class="side" role="img" aria-label="{m["pct_left"]:.0f}% left, {m["pct_right"]:.0f}% right">'
            f'<span class="l" style="width:{m["pct_left"]}%"></span>'
            f'<span class="m" style="width:{m.get("pct_midline", 0)}%"></span>'
            f'<span class="r" style="width:{m["pct_right"]}%"></span></div>')
    qc_word = {"ok": "No problems", "warn": f"{n_warn} to check", "error": f"{n_warn} problem"
               + ("s" if n_warn != 1 else "")}[worst]
    kpis = [
        ("Volume", f'{m["volume_cc"]:.2f} cm³' if m["volume_cc"] < 10 else f'{m["volume_cc"]:.1f} cm³',
         f'{m["n_voxels"]:,} voxels'),
        ("Parts", str(parts), "separate parts" if parts != 1 else "one connected lesion"),
        ("Side", f'{m["pct_left"]:.0f}% L · {m["pct_right"]:.0f}% R', side),
        ("Main territory", (f'{terrs[0]["hemisphere"]} {terrs[0]["territory"]}' if terrs else "—"),
         f'{terrs[0]["pct_of_lesion"]:.0f}% of the lesion' if terrs else "no territory data"),
        ("Main location", main[0] if main else "—", f"{main[1]:.0f}% of the lesion" if main else ""),
        ("Quality", f'<span class="status {worst}"><span class="dot"></span>{_esc(qc_word)}</span>',
         "mask quality checks"),
    ]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{_esc(k)}</div>'
                       f'<div class="v{" text" if k in ("Main territory", "Main location", "Quality", "Side") else ""}">'
                       f'{v if k in ("Quality",) else _esc(v)}</div><div class="s">'
                       f'{s if k == "Side" else _esc(s)}</div></div>' for k, v, s in kpis)
    desc = "".join(f'<p><span class="h">{_esc(h)}</span>{_emph(p)}</p>' for h, p in _desc_sections(res))
    figs = []
    if res.get("glass_uri"):
        figs.append(f'<figure><div class="plate">{_img(res["glass_uri"], "Glass-brain projections of the lesion", "img-glass")}</div>'
                    '<figcaption>Glass-brain projections (left, right, top, front)</figcaption></figure>')
    if res.get("slices_uri"):
        figs.append(f'<figure><div class="plate">{_img(res["slices_uri"], "Orthogonal slices through the lesion centre", "img-slices")}</div>'
                    f'<figcaption>Slices through the lesion centre ({", ".join(str(c) for c in m["center_of_mass_mm"])} mm)</figcaption></figure>')
    top = f'<div class="cols"><section class="card desc"><h2>Description</h2>{desc or "<p>No description.</p>"}</section>'
    top += f'<section class="card"><h2>Lesion</h2><div class="figs">{"".join(figs) or "<p class=note>No figures (plots were off).</p>"}</div></section></div>'
    quality = (f'<section class="card"><h2>Quality</h2>{_flags_html(flags)}</section>' if flags else "")

    comp = "".join(_row(n, v, "") for n, v in _composition(facts)[:9])
    dmg = "".join(_row(r["name"], r["dmg"], "dmg",
                       (f'<span class="chip">{_esc(r["function"])}</span>' if r.get("function") else "")
                       + ('<small>white matter</small>' if r.get("wm") else "")) for r in _damaged(facts))
    terr = "".join(_row(n, v, "", '<span class="chip">deep</span>' if deep else "")
                   for n, v, deep in _territories(facts, short=True))
    where = ""
    if comp or dmg or terr:
        where = ('<div class="section-title">Where the lesion is</div><div class="cols3">'
                 f'<section class="card"><h2>Share of the lesion</h2><div class="legend"><span><i style="background:var(--comp)"></i>percent of the lesion</span></div><div class="rows">{comp}</div></section>'
                 f'<section class="card"><h2>Most damaged structures</h2><div class="legend"><span><i style="background:var(--lesion)"></i>percent of the structure damaged</span></div><div class="rows">{dmg}</div></section>'
                 f'<section class="card"><h2>Vascular territories</h2><div class="legend"><span><i style="background:var(--comp)"></i>percent of the lesion</span></div><div class="rows">{terr or "<p class=note>No territory data.</p>"}</div></section>'
                 "</div>")

    details = "".join(_atlas_details(a, f"a{i}") for i, a in enumerate(res["atlases"]))
    tr = res.get("tracts")
    if tr:
        trs = _records(tr["df"])
        rows = "".join(f'<tr><td>{_esc(r["Tract"])}</td><td class="n">{r["PercentOverlap"]:.1f}%'
                       f'{_bar(r["PercentOverlap"], "dmg")}</td><td class="n">{r["PercentOfLesion"]:.1f}%</td></tr>'
                       for r in trs[:25])
        details += ('<details class="card"><summary><span class="t">White-matter tracts (voxel-overlap proxy)'
                    f'</span><span class="s">{len(trs)} tracts touched · not streamline disconnection</span>'
                    '</summary><div class="inner"><p class="note">Percent of each tract\'s voxels the lesion '
                    'touches: a coarse proxy. A small lesion can cut a tract while touching few of its voxels.</p>'
                    '<div class="tablewrap"><table><thead><tr><th>Tract</th><th class="n">% of tract voxels</th>'
                    f'<th class="n">% of lesion</th></tr></thead><tbody>{rows}</tbody></table></div></div></details>')
    disc = res.get("disconnection")
    dmod = next((x for x in manifest.get("modules", []) if x["id"] == "disconnection"), None)
    if disc is not None:
        hit = sorted((r for r in disc if r["percent_disconnected"] > 0),
                     key=lambda r: -r["percent_disconnected"])
        rows = "".join(f'<tr><td>{_esc(r["name"])}</td><td class="n">{r["percent_disconnected"]:.1f}%'
                       f'{_bar(r["percent_disconnected"], "dmg")}</td><td class="n">'
                       f'{r["n_disconnected"]:,} / {r["n_streamlines"]:,}</td></tr>' for r in hit[:25])
        details += ('<details class="card" open><summary><span class="t">Streamline disconnection</span>'
                    f'<span class="s">{len(hit)} tracts with disconnected streamlines</span></summary>'
                    '<div class="inner"><div class="tablewrap"><table><thead><tr><th>Tract</th>'
                    '<th class="n">% streamlines cut</th><th class="n">Cut / total</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div><p class="note">Method after Griffis et al., '
                    'NeuroImage: Clinical 2021.</p></div></details>')
    elif dmod and dmod["status"] != "ran":
        details += ('<details class="card"><summary><span class="t">Streamline disconnection</span>'
                    f'<span class="s">not included: {_esc(dmod.get("reason") or dmod["status"])}</span>'
                    '</summary><div class="inner"><p class="note">Build a disconnection index with '
                    '<code>tools/build_discon_index.py</code> to add this section.</p></div></details>')

    dash = (f'<div class="dash">{hero}<div class="kpis">{kpi_html}</div>{top}{quality}{where}'
            f'<div class="section-title">Details</div>{details}'
            f'{_extras_card(res, manifest)}{_about(manifest)}</div>')

    # ---------------- clinical ----------------
    clin = _clinical_single(res, manifest, m, facts, flags, ts, name, src)
    return _page(f"LQTpy · {name}", dash, clin, subtitle="Lesion report")


def _clinical_single(res, manifest, m, facts, flags, ts, name, src) -> str:
    sec = [0]

    def h2(title, extra=""):
        sec[0] += 1
        return f'<h2><span class="no">{sec[0]}</span>{_esc(title)}{extra}</h2>'

    meta = [("Lesion", name), ("Source file", src or "in-memory image"), ("Generated", ts),
            ("Scope", "Core + extras" if manifest.get("scope") != "core" else "Core only"),
            ("Software", f"lqtpy {manifest.get('lqtpy_version', '')}")]
    prof = (manifest.get("settings") or {}).get("profile")
    if prof:
        meta.append(("Description profile", prof))
    out = ['<article class="doc">', "<h1>Lesion quantification report</h1>",
           '<div class="sub">Automated analysis of a lesion mask in MNI space</div>',
           '<table class="meta"><tbody>' + "".join(
               f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in meta) + "</tbody></table>"]
    out.append(h2("Findings"))
    for h, p in _desc_sections(res):
        out.append(f"<p>{_emph(p)}</p>")
    if not _desc_sections(res):
        out.append("<p>No description was generated.</p>")
    out.append(h2("Mask quality"))
    warn = [f for f in flags if f["severity"] != "info"]
    if warn:
        out += [f'<p class="flagline {f["severity"]}">{_esc(f["message"])}</p>' for f in warn]
    else:
        out.append('<p class="ok">No quality problems found.</p>')
    notes = [f for f in flags if f["severity"] == "info"]
    if notes:
        out.append('<p class="fine">' + " ".join(_esc(f["message"]) for f in notes) + "</p>")
    out.append(h2("Measurements"))
    rows = [("Volume", f'{m["volume_cc"]} cm³ ({m["n_voxels"]:,} voxels of {m["voxel_volume_mm3"]} mm³)'),
            ("Separate parts", facts.get("n_clusters", m["n_components"])),
            ("Hemispheres", f'{m["pct_left"]:.0f}% left, {m["pct_right"]:.0f}% right'
             + (f', {m["pct_midline"]:.0f}% midline' if m.get("pct_midline") else "")),
            ("Laterality index", f'{m["laterality_index"]:+.2f}'),
            ("Centre of mass (MNI)", ", ".join(f"{c:.0f}" for c in m["center_of_mass_mm"]) + " mm"),
            ("Extent", " × ".join(f"{b:.0f}" for b in m["bbox_size_mm"]) + " mm")]
    out.append('<table><tbody>' + "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                                          for k, v in rows) + "</tbody></table>")
    comp = _composition(facts)
    dmg = _damaged(facts, limit=12)
    terrs = _territories(facts)
    if comp or dmg:
        out.append(h2("Anatomical involvement"))
        if comp:
            out.append('<table><thead><tr><th>Compartment</th><th class="n">% of lesion</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(n)}</td><td class="n">{v:.1f}%</td></tr>' for n, v in comp[:10])
                       + "</tbody></table>")
        if dmg:
            out.append('<p class="fine" style="margin-top:3mm">Most damaged structures</p>'
                       '<table><thead><tr><th>Structure</th><th>Functional label</th><th class="n">% damaged</th>'
                       '<th class="n">% of lesion</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(r["name"])}</td><td>{_esc(r.get("function") or "")}</td>'
                                 f'<td class="n">{r["dmg"]:.1f}%</td><td class="n">{r["share"]:.1f}%</td></tr>'
                                 for r in dmg) + "</tbody></table>")
        if terrs:
            out.append('<p class="fine" style="margin-top:3mm">Vascular territories (Liu et al. 2023)</p>'
                       '<table><thead><tr><th>Territory</th><th class="n">% of lesion</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(n)}{" (deep perforator)" if deep else ""}</td>'
                                 f'<td class="n">{v:.1f}%</td></tr>' for n, v, deep in terrs)
                       + "</tbody></table>")
    figs = []
    k = 0
    if res.get("glass_uri"):
        k += 1
        figs.append(f'<figure><div class="plate">{_borrow("img-glass", "Glass-brain projections")}</div>'
                    f"<figcaption>Figure {k}. Glass-brain projections of the lesion.</figcaption></figure>")
    if res.get("slices_uri"):
        k += 1
        figs.append(f'<figure><div class="plate">{_borrow("img-slices", "Orthogonal slices")}</div>'
                    f"<figcaption>Figure {k}. Orthogonal slices through the lesion's centre of mass."
                    "</figcaption></figure>")
    for i, a in enumerate(res["atlases"]):
        if a.get("flat_uris") and a["key"] == "dk":
            k += 1
            figs.append(f'<figure>{_borrow(f"a{i}-flat0", "DKT cortical damage map")}'
                        f"<figcaption>Figure {k}. Percent of each DKT cortical region damaged."
                        "</figcaption></figure>")
    if figs:
        out.append(h2("Figures"))
        out += figs
    measures = res.get("measures") or {}
    if measures:
        out.append(h2("Extra measures", '<span class="extra-tag">EXTRA</span>'))
        for nm, mm in measures.items():
            out.append(f'<p><strong>{_esc(nm.capitalize())}:</strong> '
                       f'{_esc("; ".join(mm.get("descriptors") or []))}</p>')
    out.append(h2("Scope and methods"))
    ran = [x["label"] for x in manifest.get("modules", []) if x["status"] == "ran"]
    notrun = [x for x in manifest.get("modules", []) if x["status"] != "ran"]
    out.append(f'<p class="fine"><strong>Included:</strong> {_esc(", ".join(ran))}.</p>')
    if notrun:
        out.append('<p class="fine"><strong>Not included:</strong> ' + "; ".join(
            f'{_esc(x["label"])} ({_esc(x.get("reason") or x["status"])})' for x in notrun) + ".</p>")
    out.append('<p class="fine">Voxels above the threshold count as lesion. Physical measures use the '
               'image affine; region overlap resamples the mask onto each atlas grid (nearest '
               'neighbour). Atlases: AAL (Tzourio-Mazoyer 2002), Schaefer-100 (Schaefer 2018), DKT31 '
               '(Klein &amp; Tourville 2012), JHU ICBM-DTI-81 (Mori 2008), arterial territories '
               '(Liu 2023). Functional labels describe location only, not symptoms. This report is '
               'generated automatically and is not a clinical diagnosis.</p>')
    out.append("</article>")
    return "".join(out)


# =========================================================================== #
# Group
# =========================================================================== #
def _group_rows(recs, key, limit=10, name_fn=None, skip=()) -> str:
    """Cohort-mean rows; bars scaled to the largest value in the list so
    small group means stay readable (the printed number is absolute)."""
    rows = [r for r in _records(recs) if r[key] not in skip][:limit]
    col = "CohortMeanPercentDamage" if rows and "CohortMeanPercentDamage" in rows[0] else "CohortMeanPercentOverlap"
    top = max((r[col] for r in rows), default=0) or 1.0
    out = []
    for r in rows:
        nm = name_fn(r[key]) if name_fn else r[key]
        out.append(_row(nm, r[col], "dmg",
                        f'<span class="chip">{r["NSubjects"]}/{r["NSubjectsTotal"]}</span>',
                        max_value=top, digits=1))
    return "".join(out)


def _group_table(recs, key, label, limit=30) -> str:
    recs = _records(recs)
    if not recs:
        return '<p class="note">No overlap in this group.</p>'
    suffix = "Overlap" if "CohortMeanPercentOverlap" in recs[0] else "Damage"
    body = "".join(
        f'<tr><td>{_esc(r[key])}</td><td class="n">{r["NSubjects"]}/{r["NSubjectsTotal"]}</td>'
        f'<td class="n">{r[f"CohortMeanPercent{suffix}"]:.1f}%{_bar(r[f"CohortMeanPercent{suffix}"], "dmg")}</td>'
        f'<td class="n">{r[f"MeanPercent{suffix}Affected"]:.1f}%</td><td class="n">{r[f"MaxPercent{suffix}"]:.1f}%</td></tr>'
        for r in recs[:limit])
    more = f'<p class="note">Showing {limit} of {len(recs)}; all rows are in the CSV.</p>' if len(recs) > limit else ""
    return ('<div class="tablewrap"><table><thead><tr>'
            f'<th>{_esc(label)}</th><th class="n">Subjects</th><th class="n">Mean, all subjects</th>'
            '<th class="n">Mean, affected</th><th class="n">Max</th></tr></thead>'
            f"<tbody>{body}</tbody></table></div>{more}")


def render_group(res: dict, style: "ReportStyle | None" = None) -> str:
    """The group report (dashboard + clinical views in one file)."""
    manifest = res.get("manifest") or {}
    n = res["n_subjects"]
    subjects = res["subjects"]
    ts = res.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    vols = sorted(s["morphometry"]["volume_cc"] for s in subjects)
    by = {a["key"]: a for a in res["atlases"]}

    hero = (f'<div class="hero"><div><h1>Group report</h1><div class="meta">{n} subjects · '
            f'{_esc(ts)} · analysed in {_esc(res.get("elapsed", "?"))}</div></div>'
            f'<span class="spacer"></span>{_scope_pill(manifest)}<span class="pill">Group</span></div>')
    q = res.get("n_subjects_with_warnings", 0)
    kpis = [("Subjects", str(n), ""),
            ("Median volume", f'{res["median_volume_cc"]} cm³', f"range {vols[0]}–{vols[-1]} cm³"),
            ("Max overlap", f'{res["max_overlap"]} of {n}', "subjects at the most shared voxel"),
            ("Bilateral", f'{res.get("n_bilateral", 0)} of {n}', "lesions in both hemispheres"),
            ("Quality", f'<span class="status {"warn" if q else "ok"}"><span class="dot"></span>'
                        f'{q} of {n} to check</span>', "mask quality checks")]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{_esc(k)}</div><div class="v">'
                       f'{v if k == "Quality" else _esc(v)}</div><div class="s">{_esc(s)}</div></div>'
                       for k, v, s in kpis)
    overlap = ""
    if res.get("overlap_uri"):
        overlap = (f'<section class="card"><h2>Lesion overlap</h2><div class="plate">'
                   f'{_img(res["overlap_uri"], "Lesion overlap map", "img-overlap")}</div>'
                   '<figcaption>Number of subjects with a lesion at each voxel (axial slices).'
                   '</figcaption></section>')
    subj_rows = []
    for s in subjects:
        mm = s["morphometry"]
        nm = _esc(s["name"])
        if s.get("report_path"):
            nm = f'<a href="{_esc(s["report_path"].replace(chr(92), "/"))}">{nm}</a>'
        qw = _qc_worst(s.get("qc"))
        qlabel = {"ok": "ok", "warn": "check", "error": "problem"}[qw]
        ids = ", ".join(sorted({f["id"].replace("_", " ") for f in s.get("qc", [])
                                if f["severity"] != "info"}))
        mini = (f'<span class="mini" aria-label="{mm["pct_left"]:.0f}% left"><span class="l" '
                f'style="width:{mm["pct_left"]}%"></span><span class="r" style="width:{mm["pct_right"]}%">'
                "</span></span>")
        subj_rows.append(
            f'<tr><td>{nm}</td><td class="n">{mm["volume_cc"]}</td><td class="n">{mm["n_components"]}</td>'
            f'<td>{mini}</td><td>{_esc(s.get("territory") or "—")}</td>'
            f'<td><span class="qc-badge {qw}">{qlabel}</span> <span class="muted">{_esc(ids)}</span></td>'
            f'<td class="muted">{_emph(s.get("summary", ""))}</td></tr>')
    subj = ('<section class="card"><h2>Subjects</h2><div class="tablewrap"><table><thead><tr>'
            '<th>Subject</th><th class="n">cm³</th><th class="n">Parts</th><th>L / R</th>'
            '<th>Main territory</th><th>Quality</th><th>Summary</th></tr></thead><tbody>'
            + "".join(subj_rows) + "</tbody></table></div></section>")
    cards = []
    if by.get("dk", {}).get("lobe_df"):
        cards.append(f'<section class="card"><h2>Lobes (DKT)</h2><div class="rows">'
                     f'{_group_rows(by["dk"]["lobe_df"], "Lobe")}</div></section>')
    if by.get("arterial", {}).get("network_df"):
        cards.append(f'<section class="card"><h2>Vascular territories</h2><div class="rows">'
                     f'{_group_rows(by["arterial"]["network_df"], "Network", name_fn=lambda k: _TERR.get(k, k), skip=("Ventricle",))}'
                     "</div></section>")
    if by.get("schaefer", {}).get("network_df"):
        cards.append(f'<section class="card"><h2>Yeo-7 networks</h2><div class="rows">'
                     f'{_group_rows(by["schaefer"]["network_df"], "Network")}</div></section>')
    where = ""
    if cards:
        where = ('<div class="section-title">Where the lesions are</div>'
                 '<div class="legend"><span><i style="background:var(--lesion)"></i>mean percent damaged, '
                 'all subjects (bars scaled to the largest value in each list)</span>'
                 '<span>chip: subjects affected</span></div>'
                 f'<div class="cols3">{"".join(cards)}</div>')
    details = []
    for i, a in enumerate(res["atlases"]):
        tier = '<span class="tier">EXTRA</span>' if a.get("tier") == "extra" else ""
        extra = ""
        for key, label in (("lobe_df", "Lobe"), ("network_df", "Network")):
            if a.get(key):
                extra += f'<h3 class="section-title">{label}</h3>' + _group_table(a[key], label, label)
        details.append(f'<details class="card"><summary><span class="t">{_esc(a["name"])}{tier}</span>'
                       f'<span class="s">{len(_records(a["df"]))} regions involved</span></summary>'
                       f'<div class="inner">{_flatmaps(a, f"g{i}")}{extra}'
                       f'<h3 class="section-title">Regions</h3>{_group_table(a["df"], "RegionName", "Region")}'
                       "</div></details>")
    if res.get("tracts"):
        details.append('<details class="card"><summary><span class="t">White-matter tracts (voxel-overlap proxy)'
                       '</span><span class="s">not streamline disconnection</span></summary><div class="inner">'
                       + _group_table(res["tracts"]["df"], "Tract", "Tract") + "</div></details>")
    dash = (f'<div class="dash">{hero}<div class="kpis">{kpi_html}</div>'
            f'{subj}{overlap}{where}'
            f'<div class="section-title">Details</div>{"".join(details)}{_about(manifest)}</div>')

    # clinical
    sec = [0]

    def h2(title):
        sec[0] += 1
        return f'<h2><span class="no">{sec[0]}</span>{_esc(title)}</h2>'

    c = ['<article class="doc"><h1>Group lesion report</h1>',
         f'<div class="sub">{n} subjects · generated {_esc(ts)} · lqtpy '
         f'{_esc(manifest.get("lqtpy_version", ""))}</div>', h2("Cohort")]
    c.append('<table><tbody>' + "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in [
        ("Subjects", n), ("Median volume", f'{res["median_volume_cc"]} cm³ (range {vols[0]}–{vols[-1]})'),
        ("Bilateral lesions", f'{res.get("n_bilateral", 0)} of {n}'),
        ("Masks with quality warnings", f"{q} of {n}")]) + "</tbody></table>")
    c.append(h2("Subjects"))
    c.append('<table><thead><tr><th>Subject</th><th class="n">cm³</th><th class="n">L/R %</th>'
             '<th>Main territory</th><th>Quality</th></tr></thead><tbody>' + "".join(
                 f'<tr><td>{_esc(s["name"])}</td><td class="n">{s["morphometry"]["volume_cc"]}</td>'
                 f'<td class="n">{s["morphometry"]["pct_left"]:.0f}/{s["morphometry"]["pct_right"]:.0f}</td>'
                 f'<td>{_esc(s.get("territory") or "—")}</td><td>{_esc(_qc_worst(s.get("qc")))}</td></tr>'
                 for s in subjects) + "</tbody></table>")
    c.append(h2("Involvement"))
    for key, df_key, label in (("dk", "lobe_df", "Lobe"), ("arterial", "network_df", "Network"),
                               ("dk", "df", "RegionName")):
        a = by.get(key)
        if a and a.get(df_key):
            title = {"lobe_df": "Lobes (DKT)", "network_df": "Vascular territories",
                     "df": "Regions (DKT)"}[df_key]
            c.append(f'<p class="fine" style="margin-top:3mm">{title}</p>'
                     + _group_table(a[df_key], label, label.replace("RegionName", "Region"), limit=15))
    if res.get("overlap_uri"):
        c.append(h2("Figure"))
        c.append(f'<figure><div class="plate">{_borrow("img-overlap", "Lesion overlap map")}</div>'
                 "<figcaption>Figure 1. Number of subjects with a lesion at each voxel.</figcaption></figure>")
    c.append(h2("Scope and methods"))
    c.append('<p class="fine">"Mean, all subjects" averages over every subject (unaffected subjects count '
             'as 0); "Mean, affected" averages over affected subjects only. Each subject\'s full report '
             'is linked from the dashboard view. This report is generated automatically and is not a '
             'clinical diagnosis.</p></article>')
    return _page("LQTpy · group report", dash, "".join(c), subtitle="Group report")
