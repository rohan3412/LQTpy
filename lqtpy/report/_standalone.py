"""LQTpy's self-contained HTML reports: a dashboard and a clinical report in
one file.

``render_single`` (one lesion) and ``render_group`` (a cohort) turn an
``analyze_single``/``analyze_group`` result into a single HTML file with
every figure embedded. The file holds two views, switched from its toolbar:

- **Dashboard** -- dark by default (light on request), summary first: key
  numbers, the written description (with a copy button and glossary
  popovers), quality flags, figures and a scroll-through slice viewer, then
  where the lesion is. Atlas tables, extras, downloads and the run record
  sit below, collapsed.
- **Clinical report** -- a paper-like document (serif text, numbered
  sections, figure captions). This is also what "Export PDF" prints, and
  what :func:`lqtpy.report.to_pdf` produces.

Colour is informative and used the same way everywhere:

=================  ========================================================
lesion orange      how much of a structure is damaged (bars, maps)
neutral ink        what share of the lesion lies somewhere
blue / teal        left / right hemisphere (checked for colour-vision deficiency)
violet             extras (anything not core)
status colours     QC only, always with a word (OK / Note / Check / Problem)
=================  ========================================================

Tables sort by any column (remembered per table), and hovering a table row
highlights the same region on the maps, and back. Every data value is
escaped: region labels can come from host-registered atlases and are
treated as untrusted.
"""
from __future__ import annotations

import base64
import datetime
import html
import json
import os
import re

from .. import _format as fmt
from ._assets import CSS, GLOSSARY, JS
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
    return os.path.basename(str(path).replace("\\", "/")) if path else ""


def _pct(v, digits=0) -> str:
    v = float(v)
    if 0 < v < 1 and digits == 0:
        return "<1%"
    return f"{v:.{digits}f}%"


def _json_script(id_, obj) -> str:
    text = json.dumps(obj, default=str).replace("</", "<\\/")
    return f'<script type="application/json" id="{_esc(id_)}">{text}</script>'


# =========================================================================== #
# Page
# =========================================================================== #
def _page(title: str, dash: str, clin: str, *, subtitle: str = "", tail: str = "") -> str:
    return (
        '<!doctype html><html lang="en" data-theme="dark" data-view="dashboard"><head>'
        '<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<meta name="generator" content="lqtpy"><title>{_esc(title)}</title><style>{CSS}</style></head><body>'
        '<a class="skip" href="#main">Skip to the report</a>'
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
        f'<main id="main" class="view view-dash" aria-label="Dashboard">{dash}</main>'
        f'<main class="view view-clin" aria-label="Clinical report">{clin}</main>'
        f"{tail}<script>{JS}</script></body></html>"
    )


# =========================================================================== #
# Shared pieces
# =========================================================================== #
_KIND = {"single": "Single lesion", "group": "Group", "group-member": "Single lesion (from a group)"}
_SHORT = {"aal": "AAL", "schaefer": "Schaefer", "dk": "DKT", "jhu": "JHU", "arterial": "Arterial"}
_TERR_LEGEND = ("ACA, MCA, PCA: anterior, middle and posterior cerebral artery; VB: vertebrobasilar; "
                "L/R: left/right. Territories from Liu et al. 2023.")


def _bar(value, cls="", max_value=100.0) -> str:
    w = 0.0 if not max_value else max(0.0, min(100.0, 100.0 * float(value) / float(max_value)))
    return f'<span class="track {cls}" aria-hidden="true"><i style="width:{w:.1f}%"></i></span>'


def _row(name, value, cls, extra="", max_value=100.0, digits=0, title=None, hl=None) -> str:
    h = f' data-hl="{_esc(hl)}"' if hl else ""
    return (f'<div class="row"{h}><span class="name" title="{_esc(title or name)}">{_esc(name)}{extra}</span>'
            f'{_bar(value, cls, max_value)}<span class="num">{_pct(value, digits)}</span></div>')


_GL_RE = None


def _gloss(markup: str) -> str:
    """Wrap the first occurrence of each glossary term (outside tags) in a
    focusable popover."""
    global _GL_RE
    if _GL_RE is None:
        terms = sorted(GLOSSARY, key=len, reverse=True)
        _GL_RE = re.compile(r"(?<![\w-])(" + "|".join(re.escape(t) for t in terms) + r")(?![\w-])",
                            re.IGNORECASE)
    seen = set()

    def sub(m):
        word = m.group(1)
        k = next(t for t in GLOSSARY if t.lower() == word.lower())
        if k in seen:
            return word
        seen.add(k)
        return f'<span class="gl" tabindex="0" data-def="{_esc(GLOSSARY[k])}">{word}</span>'

    parts = re.split(r"(<[^>]+>)", markup)
    return "".join(p if p.startswith("<") else _GL_RE.sub(sub, p) for p in parts)


def _emph(text: str, gloss: bool = False) -> str:
    """Escape, turning **...** spans (key findings) into <strong>."""
    parts = str(text).split("**")
    out = "".join(_esc(p) if i % 2 == 0 else f"<strong>{_esc(p)}</strong>"
                  for i, p in enumerate(parts))
    return _gloss(out) if gloss else out


def _desc_sections(res) -> "list[tuple[str, str]]":
    d = res.get("description") or {}
    return [(h, p) for h, p in (d.get("marked") or d.get("sections") or []) if h != "Quality"]


def _qc_worst(flags) -> str:
    sev = {f["severity"] for f in flags or []}
    return "error" if "error" in sev else "warn" if "warn" in sev else "ok"


def _flag_label(f) -> str:
    """'Check, moderate' / 'Problem, major' / 'Note'."""
    word = fmt.quality_label(f["severity"])
    return f'{word}, {f["grade"]}' if f.get("grade") else word


def _flag_where(f) -> str:
    """The first problem location in words ('coronal slice 81', 'at -40, 10, 5 mm')."""
    locs = f.get("locations") or []
    if not locs:
        return ""
    loc = locs[0]
    if loc.get("kind") == "slice":
        return f'{loc["plane"]} slice {loc["index"]}'
    return "at " + ", ".join(f"{v:.0f}" for v in loc["mm"]) + " mm"


def _flags_html(flags, verdict=None, score=None) -> str:
    order = {"error": 0, "warn": 1, "info": 2}
    items = sorted(flags or [], key=lambda f: order[f["severity"]])
    out = []
    if verdict and any(f["severity"] != "info" for f in items):
        out.append(f'<p class="verdict"><b>{_esc(verdict.capitalize())}</b>'
                   + (f' <span class="muted">· quality score {score}/100</span>' if score is not None else "")
                   + "</p>")
    for f in items:
        icon = {"error": "×", "warn": "!", "info": "i"}[f["severity"]]
        where = _flag_where(f)
        sug = " · ".join(_esc(x) for x in (f"Where: {where}" if where else "", f.get("suggestion") or "") if x)
        sug = f'<div class="sug">{sug}</div>' if sug else ""
        out.append(f'<div class="flag {f["severity"]}"><span class="ic" aria-hidden="true">{icon}</span>'
                   f'<div><b>{_esc(_flag_label(f))}</b>{_esc(f["message"])}{sug}</div></div>')
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


def _run_block(run: "dict | None", report_name: str) -> str:
    """The reproducibility record: versions, settings, input checksums and
    the command that repeats the analysis."""
    if not run:
        return ""
    env = run.get("environment", {})
    sets = run.get("settings", {})
    kv = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in env.items())
    sv = "".join(f"<dt>{_esc(k)}</dt><dd>{_esc(v)}</dd>" for k, v in sets.items()
                 if v not in (None, "", False) or k in ("tracts", "qc", "describe"))
    inputs = run.get("inputs", [])
    shown = inputs[:12]
    ins = "".join(f"<tr><td>{_esc(i.get('name'))}</td><td class='mono'>{_esc((i.get('sha256') or '')[:16])}…</td>"
                  f"<td>{_esc(_basename(i.get('path')) or '—')}</td></tr>" for i in shown)
    more = (f'<p class="note">{len(inputs) - len(shown)} more inputs in the run record.</p>'
            if len(inputs) > 12 else "")
    cmd = f"lqtpy rerun {report_name}"
    return (f'<h3 class="section-title">Reproduce this report</h3>'
            f'<div class="cmd"><code id="rerun-cmd">{_esc(cmd)}</code>'
            '<button type="button" class="btn small" data-copy="rerun-cmd">Copy</button></div>'
            '<p class="note">Runs the same analysis with the same settings. Inputs are found by their '
            'recorded path, or by checksum with <code>--search FOLDER</code>; a changed file is refused.</p>'
            f'<dl class="kv">{kv}{sv}</dl>'
            '<div class="tablewrap"><table data-table="run-inputs"><thead><tr><th>Input</th>'
            f'<th>SHA-256</th><th>File</th></tr></thead><tbody>{ins}</tbody></table></div>{more}')


def _about(manifest, run=None, report_name="report.html", extra_rows="") -> str:
    s = manifest.get("settings", {})
    sets = [f"lqtpy {_esc(manifest.get('lqtpy_version', ''))}",
            f"threshold &gt; {_esc(s.get('threshold', 0.5))}"]
    if s.get("profile"):
        sets.append(f"description profile: {_esc(s['profile'])}")
    return (f'<details class="card" id="about"><summary><span class="t">About this report</span>'
            f'<span class="s">{" · ".join(sets)}</span></summary><div class="inner">'
            f"{_modules(manifest)}{extra_rows}"
            '<p class="note">Physical measures use the image affine. Region overlap resamples the '
            'mask onto each atlas grid (nearest neighbour). Functional area labels describe '
            'location, not symptoms. Border zones and distances are template-based and '
            'approximate. Not for clinical decision-making on its own.</p>'
            f"{_run_block(run, report_name)}</div></details>")


def _downloads(downloads: "dict | None") -> "tuple[str, str]":
    """(card html, hidden data scripts) for the report's own tables."""
    if not downloads:
        return "", ""
    btns, data = [], []
    for i, (fname, path) in enumerate(sorted(downloads.items())):
        try:
            if os.path.getsize(path) > 5_000_000:
                continue
            with open(path, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode("ascii")
        except OSError:
            continue
        btns.append(f'<button type="button" class="btn small" data-dl="dl{i}" '
                    f'data-name="{_esc(fname)}">{_esc(fname)}</button>')
        data.append(f'<script type="text/plain" id="dl{i}">{b64}</script>')
    if not btns:
        return "", ""
    card = ('<details class="card"><summary><span class="t">Download tables</span>'
            f'<span class="s">{len(btns)} files, stored inside this report</span></summary>'
            '<div class="inner"><p class="note">These are the CSV files written next to the report; '
            'the report alone is enough to share them. <code>data_dictionary.csv</code> explains every '
            f'column and unit.</p><div class="dl">{"".join(btns)}</div></div></details>')
    return card, "".join(data)


def _plate(svg, id_) -> str:
    return f'<div class="plate" id="{_esc(id_)}">{svg}</div>' if svg else ""


def _flatmaps(a) -> str:
    out = []
    for cap, svg in a.get("flat_svgs") or []:
        out.append(f'<figure class="flatmap">{svg}<figcaption>{_esc(cap)} · hover a region, '
                   'or a table row, to find it</figcaption></figure>')
    return "".join(out)


def _viewer(res) -> "tuple[str, str]":
    """(card html, scripts) for the brainsprite slice viewer."""
    v = res.get("viewer")
    if not v:
        return "", ""
    from ..figures import brainsprite_js
    js = brainsprite_js()
    if not js:
        return "", ""
    card = ('<details class="card"><summary><span class="t">Slice viewer</span>'
            '<span class="s">scroll through the brain: click a view to move the cursor (2 mm grid)'
            '</span></summary><div class="inner"><div class="viewer" id="viewer">'
            '<canvas id="viewer-canvas" aria-label="Interactive slice viewer"></canvas>'
            f'<img id="viewer-bg" src="{v["bg"]}" alt=""><img id="viewer-ov" src="{v["overlay"]}" alt="">'
            '</div><p class="note">Lesion in orange on a tissue template. Uses brainsprite (MIT).</p>'
            '</div></details>')
    return card, _json_script("viewer-params", v["params"]) + f"<script>{js}</script>"


# =========================================================================== #
# Single lesion
# =========================================================================== #
_UNNAMED = {"white_matter": "unnamed white matter", "cortex": "cortex outside named regions",
            "deep_grey": "deep grey matter outside named regions", "csf": "CSF spaces",
            "ventricles": "ventricles", "cerebellum": "cerebellum outside named regions",
            "brainstem": "brainstem outside named regions", "outside": "outside the brain"}


def _composition(facts) -> "list[tuple[str, float]]":
    """Share of the lesion per compartment: lobes per side, named white
    matter, and the rest split by tissue (the same split the description
    uses), largest first."""
    rows = []
    by_side = facts.get("unlabelled_tissue_by_side") or {}
    for h in facts.get("hemispheres", []):
        side = h["hemisphere"] if h["hemisphere"] in ("left", "right") else "midline"
        pre = "" if side == "midline" else f"{side} "
        for lb in h["lobes"]:
            rows.append((f"{pre}{fmt.lobe_word(lb['lobe'], short=True)}", lb["pct_of_lesion"]))
        if h.get("wm_pct"):
            rows.append((f"{pre}white matter (named tracts)", h["wm_pct"]))
        if h.get("unlabelled_pct") and not by_side:
            rows.append((f"{pre}unnamed tissue", h["unlabelled_pct"]))
    outside = 0.0
    for side, comp in by_side.items():
        pre = "" if side == "midline" else f"{side} "
        for k, v in comp.items():
            if k == "outside":
                outside += v
            else:
                rows.append((f"{pre}{_UNNAMED.get(k, k)}", v))
    if outside:
        rows.append(("outside the brain", round(outside, 1)))
    return sorted([r for r in rows if r[1] >= 0.5], key=lambda r: -r[1])


def _main_structure(facts) -> "tuple[str, float] | None":
    comp = _composition(facts)
    return comp[0] if comp else None


_TISSUE_SHORT = {"cortex": "cortex", "white_matter": "WM", "deep_grey": fmt.DEEP_GREY_SHORT,
                 "cerebellum": "cerebellum", "brainstem": "brainstem", "ventricles": "ventricles",
                 "csf": "CSF", "outside": "outside"}
_LOC_SHORT = {"periventricular": "periventricular", "juxtacortical": "juxtacortical",
              "cortical": "cortical", "deep white matter": "deep WM",
              "deep grey matter": fmt.DEEP_GREY_SHORT, "infratentorial": "infratentorial",
              "outside the brain": "outside"}


def _tissue_kpi(t: dict, n_parts: int) -> "tuple[str, str]":
    comp = t.get("composition") or {}
    counts = {k: v for k, v in (t.get("counts") or {}).items() if v}
    if n_parts > 3 and counts:
        order = sorted(counts.items(), key=lambda kv: -kv[1])
        head = f"{order[0][1]} {_LOC_SHORT[order[0][0]]}"
        rest = " · ".join(f"{v} {_LOC_SHORT[k]}" for k, v in order[1:3])
        return head, f"of {n_parts}" + (f" · {rest}" if rest else "")
    top = sorted(((k, v) for k, v in comp.items() if v >= 5 and k != "outside"), key=lambda kv: -kv[1])
    return (t.get("class") or "—"), " · ".join(f"{v:.0f}% {_TISSUE_SHORT[k]}" for k, v in top[:3])


def _tissue_text(t: dict) -> str:
    comp = t.get("composition") or {}
    items = sorted(((k, v) for k, v in comp.items() if v >= 0.5), key=lambda kv: -kv[1])
    return ", ".join(f"{_TISSUE_SHORT[k]} {v:.0f}%" for k, v in items)


def _damaged(facts, limit=8) -> "list[dict]":
    rows = []
    atlas = facts.get("atlas", "dk")
    for h in facts.get("hemispheres", []):
        side = h["hemisphere"] if h["hemisphere"] in ("left", "right") else ""
        for lb in h["lobes"]:
            for r in lb["regions"]:
                rows.append({"name": f"{side} {r['name']}".strip(), "dmg": r["pct_damaged"],
                             "share": r["pct_of_lesion"], "function": r.get("function"),
                             "hl": f"{atlas}|{r['region']}" if r.get("region") else None})
        for w in h.get("wm", []):
            rows.append({"name": f"{side} {w['name']}".strip(), "dmg": w["pct_damaged"],
                         "share": w["pct_of_lesion"], "function": None, "wm": True, "hl": None})
    rows = [r for r in rows if r["share"] >= 0.5 or r["dmg"] >= 5]
    return sorted(rows, key=lambda r: -r["dmg"])[:limit]


def _territories(facts, short: bool = True) -> "list[tuple[str, float, bool]]":
    """Territory parts, short form ('L MCA · frontal part', 'L lat.
    lenticulostriate', 'bilateral VB · basilar artery')."""
    rows = []
    for t in facts.get("territories", []):
        s1 = fmt.side_short(t["hemisphere"])
        s1 = f"{s1} " if s1 else ""
        for pp in t["parts"]:
            label = pp["name"].replace(" territory", "")
            if ", " in label:
                label = f"{s1}{t['territory']} · {label.split(', ', 1)[1]}"
            elif pp["deep"]:
                if short:
                    label = label.replace("lateral ", "lat. ").replace("medial ", "med. ")
                label = f"{s1}{label}"
            else:
                label = f"{s1}{t['territory']}"
            rows.append((label, pp["pct_of_lesion"], pp["deep"]))
    return sorted([r for r in rows if r[1] >= 1], key=lambda r: -r[1])[:8]


def _atlas_details(a) -> str:
    recs = _records(a["df"])
    cov = a.get("coverage") or {}
    key = a["key"]
    head_s = (f"{len(recs)} regions involved · {cov.get('PercentLabelled', 0):.0f}% of the lesion "
              f"is in {_SHORT.get(key, key)} regions") if cov else f"{len(recs)} regions"
    tier = '<span class="tier">EXTRA</span>' if a.get("tier") == "extra" else ""
    rows = "".join(
        f'<tr data-hl="{_esc(key)}|{_esc(r["RegionName"])}" tabindex="0"><td>{_esc(r["RegionName"])}</td>'
        f'<td class="n">{r["PercentDamage"]:.1f}%{_bar(r["PercentDamage"], "dmg")}</td>'
        f'<td class="n">{r["PercentOfLesion"]:.1f}%{_bar(r["PercentOfLesion"])}</td>'
        f'<td class="n">{r["LesionVoxels"]:,}</td></tr>'
        for r in recs[:40])
    more = (f'<p class="note">Showing 40 of {len(recs)} regions; every region is in the CSV.</p>'
            if len(recs) > 40 else "")
    rollups = ""
    for k, label in (("lobe_df", "Lobe"), ("network_df", "Network")):
        rr = _records(a.get(k))
        if rr:
            col = "Lobe" if k == "lobe_df" else "Network"
            name = {"schaefer": "Yeo-7 network", "arterial": "Territory"}.get(key, label)
            body = "".join(
                f'<tr><td>{_esc(fmt.lobe_word(r[col], short=True, capital=True) if col == "Lobe" else r[col])}</td>'
                f'<td class="n">{r["PercentDamage"]:.1f}%{_bar(r["PercentDamage"], "dmg")}</td>'
                f'<td class="n">{r["PercentOfLesion"]:.1f}%{_bar(r["PercentOfLesion"])}</td></tr>'
                for r in rr)
            rollups += (f'<h3 class="section-title">{_esc(name)}</h3><div class="tablewrap">'
                        f'<table data-table="{_esc(k)}-{_esc(key)}"><thead><tr><th>{_esc(name)}</th>'
                        '<th class="n">% damaged</th><th class="n">% of lesion</th></tr></thead>'
                        f'<tbody>{body}</tbody></table></div>')
    table = ("" if not recs else
             f'<div class="tablewrap"><table data-table="regions-{_esc(key)}"><thead><tr><th>Region</th>'
             '<th class="n">% of region damaged</th><th class="n">% of lesion</th><th class="n">Voxels</th>'
             f'</tr></thead><tbody>{rows}</tbody></table></div>{more}')
    return (f'<details class="card"><summary><span class="t">{_esc(a["name"])}{tier}</span>'
            f'<span class="s">{_esc(head_s)}</span></summary><div class="inner">'
            f'{_flatmaps(a)}{rollups}{table}</div></details>')


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
            vals = {k: v for k, v in (m.get("values") or {}).items()
                    if k != "per_component" and not isinstance(v, (dict, list))}
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
    return f'<section class="card extras" aria-label="Extras"><h2>Extras</h2>{body}{missing}</section>'


def _nearby_card(res) -> str:
    rows = [r for r in res.get("key_structures") or [] if r["DistanceMM"] <= 20][:8]
    if not rows:
        return ""
    items = []
    for r in rows:
        d = "reaches it" if r["Reaches"] else f'{r["DistanceMM"]:.0f} mm'
        items.append(f'<div><span>{_esc(r["Structure"])}</span><span class="num">{d}</span></div>')
    return ('<section class="card" aria-label="Nearby structures"><h2>Nearby structures</h2>'
            '<div class="legend"><span>shortest distance from the lesion (template-based)</span></div>'
            f'<div class="dist">{"".join(items)}</div></section>')


def _mirror_card(res) -> str:
    h = res.get("homologue") or {}
    regs = h.get("regions") or []
    if not regs:
        return ""
    rows = "".join(f'<tr><td>{_esc(r["name"])}</td><td class="n">{r["pct_of_mirror"]:.0f}%</td></tr>'
                   for r in regs)
    return ('<details class="card"><summary><span class="t">Mirror image (homologous region)</span>'
            f'<span class="s">the same place in the {_esc(h.get("side", "other"))} hemisphere · '
            f'{h.get("self_overlap_pct", 0):.0f}% of the lesion overlaps its own mirror</span></summary>'
            '<div class="inner"><p class="note">The lesion flipped across the midline. The homologous '
            'region is used as an intact control region (perfusion, thickness, connectivity), and in '
            'recovery research, where the undamaged homologue of a damaged area can take over part of '
            'its work (for example the right-hemisphere homologue of Broca\'s area in aphasia).</p>'
            '<div class="tablewrap"><table data-table="mirror"><thead><tr><th>Region under the mirror image</th>'
            f'<th class="n">% of the mirror image</th></tr></thead><tbody>{rows}</tbody></table></div>'
            '</div></details>')


def _tracts_details(res) -> str:
    tr = res.get("tracts")
    if not tr:
        return ""
    trs = _records(tr["df"])
    cross = {r["Tract"]: r for r in res.get("cross_section") or []}

    def c(t, k):
        return cross.get(t, {}).get(k, 0)

    order = sorted(trs, key=lambda r: -c(r["Tract"], "MaxCrossSectionLoad"))[:25]
    rows = "".join(
        f'<tr><td>{_esc(r["Tract"])}</td><td class="n">{r["PercentOverlap"]:.1f}%'
        f'{_bar(r["PercentOverlap"], "dmg")}</td>'
        f'<td class="n">{c(r["Tract"], "MaxCrossSectionLoad"):.0f}%'
        f'{_bar(c(r["Tract"], "MaxCrossSectionLoad"), "dmg")}</td>'
        f'<td class="n">{c(r["Tract"], "BottleneckLoad"):.0f}%</td>'
        f'<td class="n">{r["PercentOfLesion"]:.1f}%</td></tr>' for r in order)
    return ('<details class="card"><summary><span class="t">White-matter tracts</span>'
            f'<span class="s">{len(trs)} tracts touched · voxel overlap and cross-section load, '
            'not streamline disconnection</span></summary><div class="inner"><p class="note">'
            '<b>% of tract voxels</b>: a coarse proxy. <b>Max cross-section</b>: the largest share of '
            'any 1 mm cross-section of the tract the lesion covers, a better guide to how much of the '
            'tract is cut. <b>Narrowest</b>: the share of its narrowest cross-section.</p>'
            '<div class="tablewrap"><table data-table="tracts"><thead><tr><th>Tract</th>'
            '<th class="n">% of tract voxels</th><th class="n">Max cross-section</th>'
            f'<th class="n">Narrowest</th><th class="n">% of lesion</th></tr></thead><tbody>{rows}'
            '</tbody></table></div></div></details>')


def render_single(res: dict, style: "ReportStyle | None" = None, *,
                  back_link: "str | None" = None, run: "dict | None" = None,
                  downloads: "dict | None" = None) -> str:
    """The single-lesion report (dashboard + clinical views in one file).

    ``run`` is the run record to embed (default: ``res["manifest"]["run"]``);
    ``downloads`` maps file names to paths of tables to embed for download.
    """
    style = style or DEFAULT_STYLE
    manifest = res.get("manifest") or {}
    run = run or manifest.get("run")
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
    bz = res.get("border_zones") or facts.get("border_zones") or {}

    # ---------------- dashboard ----------------
    back = f'<a class="back" href="{_esc(back_link)}">← Group report</a>' if back_link else ""
    hero = (f'<div class="hero"><div>{back}<h1>{_esc(name)}</h1>'
            f'<div class="meta" title="{_esc(src)}">{_esc(_basename(src))}{" · " if src else ""}{_esc(ts)} · analysed in '
            f'{_esc(res.get("elapsed", "?"))}</div></div><span class="spacer"></span>'
            f'{_scope_pill(manifest)}<span class="pill">{_esc(_KIND.get(manifest.get("kind"), "Report"))}</span></div>')
    side = (f'<div class="side" role="img" aria-label="{m["pct_left"]:.0f}% left, '
            f'{m.get("pct_midline", 0):.0f}% midline, {m["pct_right"]:.0f}% right">'
            f'<span class="l" style="width:{m["pct_left"]}%"></span>'
            f'<span class="m" style="width:{m.get("pct_midline", 0)}%"></span>'
            f'<span class="r" style="width:{m["pct_right"]}%"></span></div>')
    n_flags = f"{n_warn} flag" + ("s" if n_warn != 1 else "")
    qc_sub = "mask quality checks" if worst == "ok" else f"{n_flags} · mask quality"
    if res.get("qc_verdict") and worst != "ok":
        qc_sub = f'{res["qc_verdict"]} · {n_flags}'
    t0 = terrs[0] if terrs else None
    terr_sub = (f'{t0["pct_of_lesion"]:.0f}% of the lesion' if t0 else "no territory data") + (
        " · border zone" if bz.get("fits") else "")
    kpis = [
        ("Size", fmt.volume(m["volume_mm3"]),
         (f"{parts} separate parts" if parts != 1 else "one connected lesion")
         + f' · {m["n_voxels"]:,} voxels'),
        ("Side", f'{m["pct_left"]:.0f}% L · {m["pct_right"]:.0f}% R'
         + (f' · {m["pct_midline"]:.0f}% mid' if m.get("pct_midline", 0) >= 1 else ""), side),
        ("Main territory", fmt.territory_short(t0["hemisphere"], t0["territory"]) if t0 else "—", terr_sub),
        ("Main location", main[0] if main else "—", f"{main[1]:.0f}% of the lesion" if main else ""),
        ("Tissue", *_tissue_kpi(res.get("tissue") or {}, parts)),
        ("Quality", f'<span class="status {worst}"><span class="dot"></span>'
                    f'{_esc(fmt.quality_label(worst))}</span>', qc_sub),
    ]
    kpi_html = "".join(
        f'<div class="kpi"><div class="k">{_esc(k)}</div>'
        f'<div class="v{" text" if k != "Size" else ""}"'
        f'{"" if k == "Quality" else " title=" + chr(34) + _esc(v) + chr(34)}>'
        f'{v if k == "Quality" else _esc(v)}</div><div class="s">'
        f'{s if k == "Side" else _esc(s)}</div></div>' for k, v, s in kpis)
    secs = _desc_sections(res)
    desc = "".join(f'<p><span class="h">{_esc(h)}</span>{_emph(p, gloss=True)}</p>' for h, p in secs)
    plain = "\n\n".join(p.replace("**", "") for _, p in secs)
    copy = ('<button type="button" class="btn small tools" data-copy="desc-plain" '
            'title="Copy the description as plain text">Copy text</button>') if secs else ""
    figs = []
    if res.get("glass_svg"):
        figs.append(f'<figure>{_plate(res["glass_svg"], "fig-glass")}'
                    '<figcaption>Projections through the brain: from the left, the front and above'
                    '</figcaption></figure>')
    if res.get("slices_svg"):
        cut = ", ".join(f"{c:.0f}" for c in m["center_of_mass_mm"])
        figs.append(f'<figure>{_plate(res["slices_svg"], "fig-slices")}'
                    f'<figcaption>Slices through the lesion centre ({cut} mm)</figcaption></figure>')
    top = (f'<div class="cols"><section class="card desc" aria-label="Description"><h2>Description{copy}</h2>'
           f'{desc or "<p>No description.</p>"}<div id="desc-plain" hidden>{_esc(plain)}</div></section>')
    top += (f'<section class="card" aria-label="Figures"><h2>Lesion</h2><div class="figs">'
            f'{"".join(figs) or "<p class=note>No figures (plots were off).</p>"}</div></section></div>')
    quality = (f'<section class="card" aria-label="Quality"><h2>Quality</h2>{_flags_html(flags, res.get("qc_verdict"), res.get("qc_score"))}</section>'
               if flags else "")

    comp = "".join(_row(n, v, "") for n, v in _composition(facts)[:9])
    dmg = "".join(_row(r["name"], r["dmg"], "dmg",
                       (f'<span class="chip">{_esc(r["function"])}</span>' if r.get("function") else "")
                       + ('<small>white matter</small>' if r.get("wm") else ""), hl=r.get("hl"))
                  for r in _damaged(facts))
    terr = "".join(_row(n, v, "", '<span class="chip">deep</span>' if deep else "")
                   for n, v, deep in _territories(facts))
    if bz.get("zones"):
        z = bz["zones"][0]
        lead = "<b>Fits a border-zone pattern.</b> " if bz.get("fits") else ""
        terr += (f'<p class="note" style="margin:8px 0 0">{lead}'
                 f'{bz.get("pct_in_border_zones", 0):.0f}% in border zones (mainly the {_esc(z["label"])}).</p>')
    where = ""
    if comp or dmg or terr:
        where = ('<div class="section-title">Where the lesion is</div><div class="cols3">'
                 f'<section class="card"><h2>Share of the lesion</h2><div class="legend"><span><i style="background:var(--comp)"></i>percent of the lesion</span></div><div class="rows">{comp}</div></section>'
                 f'<section class="card"><h2>Most damaged structures</h2><div class="legend"><span><i style="background:var(--lesion)"></i>percent of the structure damaged</span></div><div class="rows">{dmg}</div></section>'
                 f'<section class="card"><h2>Vascular territories</h2><div class="legend"><span><i style="background:var(--comp)"></i>percent of the lesion</span></div><div class="rows">{terr or "<p class=note>No territory data.</p>"}</div></section>'
                 "</div>")
    nearby = _nearby_card(res)
    viewer_card, viewer_tail = _viewer(res)

    details = "".join(_atlas_details(a) for a in res["atlases"])
    details += _tracts_details(res)
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
                    '<div class="inner"><div class="tablewrap"><table data-table="disconnection"><thead><tr><th>Tract</th>'
                    '<th class="n">% streamlines cut</th><th class="n">Cut / total</th></tr></thead>'
                    f'<tbody>{rows}</tbody></table></div><p class="note">Method after Griffis et al., '
                    'NeuroImage: Clinical 2021.</p></div></details>')
    elif dmod and dmod["status"] != "ran":
        details += ('<details class="card"><summary><span class="t">Streamline disconnection</span>'
                    f'<span class="s">not included: {_esc(dmod.get("reason") or dmod["status"])}</span>'
                    '</summary><div class="inner"><p class="note">Build a disconnection index with '
                    '<code>tools/build_discon_index.py</code> to add this section.</p></div></details>')
    details += _mirror_card(res)
    dl_card, dl_data = _downloads(downloads)
    report_name = f"{name}_report.html"

    dash = (f'<div class="dash">{hero}<div class="kpis">{kpi_html}</div>{top}{quality}{where}'
            f'{nearby}{viewer_card}'
            f'<div class="section-title">Details</div>{details}'
            f'{_extras_card(res, manifest)}{dl_card}{_about(manifest, run, report_name)}</div>')

    clin = _clinical_single(res, manifest, m, facts, flags, ts, name, src)
    tail = (_json_script("lqtpy-run", run) if run else "") + dl_data + viewer_tail
    return _page(f"LQTpy · {name}", dash, clin, subtitle="Lesion report", tail=tail)


def _clinical_single(res, manifest, m, facts, flags, ts, name, src) -> str:
    sec = [0]

    def h2(title, extra=""):
        sec[0] += 1
        return f'<h2><span class="no">{sec[0]}</span>{_esc(title)}{extra}</h2>'

    meta = [("Lesion", name), ("Source file", src or "not recorded"), ("Generated", ts),
            ("Scope", "Core + extras" if manifest.get("scope") != "core" else "Core only"),
            ("Software", f"lqtpy {manifest.get('lqtpy_version', '')}")]
    sha = (manifest.get("input") or {}).get("sha256")
    if sha:
        meta.append(("Input checksum", f"SHA-256 {sha[:16]}…"))
    prof = (manifest.get("settings") or {}).get("profile")
    if prof:
        meta.append(("Description profile", prof))
    out = ['<article class="doc">', "<h1>Lesion quantification report</h1>",
           '<div class="sub">Automated analysis of a lesion mask in MNI space</div>',
           '<table class="meta"><tbody>' + "".join(
               f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>" for k, v in meta) + "</tbody></table>"]
    out.append(h2("Findings"))
    secs = _desc_sections(res)
    for _, p in secs:
        out.append(f"<p>{_emph(p)}</p>")
    if not secs:
        out.append("<p>No description was generated.</p>")
    out.append(h2("Mask quality"))
    warn = [f for f in flags if f["severity"] != "info"]
    if warn and res.get("qc_verdict"):
        out.append(f'<p class="fine">Verdict: {_esc(res["qc_verdict"])} '
                   f'(quality score {res.get("qc_score")}/100).</p>')
    if warn:
        out += [f'<p class="flagline {f["severity"]}"><strong>{_esc(_flag_label(f))}:</strong> '
                f'{_esc(f["message"])}'
                + (f' <span class="fine">({_esc(_flag_where(f))})</span>' if _flag_where(f) else "")
                + '</p>' for f in warn]
    else:
        out.append('<p class="ok"><strong>OK:</strong> no quality problems found.</p>')
    notes = [f for f in flags if f["severity"] == "info"]
    if notes:
        out.append('<p class="fine">' + " ".join(_esc(f["message"]) for f in notes) + "</p>")
    out.append(h2("Measurements"))
    rows = [("Volume", f'{fmt.volume(m["volume_mm3"])} ({m["n_voxels"]:,} voxels of {m["voxel_volume_mm3"]} mm³)'),
            ("Separate parts", facts.get("n_clusters", m["n_components"])),
            ("Hemispheres", f'{m["pct_left"]:.0f}% left, {m["pct_right"]:.0f}% right'
             + (f', {m["pct_midline"]:.0f}% midline' if m.get("pct_midline") else "")),
            ("Laterality index", f'{m["laterality_index"]:+.2f}'),
            ("Centre of mass (MNI)", ", ".join(f"{c:.0f}" for c in m["center_of_mass_mm"]) + " mm"),
            ("Extent", " × ".join(f"{b:.0f}" for b in m["bbox_size_mm"]) + " mm")]
    tis = res.get("tissue") or {}
    if tis.get("class"):
        rows.append(("Tissue", f'{tis["class"]} ({_tissue_text(tis)})'))
    out.append('<table><tbody>' + "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                                          for k, v in rows) + "</tbody></table>")
    tparts = tis.get("parts") or []
    if len(tparts) > 1:
        shown = tparts[:15]
        out.append('<p class="fine" style="margin-top:3mm">Separate parts by location '
                   f'(reaching a structure = within {tis.get("contact_mm", 3):g} mm of it in the template)</p>'
                   '<table><thead><tr><th>Part</th><th class="n">cm³</th><th>Tissue</th><th>Location</th>'
                   '<th class="n">To ventricle</th><th class="n">To cortex</th></tr></thead><tbody>'
                   + "".join(f'<tr><td>{pt["id"]}</td><td class="n" data-sort="{pt["volume_mm3"]}">'
                             f'{fmt.volume_num(pt["volume_mm3"])}</td>'
                             f'<td>{_esc(pt.get("tissue_class") or "—")}</td>'
                             f'<td>{_esc(pt["class"])}</td><td class="n">{pt["dist_ventricle_mm"]:.0f} mm</td>'
                             f'<td class="n">{pt["dist_cortex_mm"]:.0f} mm</td></tr>' for pt in shown)
                   + "</tbody></table>"
                   + (f'<p class="fine">{len(tparts) - len(shown)} smaller parts not listed; '
                      'all are in the parts CSV.</p>' if len(tparts) > len(shown) else ""))
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
            out.append('<p class="fine" style="margin-top:3mm">Vascular territories</p>'
                       '<table><thead><tr><th>Territory</th><th class="n">% of lesion</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(n)}{" (deep perforator)" if deep else ""}</td>'
                                 f'<td class="n">{v:.1f}%</td></tr>' for n, v, deep in terrs)
                       + f'</tbody></table><p class="fine">{_TERR_LEGEND}</p>')
        near = [r for r in res.get("key_structures") or [] if r["DistanceMM"] <= 10]
        if near:
            out.append('<p class="fine" style="margin-top:3mm">Key structures within 10 mm</p>'
                       '<table><thead><tr><th>Structure</th><th class="n">Distance</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(r["Structure"])}</td><td class="n">'
                                 f'{"reaches it" if r["Reaches"] else str(round(r["DistanceMM"])) + " mm"}</td></tr>'
                                 for r in near) + "</tbody></table>")
        cross = [r for r in res.get("cross_section") or [] if r["MaxCrossSectionLoad"] >= 10][:8]
        if cross:
            out.append('<p class="fine" style="margin-top:3mm">White-matter tracts: largest share of a '
                       'cross-section covered</p><table><thead><tr><th>Tract</th><th class="n">Max cross-section</th>'
                       '<th class="n">Narrowest</th><th class="n">% of tract voxels</th></tr></thead><tbody>'
                       + "".join(f'<tr><td>{_esc(r["Tract"])}</td><td class="n">{r["MaxCrossSectionLoad"]:.0f}%</td>'
                                 f'<td class="n">{r["BottleneckLoad"]:.0f}%</td>'
                                 f'<td class="n">{r["PercentOverlap"]:.1f}%</td></tr>' for r in cross)
                       + "</tbody></table>")
    figs = []
    k = 0
    if res.get("glass_svg"):
        k += 1
        figs.append('<figure><div class="plate" data-copy-from="fig-glass"></div>'
                    f"<figcaption>Figure {k}. Projections of the lesion through the brain (from the left, "
                    "the front and above).</figcaption></figure>")
    if res.get("slices_svg"):
        k += 1
        figs.append('<figure><div class="plate" data-copy-from="fig-slices"></div>'
                    f"<figcaption>Figure {k}. Orthogonal slices through the lesion's centre of mass."
                    "</figcaption></figure>")
    for a in res["atlases"]:
        for cap, svg in a.get("flat_svgs") or []:
            if a["key"] == "dk":
                k += 1
                figs.append(f"<figure>{svg}<figcaption>Figure {k}. Percent of each DKT cortical region "
                            "damaged.</figcaption></figure>")
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
    env = ((manifest.get("run") or {}).get("environment") or {})
    if env:
        out.append('<p class="fine"><strong>Software:</strong> ' + ", ".join(
            f"{_esc(k)} {_esc(v)}" for k, v in env.items() if k != "platform") + ".</p>")
    out.append('<p class="fine">Voxels above the threshold count as lesion. Physical measures use the '
               'image affine; region overlap resamples the mask onto each atlas grid (nearest '
               'neighbour). Atlases: AAL (Tzourio-Mazoyer 2002), Schaefer-100 (Schaefer 2018), DKT31 '
               '(Klein &amp; Tourville 2012), JHU ICBM-DTI-81 (Mori 2008), arterial territories '
               '(Liu 2023); tissue classes from DKT31 and the ICBM 2009a white-matter map (Fonov '
               '2011); tracts from HCP842 (Yeh 2018). Location classes, border zones and distances are '
               'template-based and approximate. Functional labels describe location only, not '
               'symptoms. This report is generated automatically and is not a clinical diagnosis.</p>')
    out.append("</article>")
    return "".join(out)


# =========================================================================== #
# Group
# =========================================================================== #
def _group_rows(recs, key, limit=10, name_fn=None, skip=(), title_fn=None, hl_prefix=None) -> str:
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
                        max_value=top, digits=1, title=title_fn(r[key]) if title_fn else None,
                        hl=f"{hl_prefix}|{r[key]}" if hl_prefix else None))
    return "".join(out)


def _group_table(recs, key, label, limit=30, table_id=None, hl_prefix=None, name_fn=None) -> str:
    recs = _records(recs)
    if not recs:
        return '<p class="note">No overlap in this group.</p>'
    suffix = "Overlap" if "CohortMeanPercentOverlap" in recs[0] else "Damage"
    body = []
    for r in recs[:limit]:
        hl = f' data-hl="{_esc(hl_prefix)}|{_esc(r[key])}" tabindex="0"' if hl_prefix else ""
        cm = r[f"CohortMeanPercent{suffix}"]
        body.append(f'<tr{hl}><td>{_esc(name_fn(r[key]) if name_fn else r[key])}</td>'
                    f'<td class="n">{r["NSubjects"]}/{r["NSubjectsTotal"]}</td>'
                    f'<td class="n">{cm:.1f}%{_bar(cm, "dmg")}</td>'
                    f'<td class="n">{r[f"MeanPercent{suffix}Affected"]:.1f}%</td>'
                    f'<td class="n">{r[f"MaxPercent{suffix}"]:.1f}%</td></tr>')
    more = f'<p class="note">Showing {limit} of {len(recs)}; all rows are in the CSV.</p>' if len(recs) > limit else ""
    tid = f' data-table="{_esc(table_id)}"' if table_id else ""
    return (f'<div class="tablewrap"><table{tid}><thead><tr>'
            f'<th>{_esc(label)}</th><th class="n">Subjects</th><th class="n">Mean, all subjects</th>'
            '<th class="n">Mean, affected</th><th class="n">Max</th></tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{more}')


def _group_tissue_kpi(classes: dict, n: int) -> list:
    if not classes:
        return []
    order = sorted(classes.items(), key=lambda kv: -kv[1])
    rest = " · ".join(f"{v} {k}" for k, v in order[1:3])
    return [("Tissue", order[0][0], f"{order[0][1]} of {n}" + (f" · {rest}" if rest else ""))]


def _yeo_names() -> dict:
    try:
        from ..atlases import _YEO7_NAMES
        return dict(_YEO7_NAMES)
    except Exception:
        return {}


def _categorical(subjects, col) -> "list[str] | None":
    raw = [s["meta"].get(col, "") for s in subjects if str(s["meta"].get(col, "")) != ""]
    if raw and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in raw):
        return None  # numbers (age, scores) are columns to sort, not chips
    uniq = sorted({str(v) for v in raw})
    return uniq if 1 < len(uniq) <= 8 else None


def _filters(subjects, meta_cols) -> "tuple[str, list[tuple[str, str]]]":
    """Filter bar html, and [(field id, metadata column)] used by rows."""
    groups = []

    def chips(field, label, values):
        if len(values) < 2:
            return ""
        b = "".join(f'<button type="button" class="fchip" data-f="{_esc(field)}" data-v="{_esc(v)}" '
                    f'aria-pressed="false">{_esc(v)}</button>' for v in values)
        return (f'<span class="fgroup" role="group" aria-label="{_esc(label)}">'
                f'<span class="lab">{_esc(label)}</span>{b}</span>')

    qs = [q for q in ("OK", "Note", "Check", "Problem")
          if any(fmt.quality_label(s.get("qc_worst") or "ok") == q for s in subjects)]
    groups.append(chips("qc", "Quality", qs))
    groups.append(chips("terr", "Territory", sorted({s.get("territory") or "—" for s in subjects})[:12]))
    groups.append(chips("tissue", "Tissue", sorted({(s.get("tissue") or {}).get("class") or "—"
                                                    for s in subjects})))
    mfields = []
    for i, c in enumerate(meta_cols):
        vals = _categorical(subjects, c)
        if vals:
            mfields.append((f"m{i}", c))
            groups.append(chips(f"m{i}", c, vals))
    bar = ('<div class="filters"><input type="search" id="subj-search" placeholder="Search subjects" '
           'aria-label="Search subjects">' + "".join(g for g in groups if g)
           + '<button type="button" class="btn small" id="subj-clear">Clear</button>'
             '<span class="shown" id="subj-shown" aria-live="polite"></span></div>')
    return bar, mfields


def _comparison_section(cmp_: "dict | None") -> str:
    if not cmp_:
        return ""
    a, b = cmp_["a"], cmp_["b"]
    maps = ""
    svgs = cmp_.get("overlap_svgs") or {}
    if svgs:
        cells = []
        for g, n in ((a, cmp_["n_a"]), (b, cmp_["n_b"])):
            s = svgs.get(g)
            if s:
                cells.append(f'<figure><div class="plate"><div class="theme-img dark-img">{s["dark"]}</div>'
                             f'<div class="theme-img light-img">{s["light"]}</div></div>'
                             f'<figcaption>{_esc(cmp_["column"])} = {_esc(g)} (n = {n})</figcaption></figure>')
        maps = f'<div class="cols2">{"".join(cells)}</div>'
    tables = []
    for t in cmp_["tables"]:
        rows = t["rows"][:20]
        body = "".join(
            f'<tr><td>{_esc(r["Name"])}</td>'
            f'<td class="n">{r["NAffectedA"]}/{r["NA"]}</td><td class="n">{r["MeanA"]:.1f}%</td>'
            f'<td class="n">{r["NAffectedB"]}/{r["NB"]}</td><td class="n">{r["MeanB"]:.1f}%</td>'
            f'<td class="n">{r["DiffMean"]:+.1f}</td><td class="n">{r["PMannWhitney"]:.3g}</td>'
            f'<td class="n">{r["QMannWhitney"]:.3g}</td><td class="n">{r["PFisher"]:.3g}</td></tr>'
            for r in rows)
        more = (f'<p class="note">Top 20 of {len(t["rows"])}; all rows are in the CSV.</p>'
                if len(t["rows"]) > 20 else "")
        tables.append(
            f'<details class="card"{" open" if not tables else ""}><summary><span class="t">{_esc(t["label"])}</span>'
            f'<span class="s">{len(t["rows"])} rows · sorted by Mann-Whitney p</span></summary><div class="inner">'
            f'<div class="tablewrap"><table data-table="compare-{_esc(t["key"])}"><thead><tr><th>Structure</th>'
            f'<th class="n">{_esc(a)}: affected</th><th class="n">{_esc(a)}: mean</th>'
            f'<th class="n">{_esc(b)}: affected</th><th class="n">{_esc(b)}: mean</th>'
            '<th class="n">Diff.</th><th class="n">p (MW)</th><th class="n">q (MW)</th><th class="n">p (Fisher)</th>'
            f'</tr></thead><tbody>{body}</tbody></table></div>{more}</div></details>')
    return (f'<div class="section-title">Comparison: {_esc(cmp_["column"])} = {_esc(a)} (n = {cmp_["n_a"]}) '
            f'vs {_esc(b)} (n = {cmp_["n_b"]})</div>'
            '<section class="card"><h2>Lesion overlap by group</h2>'
            f'{maps}<p class="note">Means count unaffected subjects as 0. p (MW): Mann-Whitney U on percent '
            'damaged; q: Benjamini-Hochberg FDR; p (Fisher): affected vs not. A screening comparison, '
            'not a lesion-symptom mapping analysis (no covariates or lesion-volume correction).</p></section>'
            + "".join(tables))


def _cohort_quality_card(res) -> str:
    dups = res.get("duplicates") or []
    outl = res.get("outliers") or {}
    unmatched = res.get("metadata_unmatched") or []
    if not dups and not outl and not unmatched:
        return ""
    items = []
    for d in dups:
        dice = f" (Dice {d['dice']:.2f})" if d["kind"] != "identical" else ""
        items.append({"severity": "warn", "message": f"{d['a']} and {d['b']} are {d['kind']} masks{dice}.",
                      "suggestion": "Check for a duplicated file."})
    for nm, its in outl.items():
        for it in its:
            items.append({"severity": "warn", "message": f"{nm}: {it['message']}", "suggestion": ""})
    if unmatched:
        items.append({"severity": "info",
                      "message": f"{len(unmatched)} metadata row(s) matched no subject: "
                                 + ", ".join(unmatched[:6]) + ("…" if len(unmatched) > 6 else ""),
                      "suggestion": ""})
    return f'<section class="card" aria-label="Cohort checks"><h2>Cohort checks</h2>{_flags_html(items)}</section>'


def _groups_card(res) -> str:
    groups = res.get("groups")
    if not groups:
        return ""
    rows = "".join(f'<tr><td>{_esc(g["value"])}</td><td class="n">{g["n"]}</td>'
                   f'<td class="n" data-sort="{g["median_volume_mm3"]}">{fmt.volume(g["median_volume_mm3"])}</td>'
                   f'<td class="n">{g["n_left"]}</td><td class="n">{g["n_bilateral"]}</td>'
                   f'<td>{_esc(g["top_territory"] or "—")}</td><td class="n">{g["n_flagged"]}</td></tr>'
                   for g in groups)
    return (f'<section class="card"><h2>By {_esc(res["group_by"])}</h2><div class="tablewrap">'
            '<table data-table="groups"><thead><tr><th>Group</th><th class="n">Subjects</th>'
            '<th class="n">Median volume</th><th class="n">Mostly left</th><th class="n">Bilateral</th>'
            f'<th>Most common territory</th><th class="n">To check</th></tr></thead><tbody>{rows}</tbody>'
            '</table></div></section>')


def _lobe_label(x):
    return fmt.lobe_word(x, short=True, capital=True)


def render_group(res: dict, style: "ReportStyle | None" = None, *, run: "dict | None" = None,
                 downloads: "dict | None" = None) -> str:
    """The group report (dashboard + clinical views in one file)."""
    manifest = res.get("manifest") or {}
    run = run or manifest.get("run")
    n = res["n_subjects"]
    subjects = res["subjects"]
    for s in subjects:
        s.setdefault("meta", {})
    ts = res.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    vols = sorted(s["morphometry"]["volume_mm3"] for s in subjects)
    by = {a["key"]: a for a in res["atlases"]}
    meta_cols = res.get("metadata_columns") or []

    hero = (f'<div class="hero"><div><h1>Group report</h1><div class="meta">{n} subjects · '
            f'{_esc(ts)} · analysed in {_esc(res.get("elapsed", "?"))}</div></div>'
            f'<span class="spacer"></span>{_scope_pill(manifest)}<span class="pill">Group</span></div>')
    q = res.get("n_subjects_with_warnings", 0)
    med = res.get("median_volume_mm3", res.get("median_volume_cc", 0) * 1000)
    kpis = [("Subjects", str(n), f"{len(meta_cols)} metadata columns" if meta_cols else ""),
            ("Median volume", fmt.volume(med), f"range {fmt.volume_num(vols[0])}–{fmt.volume(vols[-1])}"),
            ("Max overlap", f'{res["max_overlap"]} of {n}', "subjects at the most shared voxel"),
            ("Bilateral", f'{res.get("n_bilateral", 0)} of {n}', "lesions in both hemispheres"),
            *_group_tissue_kpi(res.get("tissue_classes") or {}, n),
            ("Quality", f'<span class="status {"warn" if q else "ok"}"><span class="dot"></span>'
                        f'{fmt.quality_label("warn" if q else "ok")}</span>',
             f"{q} of {n} masks flagged" if q else "all masks passed")]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{_esc(k)}</div>'
                       f'<div class="v{" text" if k in ("Tissue", "Quality") else ""}">'
                       f'{v if k == "Quality" else _esc(v)}</div><div class="s">{_esc(s)}</div></div>'
                       for k, v, s in kpis)
    overlap = ""
    ov = res.get("overlap_svgs")
    if ov:
        overlap = ('<section class="card" aria-label="Lesion overlap"><h2>Lesion overlap</h2><div class="plate">'
                   f'<div class="theme-img dark-img">{ov["dark"]}</div><div class="theme-img light-img">'
                   f'{ov["light"]}</div></div><figcaption>Number of subjects with a lesion at each voxel '
                   '(axial slices).</figcaption></section>')

    filt, mfields = _filters(subjects, meta_cols)
    show_meta = meta_cols[:4]
    has_thumbs = any(x.get("thumb") for x in subjects)
    subj_rows = []
    for s in subjects:
        mm = s["morphometry"]
        nm = _esc(s["name"])
        if s.get("report_path"):
            nm = f'<a href="{_esc(s["report_path"].replace(chr(92), "/"))}">{nm}</a>'
        qw = s.get("qc_worst") or "ok"
        qall = s.get("qc_all", s.get("qc", []))
        ids = ", ".join(sorted({f["id"].replace("_", " ") for f in qall if f["severity"] != "info"}))
        mini = (f'<span class="mini" role="img" aria-label="{mm["pct_left"]:.0f}% left, {mm["pct_right"]:.0f}% right">'
                f'<span class="l" style="width:{mm["pct_left"]}%"></span><span class="r" style="width:{mm["pct_right"]}%">'
                "</span></span>")
        thumb = ""
        if has_thumbs:
            thumb = (f'<td><img class="thumb" src="{s["thumb"]}" alt="{_esc(s["name"])}, seen from above"></td>'
                     if s.get("thumb") else "<td></td>")
        fattrs = (f' data-subject="{_esc(s["name"])}" data-f-qc="{_esc(fmt.quality_label(qw))}"'
                  f' data-f-terr="{_esc(s.get("territory") or "—")}"'
                  f' data-f-tissue="{_esc((s.get("tissue") or {}).get("class") or "—")}"'
                  + "".join(f' data-f-{f}="{_esc(s["meta"].get(c, ""))}"' for f, c in mfields))
        metas = "".join(f'<td>{_esc(s["meta"].get(c, ""))}</td>' for c in show_meta)
        bzc = '<span class="chip">border zone</span>' if s.get("border_zone") else ""
        qsort = {"ok": 0, "info": 0, "warn": 1, "error": 2}[qw]
        subj_rows.append(
            f'<tr{fattrs}>{thumb}<td>{nm}</td>{metas}<td class="n" data-sort="{mm["volume_mm3"]}">'
            f'{fmt.volume_num(mm["volume_mm3"])}</td><td class="n">{mm["n_components"]}</td>'
            f'<td data-sort="{mm["pct_left"]:.1f}">{mini}</td><td>{_esc(s.get("territory") or "—")}{bzc}</td>'
            f'<td>{_esc((s.get("tissue") or {}).get("class") or "—")}</td>'
            f'<td data-sort="{qsort}"><span class="qc-badge {qw}">'
            f'{fmt.quality_label(qw)}</span> <span class="muted">{_esc(ids)}</span></td>'
            f'<td class="muted">{_emph(s.get("summary_marked") or s.get("summary", ""))}</td></tr>')
    thumb_h = '<th><span class="sr">Thumbnail</span></th>' if has_thumbs else ""
    subj = (f'<section class="card" aria-label="Subjects"><h2>Subjects<span class="count">{n}</span></h2>{filt}'
            f'<div class="tablewrap{" scroll" if n > 12 else ""}"><table data-table="subjects"><thead><tr>'
            f'{thumb_h}<th>Subject</th>' + "".join(f"<th>{_esc(c)}</th>" for c in show_meta)
            + '<th class="n">cm³</th><th class="n">Parts</th><th title="percent left">L / R</th>'
            '<th>Main territory</th><th>Tissue</th><th>Quality</th><th>Summary</th></tr></thead><tbody>'
            + "".join(subj_rows) + "</tbody></table></div></section>")

    yeo = _yeo_names()
    cards = []
    if by.get("dk", {}).get("lobe_df"):
        cards.append('<section class="card"><h2>Lobes (DKT)</h2><div class="rows" data-rollup="dk">'
                     f'{_group_rows(by["dk"]["lobe_df"], "Lobe", name_fn=_lobe_label)}</div></section>')
    if by.get("arterial", {}).get("network_df"):
        rows = _group_rows(by["arterial"]["network_df"], "Network", skip=("Ventricle",),
                           title_fn=lambda k: fmt.TERRITORY_NAMES.get(k, k))
        cards.append('<section class="card"><h2>Vascular territories</h2>'
                     f'<div class="rows" data-rollup="arterial" data-skip="Ventricle">{rows}</div></section>')
    if by.get("schaefer", {}).get("network_df"):
        rows = _group_rows(by["schaefer"]["network_df"], "Network", name_fn=lambda k: yeo.get(k, k))
        cards.append(f'<section class="card"><h2>Yeo-7 networks</h2><div class="rows" data-rollup="schaefer">'
                     f'{rows}</div></section>')
    where = ""
    if cards:
        where = ('<div class="section-title">Where the lesions are</div>'
                 '<div class="legend"><span><i style="background:var(--lesion)"></i>mean percent damaged, '
                 'subjects shown in the table (bars scaled to the largest value in each list)</span>'
                 '<span>chip: subjects affected</span></div>'
                 f'<div class="cols3">{"".join(cards)}</div>')
    labels = {"dk": {k: _lobe_label(k) for s in subjects for k in (s.get("rollups") or {}).get("dk", {})},
              "schaefer": yeo, "arterial": {}}
    filter_data = {"subjects": [{"name": s["name"], "rollups": s.get("rollups") or {}} for s in subjects],
                   "labels": labels}

    details = []
    for a in res["atlases"]:
        tier = '<span class="tier">EXTRA</span>' if a.get("tier") == "extra" else ""
        extra = ""
        for k, label in (("lobe_df", "Lobe"), ("network_df", "Network")):
            if a.get(k):
                extra += (f'<h3 class="section-title">{label}</h3>'
                          + _group_table(a[k], label, label, table_id=f"g{k}-{a['key']}",
                                         name_fn=_lobe_label if k == "lobe_df" else None))
        regions = _group_table(a["df"], "RegionName", "Region", table_id="gregions-" + a["key"],
                               hl_prefix=a["key"])
        details.append(f'<details class="card"><summary><span class="t">{_esc(a["name"])}{tier}</span>'
                       f'<span class="s">{len(_records(a["df"]))} regions involved</span></summary>'
                       f'<div class="inner">{_flatmaps(a)}{extra}'
                       f'<h3 class="section-title">Regions</h3>{regions}</div></details>')
    if res.get("tracts"):
        details.append('<details class="card"><summary><span class="t">White-matter tracts (voxel-overlap proxy)'
                       '</span><span class="s">not streamline disconnection</span></summary><div class="inner">'
                       + _group_table(res["tracts"]["df"], "Tract", "Tract", table_id="gtracts")
                       + "</div></details>")
    dl_card, dl_data = _downloads(downloads)
    dash = (f'<div class="dash">{hero}<div class="kpis">{kpi_html}</div>'
            f'{subj}{_cohort_quality_card(res)}{_groups_card(res)}{overlap}{where}'
            f'{_comparison_section(res.get("comparison"))}'
            f'<div class="section-title">Details</div>{"".join(details)}{dl_card}'
            f'{_about(manifest, run, "group_report.html")}</div>')

    # clinical
    sec = [0]

    def h2(title):
        sec[0] += 1
        return f'<h2><span class="no">{sec[0]}</span>{_esc(title)}</h2>'

    c = ['<article class="doc"><h1>Group lesion report</h1>',
         f'<div class="sub">{n} subjects · generated {_esc(ts)} · lqtpy '
         f'{_esc(manifest.get("lqtpy_version", ""))}</div>', h2("Cohort")]
    cohort_rows = [
        ("Subjects", n),
        ("Median volume", f'{fmt.volume(med)} (range {fmt.volume_num(vols[0])}–{fmt.volume(vols[-1])})'),
        ("Bilateral lesions", f'{res.get("n_bilateral", 0)} of {n}')]
    if res.get("tissue_classes"):
        cohort_rows.append(("Tissue class", "; ".join(f"{k} {v}" for k, v in res["tissue_classes"].items())))
    cohort_rows.append(("Masks flagged for checking", f"{q} of {n}"))
    if res.get("duplicates"):
        cohort_rows.append(("Duplicate pairs", len(res["duplicates"])))
    if res.get("outliers"):
        cohort_rows.append(("Outliers", ", ".join(res["outliers"])))
    c.append('<table><tbody>' + "".join(f"<tr><td>{_esc(k)}</td><td>{_esc(v)}</td></tr>"
                                        for k, v in cohort_rows) + "</tbody></table>")
    c.append(h2("Subjects"))
    c.append('<table><thead><tr><th>Subject</th>' + "".join(f"<th>{_esc(x)}</th>" for x in show_meta[:2])
             + '<th class="n">cm³</th><th class="n">L/R %</th>'
             '<th>Main territory</th><th>Tissue</th><th>Quality</th></tr></thead><tbody>' + "".join(
                 f'<tr><td>{_esc(s["name"])}</td>'
                 + "".join(f"<td>{_esc(s['meta'].get(x, ''))}</td>" for x in show_meta[:2])
                 + f'<td class="n">{fmt.volume_num(s["morphometry"]["volume_mm3"])}</td>'
                 f'<td class="n">{s["morphometry"]["pct_left"]:.0f}/{s["morphometry"]["pct_right"]:.0f}</td>'
                 f'<td>{_esc(s.get("territory") or "—")}</td>'
                 f'<td>{_esc((s.get("tissue") or {}).get("class") or "—")}</td>'
                 f'<td>{_esc(fmt.quality_label(s.get("qc_worst")))}</td></tr>'
                 for s in subjects) + "</tbody></table>")
    c.append(h2("Involvement"))
    for key, df_key, label in (("dk", "lobe_df", "Lobe"), ("arterial", "network_df", "Network"),
                               ("dk", "df", "RegionName")):
        a = by.get(key)
        if a and a.get(df_key):
            title = {"lobe_df": "Lobes (DKT)", "network_df": "Vascular territories",
                     "df": "Regions (DKT)"}[df_key]
            c.append(f'<p class="fine" style="margin-top:3mm">{title}</p>'
                     + _group_table(a[df_key], label, label.replace("RegionName", "Region"), limit=15,
                                    name_fn=_lobe_label if df_key == "lobe_df" else None))
    cmp_ = res.get("comparison")
    if cmp_ and cmp_["tables"]:
        c.append(h2(f"Comparison: {cmp_['column']}"))
        c.append(f'<p class="fine">{_esc(cmp_["a"])} (n = {cmp_["n_a"]}) vs {_esc(cmp_["b"])} (n = {cmp_["n_b"]}); '
                 'rows with the smallest Mann-Whitney p first. Screening statistics, not lesion-symptom mapping.</p>')
        for t in cmp_["tables"][:3]:
            body = "".join(
                f'<tr><td>{_esc(r["Name"])}</td><td class="n">{r["MeanA"]:.1f}% ({r["NAffectedA"]}/{r["NA"]})</td>'
                f'<td class="n">{r["MeanB"]:.1f}% ({r["NAffectedB"]}/{r["NB"]})</td>'
                f'<td class="n">{r["DiffMean"]:+.1f}</td><td class="n">{r["PMannWhitney"]:.3g}</td>'
                f'<td class="n">{r["QMannWhitney"]:.3g}</td></tr>' for r in t["rows"][:10])
            c.append(f'<p class="fine" style="margin-top:3mm">{_esc(t["label"])}</p><table><thead><tr>'
                     f'<th>Structure</th><th class="n">{_esc(cmp_["a"])}</th><th class="n">{_esc(cmp_["b"])}</th>'
                     f'<th class="n">Diff.</th><th class="n">p</th><th class="n">q</th></tr></thead><tbody>{body}'
                     "</tbody></table>")
    if ov:
        c.append(h2("Figure"))
        c.append(f'<figure><div class="plate">{ov["light"]}</div>'
                 "<figcaption>Figure 1. Number of subjects with a lesion at each voxel.</figcaption></figure>")
    c.append(h2("Scope and methods"))
    ran = [x["label"] for x in manifest.get("modules", []) if x["status"] == "ran"]
    c.append(f'<p class="fine"><strong>Included:</strong> {_esc(", ".join(ran))}.</p>')
    c.append('<p class="fine">"Mean, all subjects" averages over every subject (unaffected subjects count '
             'as 0); "Mean, affected" averages over affected subjects only. Each subject\'s full report '
             'is linked from the dashboard view. This report is generated automatically and is not a '
             'clinical diagnosis.</p></article>')
    tail = ((_json_script("lqtpy-run", run) if run else "")
            + _json_script("lqtpy-subjects", filter_data) + dl_data)
    return _page("LQTpy · group report", dash, "".join(c), subtitle="Group report", tail=tail)
