"""LQTpy's own self-contained, single-file HTML report (``analyze_single``/
``analyze_group`` with ``plots=True``).

Composes the same scoped, embeddable fragments a host application would use
(``lqtpy.report.fragments``) for the sections that have a direct
``lqtpy.api`` equivalent -- morphometry, per-atlas parcel damage, Yeo-7
network rollup, and the white-matter voxel-overlap proxy -- around this
module's own page chrome (plots, per-atlas ggseg flat maps, the AAL L/R
homolog table, and -- for the group report -- every aggregated
multi-subject table, none of which match a single-lesion ``lqtpy.api``
record shape, so they keep this module's original, unscoped rendering).

Like the original single-file ``lqtpy/report.py`` this replaces, this module
imports nothing from the rest of ``lqtpy`` and never imports matplotlib/
nilearn/pandas itself -- it only ever receives already-built plot URIs and
DataFrame-or-records tables from ``lqtpy.analysis`` (which imports this
module lazily, only when ``plots=True``).
"""
from __future__ import annotations

import datetime
import html

from . import fragments
from .style import DEFAULT_STYLE, ReportStyle

_CSS = """
:root { --fg:#1b1f24; --muted:#5b6673; --line:#e3e7ec; --accent:#c1443c;
        --bg:#ffffff; --card:#f7f9fb; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       color:var(--fg); background:var(--bg); margin:0; line-height:1.5; }
.wrap { max-width:1040px; margin:0 auto; padding:32px 24px 64px; }
h1 { font-size:26px; margin:0 0 4px; }
h2 { font-size:19px; margin:36px 0 12px; padding-bottom:6px;
     border-bottom:2px solid var(--line); }
h3 { font-size:15px; margin:20px 0 8px; color:var(--muted); }
.sub { color:var(--muted); font-size:13px; margin-bottom:8px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
        gap:12px; margin:16px 0; }
.stat { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:12px 14px; }
.stat .v { font-size:22px; font-weight:650; }
.stat .k { font-size:12px; color:var(--muted); text-transform:uppercase;
           letter-spacing:.03em; }
img.plot { max-width:100%; height:auto; display:block; margin:8px auto 4px;
           border:1px solid var(--line); border-radius:8px; background:#fff; }
table { border-collapse:collapse; width:100%; font-size:13px; margin:8px 0 4px; }
th,td { text-align:left; padding:6px 10px; border-bottom:1px solid var(--line); }
th { background:var(--card); font-weight:600; }
td.num,th.num { text-align:right; font-variant-numeric:tabular-nums; }
.bar { display:inline-block; height:9px; background:var(--accent);
       border-radius:3px; vertical-align:middle; }
.tag { display:inline-block; padding:1px 8px; border-radius:999px;
       background:var(--card); border:1px solid var(--line); font-size:12px;
       color:var(--muted); }
.foot { margin-top:48px; color:var(--muted); font-size:12px;
        border-top:1px solid var(--line); padding-top:12px; }
.warn { background:#fff6e5; border:1px solid #f0d089; color:#7a5b12;
        padding:8px 12px; border-radius:8px; font-size:13px; margin:10px 0; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
@media (max-width:720px){ .two{grid-template-columns:1fr;} }
"""


def _esc(x) -> str:
    return html.escape(str(x))


def _stat(k, v) -> str:
    return f'<div class="stat"><div class="v">{_esc(v)}</div><div class="k">{_esc(k)}</div></div>'


def _records(records_or_frame) -> "list[dict]":
    """Accept records (list[dict]), a pandas DataFrame, or None; normalise
    to records. Duck-typed (``.to_dict``/``.columns``) so this module never
    has to ``import pandas`` itself -- ``analyze_single``/``analyze_group``
    only build a DataFrame here at all once matplotlib/pandas are already
    required (``plots=True``)."""
    if records_or_frame is None:
        return []
    if hasattr(records_or_frame, "to_dict") and hasattr(records_or_frame, "columns"):
        return records_or_frame.to_dict("records")
    return list(records_or_frame)


def _img(uri, alt="") -> str:
    if not uri:
        return ""
    return f'<img class="plot" src="{uri}" alt="{_esc(alt)}">'


def _flat_maps(a) -> str:
    """Render an atlas's list of (caption, uri) ggseg-style flat maps."""
    out = []
    for cap, uri in a.get("flat_uris", []):
        out.append(_img(uri, cap))
        if cap:
            out.append(f'<p class="sub" style="text-align:center">{_esc(cap)}</p>')
    return "".join(out)


def _group_network_block(a) -> str:
    """Group Yeo-7 network bar + table for the Schaefer section."""
    df = a.get("network_df")
    if df is None or df.empty:
        return ""
    rows = []
    for _, r in df.iterrows():
        rows.append(
            f'<tr><td>{_esc(r["Network"])}</td>'
            f'<td class="num">{int(r["NSubjects"])}</td>'
            f'<td class="num">{r["MeanPercentDamage"]:.1f}%</td>'
            f'<td class="num">{r["MaxPercentDamage"]:.1f}%</td></tr>'
        )
    table = (
        '<h3>Functional network involvement (Yeo-7)</h3>'
        '<table><thead><tr><th>Network</th><th class="num">#subjects</th>'
        '<th class="num">mean % dmg</th><th class="num">max % dmg</th>'
        '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
    )
    return ('<div class="two">'
            f'<div>{_img(a.get("network_bar_uri"),"group network bar")}</div>'
            f'<div>{table}</div></div>')


def _homolog_table(df, top=15) -> str:
    """Left/right homologous-region symmetry table (AAL)."""
    if df is None or df.empty:
        return ""
    d = df.head(top)
    rows = []
    for _, r in d.iterrows():
        rows.append(
            f'<tr><td>{_esc(r["Region"])}</td>'
            f'<td class="num">{r["LeftPercent"]:.1f}%</td>'
            f'<td class="num">{r["RightPercent"]:.1f}%</td>'
            f'<td class="num">{r["AbsAsymmetry"]:.1f}</td></tr>'
        )
    note = "" if len(df) <= top else f'<p class="sub">Showing top {top} of {len(df)} homologous pairs.</p>'
    return (
        '<h3>Left/right symmetry (homologous regions)</h3>'
        '<table><thead><tr><th>Region</th><th class="num">Left %</th>'
        '<th class="num">Right %</th><th class="num">|asym|</th>'
        '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>" + note
    )


_TRACT_CAVEAT = (
    "Structural-overlap proxy: the percent of each tract's voxels that the "
    "lesion touches - <em>not</em> streamline-level disconnection (a small "
    "lesion can fully disconnect a tract while overlapping few voxels). Tracts "
    "are in DSI-Studio template space, aligned to the lesion to ~1-2 mm."
)


def _group_tract_table(df, top=20) -> str:
    """Group white-matter tract overlap table (mean/max % overlap across
    subjects) -- aggregated columns, so this doesn't match
    ``tract_proxy_overlap``'s per-lesion record shape and isn't rendered via
    ``fragments.tract_proxy_fragment``."""
    if df is None or df.empty:
        return '<p class="sub">No white-matter tracts overlap this group.</p>'
    d = df.head(top)
    rows = []
    for _, r in d.iterrows():
        rows.append(
            f'<tr><td>{_esc(r["Tract"])}</td>'
            f'<td class="num">{int(r["NSubjects"])}</td>'
            f'<td class="num">{r["MeanPercentOverlap"]:.1f}%</td>'
            f'<td class="num">{r["MaxPercentOverlap"]:.1f}%</td></tr>'
        )
    head = ('<th>Tract</th><th class="num">#subjects</th>'
            '<th class="num">mean % overlap</th><th class="num">max % overlap</th>')
    return (f'<table><thead><tr>{head}</tr></thead><tbody>'
            + "".join(rows) + "</tbody></table>")


def _group_tract_section(tr) -> str:
    """White-matter tract involvement section for the group report."""
    if not tr:
        return ""
    return (
        '<h2>White-matter tract involvement</h2>'
        f'<p class="sub">{_TRACT_CAVEAT}</p>'
        '<div class="two">'
        f'<div>{_img(tr.get("bar_uri"),"tract overlap bar")}</div>'
        f'<div>{_group_tract_table(tr["df"])}</div>'
        '</div>'
    )


def render_single(res: dict, style: "ReportStyle | None" = None) -> str:
    style = style or DEFAULT_STYLE
    m = res["morphometry"]
    warn = ""
    if not res.get("was_binary", True):
        warn += '<div class="warn">Input was not binary; thresholded to a 0/1 mask before analysis.</div>'

    morphometry_html = fragments.morphometry_fragment(m, style)
    plots = f'<div class="two">{_img(res.get("glass_uri"),"glass brain")}{_img(res.get("slices_uri"),"slices")}</div>'

    atlas_sections = []
    for a in res["atlases"]:
        parts = [f'<h2>{_esc(a["name"])}</h2>']
        parts.append(_flat_maps(a))
        if a.get("network_df") is not None:
            net_records = _records(a["network_df"])
            parts.append('<div class="two">')
            parts.append(f'<div>{_img(a.get("network_bar_uri"),"network bar")}</div>')
            parts.append(f'<div>{fragments.network_rollup_fragment(net_records, style)}</div>')
            parts.append('</div>')
        if a.get("lobe_df") is not None:
            lobe_records = _records(a["lobe_df"])
            parts.append(fragments.network_rollup_fragment(
                lobe_records, style, title=f'Lobe involvement — {a["name"]}',
                key="Lobe"))
        damage_records = _records(a["df"])
        parts.append('<div class="two">')
        parts.append(f'<div>{_img(a.get("bar_uri"),"damage bar chart")}</div>')
        parts.append(f'<div>{fragments.parcel_damage_fragment(damage_records, a["name"], style)}</div>')
        parts.append('</div>')
        if a.get("homolog_df") is not None:
            parts.append(_homolog_table(a["homolog_df"]))
        atlas_sections.append("".join(parts))

    tract_html = ""
    tr = res.get("tracts")
    if tr:
        tract_records = _records(tr["df"])
        tract_html = (
            '<div class="two">'
            f'<div>{_img(tr.get("bar_uri"),"tract overlap bar")}</div>'
            f'<div>{fragments.tract_proxy_fragment(tract_records, style)}</div>'
            '</div>'
        )

    ts = res.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LQTpy report - {_esc(res['name'])}</title><style>{_CSS}
{fragments.stylesheet(style)}</style></head>
<body><div class="wrap">
<h1>Lesion report: {_esc(res['name'])}</h1>
<div class="sub">{_esc(res.get('source_path',''))} &nbsp;|&nbsp; generated {_esc(ts)}
&nbsp;|&nbsp; analysed in {res.get('elapsed','?')}</div>
{warn}
{morphometry_html}
{plots}
{''.join(atlas_sections)}
{tract_html}
<div class="foot">LQTpy - Lesion Quantification Toolkit. Physical measures use the
input image affine; region overlap uses each atlas grid (nearest-neighbour resampling).</div>
</div></body></html>"""


def render_group(res: dict, style: "ReportStyle | None" = None) -> str:
    # `style` is accepted for interface parity with render_single (a caller
    # shouldn't have to branch on report type to pass one), but every table
    # here is a cross-subject aggregate (NSubjects/Mean.../Max...) with no
    # matching single-lesion lqtpy.api record shape, so none of it renders
    # through lqtpy.report.fragments -- see the module docstring.
    style = style or DEFAULT_STYLE
    n = res["n_subjects"]
    subj_rows = []
    for s in res["subjects"]:
        m = s["morphometry"]
        subj_rows.append(
            f'<tr><td>{_esc(s["name"])}</td>'
            f'<td class="num">{m["volume_cc"]}</td>'
            f'<td class="num">{m["n_voxels"]:,}</td>'
            f'<td class="num">{m["n_components"]}</td>'
            f'<td class="num">{m["pct_left"]:.0f}/{m["pct_right"]:.0f}</td>'
            f'<td class="num">{s["top_region"]}</td></tr>'
        )
    subj_table = (
        '<table><thead><tr><th>Subject</th><th class="num">Volume (cc)</th>'
        '<th class="num">Voxels</th><th class="num">Clusters</th>'
        '<th class="num">L/R %</th><th>Top damaged region</th></tr></thead><tbody>'
        + "".join(subj_rows) + "</tbody></table>"
    )

    vols = [s["morphometry"]["volume_cc"] for s in res["subjects"]]
    stats = "".join([
        _stat("Subjects", n),
        _stat("Median volume", f'{res["median_volume_cc"]} cc'),
        _stat("Volume range", f'{min(vols)}-{max(vols)} cc'),
        _stat("Max overlap", f'{res["max_overlap"]} subj'),
        _stat("Median laterality", f'{res.get("median_laterality_index", 0.0):+.2f}'),
        _stat("Bilateral", f'{res.get("n_bilateral", 0)} / {n}'),
    ])

    group_sections = []
    for a in res["atlases"]:
        rows = []
        d = a["df"].head(20)
        for _, r in d.iterrows():
            rows.append(
                f'<tr><td>{_esc(r["RegionName"])}</td>'
                f'<td class="num">{int(r["NSubjects"])}</td>'
                f'<td class="num">{r["MeanPercentDamage"]:.1f}%</td>'
                f'<td class="num">{r["MaxPercentDamage"]:.1f}%</td></tr>'
            )
        table = (
            '<table><thead><tr><th>Region</th><th class="num">#subjects</th>'
            '<th class="num">mean % dmg</th><th class="num">max % dmg</th>'
            '</tr></thead><tbody>' + "".join(rows) + "</tbody></table>"
        )
        parts = [f'<h2>{_esc(a["name"])}</h2>']
        parts.append(_flat_maps(a))
        if a.get("network_df") is not None:
            parts.append(_group_network_block(a))
        parts.append('<div class="two">')
        parts.append(f'<div>{_img(a.get("bar_uri"),"group bar")}</div>')
        parts.append(f'<div>{table}</div>')
        parts.append('</div>')
        group_sections.append("".join(parts))

    tract_section = _group_tract_section(res.get("tracts"))
    ts = res.get("timestamp", datetime.datetime.now().isoformat(timespec="seconds"))
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LQTpy group report</title><style>{_CSS}</style></head>
<body><div class="wrap">
<h1>Group lesion report</h1>
<div class="sub">n={n} subjects &nbsp;|&nbsp; generated {_esc(ts)}
&nbsp;|&nbsp; analysed in {res.get('elapsed','?')}</div>
<h2>Overview</h2>
<div class="grid">{stats}</div>
{_img(res.get("overlap_uri"),"lesion overlap map")}
<h2>Per-subject summary</h2>
{subj_table}
{''.join(group_sections)}
{tract_section}
<div class="foot">LQTpy - Lesion Quantification Toolkit. Group damage = frequency and
severity of region overlap across subjects sharing the same atlas grid.</div>
</div></body></html>"""
