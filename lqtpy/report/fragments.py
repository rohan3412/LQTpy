"""Self-contained, embeddable HTML fragments for LQTpy results.

A host application (e.g. a clinical viewer composing its own report page)
embeds these directly for sections it doesn't render itself: per-tract
disconnection, network rollup, parcel-damage tables, morphometry detail.
LQTpy's own standalone report (``lqtpy.report``) uses the same functions.

Every function here is pure: given the record shapes ``lqtpy.api`` returns
(plus an optional :class:`~lqtpy.report.style.ReportStyle`), it returns an
HTML string. **This module never imports matplotlib, nilearn, or pandas**
(``tests/test_import_hygiene.py`` checks this via a subprocess probe) --
plotting/report-page assembly lives in ``lqtpy.report`` (the standalone
report), not here.

Isolation contract (see ``tests/test_report_fragments.py`` for the checks):

- Every rule in :func:`stylesheet` selects only ``.lqt-`` classes -- no
  bare element selectors, no ``*``/``:root``/``html``/``body``, no
  ``!important``, no ids.
- Every fragment's markup sits inside a single ``lqt-root`` wrapper and
  uses semantic HTML (``<section>``, ``<table>`` with ``<thead>``/
  ``<th scope="col">``) with an accessible heading.
- Every data value (region/tract names, atlas names) is escaped with
  ``html.escape`` -- atlas labels come from files a host may register
  itself (:func:`lqtpy.atlases.register_atlas`), so they're treated as
  untrusted.
- Severity colouring is never the only signal: the percentage number is
  always printed alongside the colour bar.
"""
from __future__ import annotations

import html

from .style import DEFAULT_STYLE, ReportStyle

__all__ = [
    "stylesheet", "morphometry_fragment", "parcel_damage_fragment",
    "network_rollup_fragment", "disconnection_fragment",
    "tract_proxy_fragment", "render_sections",
]

_GRIFFIS_CITATION = (
    "Griffis JC, Metcalf NV, Corbetta M, Shulman GL. “Lesion Quantification "
    "Toolkit: A MATLAB software tool for estimating grey matter damage and white "
    "matter disconnections in patients with focal brain lesions.” "
    "NeuroImage: Clinical 30:102639 (2021)."
)


# --------------------------------------------------------------------------- #
# Small shared helpers.
# --------------------------------------------------------------------------- #
def _esc(value) -> str:
    return html.escape(str(value))


def _fmt_pct(value: float, style: ReportStyle) -> str:
    return f"{float(value):.{style.percent_decimals}f}"


def _bar(percent: float, style: ReportStyle, max_percent: float = 100.0) -> str:
    """A small horizontal severity bar. Colour is never the only signal --
    callers always print the percentage text alongside this."""
    max_percent = max(float(max_percent), 1e-9)
    width = max(0.0, min(100.0, 100.0 * float(percent) / max_percent))
    color = _esc(style.severity.bucket(float(percent)))
    return (
        f'<span class="lqt-bar-track"><span class="lqt-bar-fill" '
        f'style="width:{width:.1f}%;background:{color}"></span></span>'
    )


def _stat_card(key, value) -> str:
    return (
        f'<div class="lqt-stat"><div class="lqt-stat-v">{_esc(value)}</div>'
        f'<div class="lqt-stat-k">{_esc(key)}</div></div>'
    )


def _empty(style: ReportStyle, message: str) -> str:
    return f'<p class="lqt-empty">{_esc(message)}</p>'


def _section(css_class: str, heading: str, body: str) -> str:
    return f'<section class="lqt-root lqt-section {css_class}">{heading}{body}</section>'


# --------------------------------------------------------------------------- #
# stylesheet()
# --------------------------------------------------------------------------- #
def stylesheet(style: "ReportStyle | None" = None) -> str:
    """Scoped CSS for every fragment in this module, generated once from
    ``style``. A host embedding several fragments includes this a single
    time (e.g. ``<style>{stylesheet()}</style>``).

    Every selector below is composed only of ``.lqt-`` classes (plus
    pseudo-classes like ``:nth-child``) -- no bare element selectors, no
    ``*``/``:root``/``html``/``body``, no ids, no ``!important`` -- so the
    stylesheet is safe to drop into a page with its own, unrelated CSS.
    Colours become CSS custom properties declared on ``.lqt-root`` itself,
    never on ``:root``. Several rules also pin properties a hostile host
    stylesheet commonly resets on bare elements (``table``, ``h2``,
    ``section``, a blanket ``*`` rule) -- class selectors outrank
    same-specificity element selectors, so these stay intact next to a
    host's own ``table {}``/``h2 {}``/``* {}`` rules.
    """
    style = style or DEFAULT_STYLE
    sev = style.severity
    return f"""\
.lqt-root {{
  --lqt-accent: {style.accent};
  --lqt-text: {style.text};
  --lqt-muted: {style.muted_text};
  --lqt-border: {style.border};
  --lqt-surface: {style.surface};
  --lqt-stripe: {style.row_stripe};
  --lqt-sev-low: {sev.low};
  --lqt-sev-medium: {sev.medium};
  --lqt-sev-high: {sev.high};
  display: block;
  box-sizing: border-box;
  margin: 0;
  padding: 0;
  color: var(--lqt-text);
  font-family: inherit;
  font-size: 14px;
  line-height: 1.5;
  text-transform: none;
}}
.lqt-section {{ display: block; box-sizing: border-box; margin: 0 0 28px; padding: 0; }}
.lqt-h2 {{
  display: block; font-size: 18px; font-weight: 650; margin: 0 0 10px; padding: 0 0 6px;
  border: none; border-bottom: 2px solid var(--lqt-border); color: var(--lqt-text);
  text-transform: none; box-sizing: border-box;
}}
.lqt-h3 {{
  display: block; font-size: 13px; font-weight: 600; margin: 14px 0 6px; padding: 0;
  color: var(--lqt-muted); text-transform: none;
}}
.lqt-note {{ font-size: 12px; color: var(--lqt-muted); margin: 4px 0 10px; padding: 0; }}
.lqt-empty {{ font-size: 13px; color: var(--lqt-muted); margin: 6px 0; padding: 0; }}
.lqt-em {{ font-style: italic; margin: 0; padding: 0; }}
.lqt-cite {{ font-size: 11px; color: var(--lqt-muted); margin: 0; padding: 0; }}
.lqt-badge {{
  display: inline-block; padding: 1px 8px; margin: 0 0 0 8px; border-radius: 999px;
  background: var(--lqt-surface); border: 1px solid var(--lqt-border); font-size: 11px;
  color: var(--lqt-muted); text-transform: none; box-sizing: border-box;
}}
.lqt-badge-proxy {{ background: var(--lqt-sev-medium); border-color: var(--lqt-sev-medium); color: #1b1f24; }}
.lqt-grid {{
  display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px; margin: 10px 0; padding: 0; box-sizing: border-box;
}}
.lqt-stat {{
  display: block; background: var(--lqt-surface); border: 1px solid var(--lqt-border);
  border-radius: 10px; padding: 10px 12px; margin: 0; box-sizing: border-box;
}}
.lqt-stat-v {{ font-size: 20px; font-weight: 650; margin: 0; padding: 0; }}
.lqt-stat-k {{
  font-size: 11px; color: var(--lqt-muted); text-transform: uppercase; letter-spacing: .03em;
  margin: 0; padding: 0;
}}
.lqt-table {{
  display: table; border-collapse: collapse; width: 100%; font-size: 13px;
  margin: 6px 0; padding: 0; border: none; box-sizing: border-box;
}}
.lqt-th {{
  display: table-cell; text-align: left; padding: 6px 10px; margin: 0;
  border: none; border-bottom: 1px solid var(--lqt-border); background: var(--lqt-surface);
  font-weight: 600; font-size: 13px; box-sizing: border-box;
}}
.lqt-td {{
  display: table-cell; text-align: left; padding: 6px 10px; margin: 0;
  border: none; border-bottom: 1px solid var(--lqt-border); font-size: 13px;
  box-sizing: border-box;
}}
.lqt-num {{ text-align: right; font-variant-numeric: tabular-nums; }}
.lqt-row {{ display: table-row; margin: 0; padding: 0; }}
.lqt-row:nth-child(even) {{ background: var(--lqt-stripe); }}
.lqt-bar-track {{
  display: inline-block; width: 90px; height: 8px; margin: 0 0 0 6px; padding: 0;
  background: var(--lqt-border); border-radius: 3px; vertical-align: middle; box-sizing: border-box;
}}
.lqt-bar-fill {{ display: block; height: 100%; margin: 0; padding: 0; border-radius: 3px; box-sizing: border-box; }}
.lqt-proxy {{ border-left: 3px solid var(--lqt-sev-medium); padding-left: 10px; }}
.lqt-discon {{ border-left: 3px solid var(--lqt-accent); padding-left: 10px; }}
"""


# --------------------------------------------------------------------------- #
# morphometry_fragment
# --------------------------------------------------------------------------- #
def morphometry_fragment(stats: dict, style: "ReportStyle | None" = None) -> str:
    """Render the morphometry stat grid for one lesion.

    ``stats`` is exactly what :func:`lqtpy.api.lesion_stats` (equivalently
    :func:`lqtpy.morphometry.compute_morphometry`) returns.
    """
    style = style or DEFAULT_STYLE
    vol = style.volume_from_mm3(float(stats["volume_mm3"]))
    unit = style.volume_unit_label()
    li = stats.get("laterality_index", 0.0)
    lat_tag = "bilateral" if stats.get("bilateral") else "unilateral"

    cards = [
        _stat_card("Volume", f"{vol:.{style.volume_decimals}f} {unit}"),
        _stat_card("Voxels", f'{stats["n_voxels"]:,}'),
        _stat_card("Components", stats["n_components"]),
        _stat_card("Largest cluster", f'{_fmt_pct(stats["largest_component_pct"], style)}%'),
        _stat_card("Left / Right", f'{stats["pct_left"]:.0f}% / {stats["pct_right"]:.0f}%'),
        _stat_card(f"Laterality ({lat_tag})", f'{li:+.2f}'),
    ]
    if style.show_secondary():
        cards.append(_stat_card(
            "Centre (mm)", ", ".join(str(v) for v in stats["center_of_mass_mm"])))

    detail = ""
    if style.show_secondary():
        bbox = " x ".join(str(v) for v in stats["bbox_size_mm"])
        vox = " x ".join(str(v) for v in stats["voxel_size_mm"])
        grid = " x ".join(str(v) for v in stats["shape"])
        detail = (
            f'<p class="lqt-note">Bounding box: {_esc(bbox)} mm &nbsp;|&nbsp; '
            f'Voxel size: {_esc(vox)} mm &nbsp;|&nbsp; Grid: {_esc(grid)}</p>'
        )

    heading = '<h2 class="lqt-h2">Morphometry</h2>'
    body = f'<div class="lqt-grid">{"".join(cards)}</div>{detail}'
    return _section("lqt-morphometry", heading, body)


# --------------------------------------------------------------------------- #
# parcel_damage_fragment
# --------------------------------------------------------------------------- #
def parcel_damage_fragment(records: "list[dict]", atlas_name: str,
                           style: "ReportStyle | None" = None) -> str:
    """Render the per-region atlas-overlap ("parcel damage") table.

    ``records`` is exactly what :func:`lqtpy.api.atlas_overlap` returns for
    one lesion/atlas (keys ``LabelID``, ``RegionName``, ``Group``,
    ``RegionVoxels``, ``LesionVoxels``, ``PercentDamage``,
    ``PercentOfLesion``), sorted by ``PercentDamage`` descending.
    """
    style = style or DEFAULT_STYLE
    heading = f'<h2 class="lqt-h2">Parcel damage — {_esc(atlas_name)}</h2>'
    if not records:
        return _section("lqt-parcel-damage", heading,
                        _empty(style, "No atlas regions overlap this lesion."))

    top_n = style.top_n()
    shown = records if top_n is None else records[:top_n]
    max_damage = max((r["PercentDamage"] for r in records), default=1.0) or 1.0

    rows = []
    for r in shown:
        cells = [f'<td class="lqt-td">{_esc(r["RegionName"])}</td>']
        if style.show_secondary():
            cells.append(f'<td class="lqt-td">{_esc(r["Group"])}</td>')
        cells.append(
            f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentDamage"], style)}%'
            f'{_bar(r["PercentDamage"], style, max_damage)}</td>'
        )
        if style.show_secondary():
            cells.append(f'<td class="lqt-td lqt-num">{r["LesionVoxels"]:,}</td>')
            cells.append(
                f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentOfLesion"], style)}%</td>')
        rows.append(f'<tr class="lqt-row">{"".join(cells)}</tr>')

    head = ['<th class="lqt-th" scope="col">Region</th>']
    if style.show_secondary():
        head.append('<th class="lqt-th" scope="col">Group</th>')
    head.append('<th class="lqt-th lqt-num" scope="col">% region damaged</th>')
    if style.show_secondary():
        head.append('<th class="lqt-th lqt-num" scope="col">Lesion voxels</th>')
        head.append('<th class="lqt-th lqt-num" scope="col">% of lesion</th>')

    note = ""
    if top_n is not None and len(records) > top_n:
        note = f'<p class="lqt-note">Showing top {top_n} of {len(records)} affected regions.</p>'

    table = (
        f'<table class="lqt-table"><thead><tr class="lqt-row">{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{note}'
    )
    return _section("lqt-parcel-damage", heading, table)


# --------------------------------------------------------------------------- #
# network_rollup_fragment
# --------------------------------------------------------------------------- #
#: The seven Yeo-7 network names, as they appear in Schaefer region labels.
YEO7_NETWORKS = frozenset(
    {"Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default"})


def network_rollup_fragment(records: "list[dict]", style: "ReportStyle | None" = None,
                            *, title: "str | None" = None, key: str = "Network") -> str:
    """Render a group rollup table for one atlas.

    ``records`` is exactly what :func:`lqtpy.api.network_rollup_for` returns
    (keys ``Network``, ``NetworkVoxels``, ``LesionVoxels``,
    ``PercentDamage``, ``PercentOfLesion``), sorted by ``PercentDamage``
    descending.

    The heading follows the DATA, not the function name. "Network" here is
    whatever the atlas puts in its region Group field: Yeo-7 functional
    networks for Schaefer, but hemispheres for AAL and DKT. Printing
    hemispheres under a "Yeo-7" heading in a clinical report would be a
    straightforward misstatement, so the Yeo-7 wording is used only when the
    groups really are the seven Yeo networks. Pass ``title`` to override.

    ``key`` is the record field holding the group label -- ``"Network"`` by
    default. Pass ``key="Lobe"`` to render :func:`lqtpy.api.lobe_overlap`
    records instead (same shape otherwise: a row per lobe, keys ``Lobe``,
    ``LobeVoxels``, ``LesionVoxels``, ``PercentDamage``, ``PercentOfLesion``)
    -- this is the same rollup shape either way, so it reuses this fragment
    rather than duplicating it; pass ``title`` too, since the Yeo-7
    auto-titling below only applies to ``key="Network"``.
    """
    style = style or DEFAULT_STYLE
    groups = {str(r.get(key, "")) for r in records}
    if title is not None:
        label = title
    elif key == "Network" and records and groups <= YEO7_NETWORKS:
        label = "Functional network involvement (Yeo-7)"
    elif key == "Network":
        label = "Region group involvement"
    else:
        label = f"{key} involvement"
    heading = f'<h2 class="lqt-h2">{_esc(label)}</h2>'
    if not records:
        return _section("lqt-network-rollup", heading,
                        _empty(style, f"No {key.lower()}s overlap this lesion."))

    top_n = style.top_n()
    shown = records if top_n is None else records[:top_n]

    rows = []
    for r in shown:
        cells = [
            f'<td class="lqt-td">{_esc(r[key])}</td>',
            f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentDamage"], style)}%</td>',
        ]
        if style.show_secondary():
            cells.append(
                f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentOfLesion"], style)}%</td>')
            cells.append(f'<td class="lqt-td lqt-num">{r["LesionVoxels"]:,}</td>')
        rows.append(f'<tr class="lqt-row">{"".join(cells)}</tr>')

    head = [
        f'<th class="lqt-th" scope="col">{_esc(key)}</th>',
        f'<th class="lqt-th lqt-num" scope="col">% {_esc(key.lower())} damaged</th>',
    ]
    if style.show_secondary():
        head.append('<th class="lqt-th lqt-num" scope="col">% of lesion</th>')
        head.append('<th class="lqt-th lqt-num" scope="col">Lesion voxels</th>')

    note = ""
    if top_n is not None and len(records) > top_n:
        note = f'<p class="lqt-note">Showing top {top_n} of {len(records)} {key.lower()}s.</p>'

    table = (
        f'<table class="lqt-table"><thead><tr class="lqt-row">{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{note}'
    )
    return _section("lqt-network-rollup", heading, table)


# --------------------------------------------------------------------------- #
# disconnection_fragment
# --------------------------------------------------------------------------- #
def disconnection_fragment(records: "list[dict]", style: "ReportStyle | None" = None) -> str:
    """Render the real, per-tract **streamline disconnection** table.

    ``records`` is exactly what :func:`lqtpy.api.tract_disconnection`
    returns (keys ``tract``, ``name``, ``n_streamlines``, ``n_disconnected``,
    ``percent_disconnected``), sorted by tract code.

    Deliberately visually distinct from :func:`tract_proxy_fragment`: a
    different title, a different accent (``lqt-discon`` vs ``lqt-proxy``),
    and a one-line method note plus the Griffis et al. 2021 citation in
    small print, so a reader can never confuse this with the voxel-overlap
    proxy.
    """
    style = style or DEFAULT_STYLE
    heading = (
        '<h2 class="lqt-h2">Streamline disconnection'
        '<span class="lqt-badge">per-tract, streamline-level</span></h2>'
    )
    note = (
        '<p class="lqt-note">Percent of each tract’s reference streamlines passing '
        'through the lesion. <span class="lqt-cite">' + _esc(_GRIFFIS_CITATION) + '</span></p>'
    )
    if not records:
        return _section("lqt-discon", heading,
                        note + _empty(style, "No disconnection data available."))

    pool = records if style.show_zero_rows() else [
        r for r in records if r["percent_disconnected"] > 0]
    top_n = style.top_n()
    shown = pool if top_n is None else sorted(
        pool, key=lambda r: r["percent_disconnected"], reverse=True)[:top_n]

    rows = []
    for r in shown:
        cells = [
            f'<td class="lqt-td">{_esc(r["name"])}</td>',
            f'<td class="lqt-td lqt-num">{_fmt_pct(r["percent_disconnected"], style)}%'
            f'{_bar(r["percent_disconnected"], style)}</td>',
        ]
        if style.show_secondary():
            cells.append(
                f'<td class="lqt-td lqt-num">{r["n_disconnected"]:,} / {r["n_streamlines"]:,}</td>')
        rows.append(f'<tr class="lqt-row">{"".join(cells)}</tr>')

    head = [
        '<th class="lqt-th" scope="col">Tract</th>',
        '<th class="lqt-th lqt-num" scope="col">% streamlines disconnected</th>',
    ]
    if style.show_secondary():
        head.append('<th class="lqt-th lqt-num" scope="col">Disconnected / total</th>')

    tail_note = ""
    if top_n is not None and len(pool) > top_n:
        tail_note = f'<p class="lqt-note">Showing top {top_n} of {len(pool)} tracts.</p>'

    table = (
        f'<table class="lqt-table"><thead><tr class="lqt-row">{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{tail_note}'
    )
    return _section("lqt-discon", heading, note + table)


# --------------------------------------------------------------------------- #
# tract_proxy_fragment
# --------------------------------------------------------------------------- #
def tract_proxy_fragment(records: "list[dict]", style: "ReportStyle | None" = None) -> str:
    """Render the white-matter **voxel-overlap proxy** table.

    ``records`` is exactly what :func:`lqtpy.api.tract_proxy_overlap`
    returns (keys ``Tract``, ``Hemi``, ``TractVoxels``, ``LesionVoxels``,
    ``PercentOverlap``, ``PercentOfLesion``), sorted by ``PercentOverlap``
    descending; zero-overlap tracts already dropped by the API.

    Deliberately labelled and styled to be unmistakable as a **proxy, not
    disconnection**: a ``NOT disconnection`` badge in the heading and a
    different accent class (``lqt-proxy`` vs ``lqt-discon``) from
    :func:`disconnection_fragment`.
    """
    style = style or DEFAULT_STYLE
    heading = (
        '<h2 class="lqt-h2">Voxel-overlap proxy — white-matter tracts'
        '<span class="lqt-badge lqt-badge-proxy">NOT disconnection</span></h2>'
    )
    note = (
        '<p class="lqt-note">Percent of each tract’s <span class="lqt-em">voxels</span> '
        'the lesion’s footprint touches — a coarse structural proxy, not '
        'streamline-level disconnection. A small, well-placed lesion can fully sever a '
        'tract while overlapping only a few of its voxels.</p>'
    )
    if not records:
        return _section("lqt-proxy", heading,
                        note + _empty(style, "No white-matter tracts overlap this lesion."))

    top_n = style.top_n()
    shown = records if top_n is None else records[:top_n]
    max_overlap = max((r["PercentOverlap"] for r in records), default=1.0) or 1.0

    rows = []
    for r in shown:
        # r["Tract"] already embeds "(L)"/"(R)" when applicable (see
        # tracts.py's _pretty()) -- r["Hemi"] is the same information split
        # out as its own field, not a second suffix to append.
        cells = [
            f'<td class="lqt-td">{_esc(r["Tract"])}</td>',
            f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentOverlap"], style)}%'
            f'{_bar(r["PercentOverlap"], style, max_overlap)}</td>',
        ]
        if style.show_secondary():
            cells.append(f'<td class="lqt-td lqt-num">{r["LesionVoxels"]:,}</td>')
            cells.append(
                f'<td class="lqt-td lqt-num">{_fmt_pct(r["PercentOfLesion"], style)}%</td>')
        rows.append(f'<tr class="lqt-row">{"".join(cells)}</tr>')

    head = [
        '<th class="lqt-th" scope="col">Tract</th>',
        '<th class="lqt-th lqt-num" scope="col">% tract voxels overlapped</th>',
    ]
    if style.show_secondary():
        head.append('<th class="lqt-th lqt-num" scope="col">Lesion voxels</th>')
        head.append('<th class="lqt-th lqt-num" scope="col">% of lesion</th>')

    tail_note = ""
    if top_n is not None and len(records) > top_n:
        tail_note = f'<p class="lqt-note">Showing top {top_n} of {len(records)} tracts.</p>'

    table = (
        f'<table class="lqt-table"><thead><tr class="lqt-row">{"".join(head)}</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>{tail_note}'
    )
    return _section("lqt-proxy", heading, note + table)


# --------------------------------------------------------------------------- #
# render_sections
# --------------------------------------------------------------------------- #
def render_sections(results: dict, style: "ReportStyle | None" = None) -> str:
    """Render ``style``'s selected sections, in its order, skipping any
    section whose result is absent from ``results`` (rather than raising).

    ``results`` keys are section ids (see
    :data:`lqtpy.report.style.VALID_SECTIONS`); values are the record/dict
    shape each corresponding fragment expects -- except ``"parcel_damage"``,
    which needs an atlas name alongside its records, so its value is either
    a single ``{"records": [...], "atlas_name": "..."}`` dict or a list of
    such dicts (to render more than one atlas).
    """
    style = style or DEFAULT_STYLE

    def _parcel_damage(value):
        entries = value if isinstance(value, list) else [value]
        return "".join(
            parcel_damage_fragment(e["records"], e["atlas_name"], style) for e in entries)

    renderers = {
        "morphometry": lambda v: morphometry_fragment(v, style),
        "parcel_damage": _parcel_damage,
        "network_rollup": lambda v: network_rollup_fragment(v, style),
        "disconnection": lambda v: disconnection_fragment(v, style),
        "tract_proxy": lambda v: tract_proxy_fragment(v, style),
    }

    parts = []
    for section_id in style.sections:
        value = results.get(section_id)
        if value is None:
            continue
        parts.append(renderers[section_id](value))
    return "".join(parts)
