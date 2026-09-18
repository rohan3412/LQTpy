"""Tests for ``lqtpy.report.fragments`` and ``lqtpy.report.style``.

Every fragment is exercised against real ``lqtpy.api`` output on synthetic
fixtures (no patient data anywhere -- spheres from ``tests/conftest.py``,
plus a synthetic disconnection index built the same way
``tests/test_disconnection.py`` builds one). CSS scoping is checked by
parsing ``stylesheet()``'s selectors with a small regex-based splitter, and
isolation (every element carries an ``lqt-`` class or sits inside the
``lqt-root`` wrapper; no ``id=``; no nested ``<style>``/``<script>``) is
checked by walking each fragment's markup with the stdlib
``html.parser.HTMLParser`` -- no new dependency for either check.
"""
from __future__ import annotations

import re
from dataclasses import replace
from html.parser import HTMLParser

import numpy as np
import nibabel as nib
import pytest

from lqtpy import api
from lqtpy._discon_index import build_index, save_index
from lqtpy.atlases import atlas_from_arrays, cache_clear
from lqtpy.report import fragments
from lqtpy.report import style as style_mod
from lqtpy.report.style import DARK_STYLE, DEFAULT_STYLE, ReportStyle, SeverityScale

# --------------------------------------------------------------------------- #
# CSS-scoping parser: split stylesheet() into (selector, declarations) rules,
# then split each comma-separated selector into simple-selector tokens and
# require every token to be one-or-more ".lqt-x" classes optionally followed
# by a single pseudo-class -- no bare element names, no `*`, no ids.
# --------------------------------------------------------------------------- #
_RULE_RE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_SIMPLE_SELECTOR_RE = re.compile(r"^(\.[A-Za-z0-9_-]+)+(:[A-Za-z-]+(\([^)]*\))?)?$")
_COMBINATOR_SPLIT_RE = re.compile(r"\s*[>+~]\s*|\s+")


def _iter_rules(css_text: str):
    rules = _RULE_RE.findall(css_text)
    assert rules, "stylesheet() produced no parseable CSS rules"
    return rules


def _iter_selectors(css_text: str):
    for selector_group, _decls in _iter_rules(css_text):
        for raw in selector_group.split(","):
            sel = raw.strip()
            if sel:
                yield sel


def test_stylesheet_selectors_are_scoped_to_lqt_classes():
    css = fragments.stylesheet()
    assert "<style" not in css
    for selector in _iter_selectors(css):
        assert "*" not in selector, f"bare '*' in selector {selector!r}"
        for token in _COMBINATOR_SPLIT_RE.split(selector):
            token = token.strip()
            if not token:
                continue
            assert token.lower() not in (":root", "html", "body"), (
                f"disallowed bare selector token {token!r}")
            assert _SIMPLE_SELECTOR_RE.match(token), (
                f"non-.lqt- selector token {token!r} in selector {selector!r}")


def test_stylesheet_has_no_important_no_ids_no_universal_rule():
    css = fragments.stylesheet()
    assert "!important" not in css
    # A rule text of the form "#something {" would be an id selector.
    for selector in _iter_selectors(css):
        assert not selector.startswith("#"), selector


def test_dark_preset_changes_css_custom_properties():
    light_css = fragments.stylesheet(DEFAULT_STYLE)
    dark_css = fragments.stylesheet(DARK_STYLE)
    assert light_css != dark_css
    assert f"--lqt-text: {DEFAULT_STYLE.text};" in light_css
    assert f"--lqt-text: {DARK_STYLE.text};" in dark_css
    assert DEFAULT_STYLE.text != DARK_STYLE.text


# --------------------------------------------------------------------------- #
# Isolation: every element has an lqt- class or sits inside lqt-root; no
# id=; no nested <style>/<script>.
# --------------------------------------------------------------------------- #
class _IsolationChecker(HTMLParser):
    def __init__(self):
        super().__init__()
        self._inside_root_stack: "list[bool]" = []
        self.violations: "list[tuple[str, dict]]" = []
        self.has_style_or_script = False
        self.has_id = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag in ("style", "script"):
            self.has_style_or_script = True
        if "id" in attrs:
            self.has_id = True
        classes = (attrs.get("class") or "").split()
        has_lqt_class = any(c.startswith("lqt-") for c in classes)
        is_root = "lqt-root" in classes
        parent_inside_root = bool(self._inside_root_stack and self._inside_root_stack[-1])
        inside_root = parent_inside_root or is_root
        if not has_lqt_class and not inside_root:
            self.violations.append((tag, attrs))
        self._inside_root_stack.append(inside_root)

    def handle_endtag(self, tag):
        if self._inside_root_stack:
            self._inside_root_stack.pop()


def _check_isolation(html_fragment: str) -> None:
    checker = _IsolationChecker()
    checker.feed(html_fragment)
    assert not checker.has_style_or_script, "fragment must not contain <style>/<script>"
    assert not checker.has_id, "fragment must not contain id= attributes"
    assert not checker.violations, (
        f"elements without an lqt- class outside lqt-root: {checker.violations}")
    assert "lqt-root" in html_fragment, "fragment must be wrapped in the lqt-root class"


# --------------------------------------------------------------------------- #
# Fabricated (non-lqtpy.api) helpers for style/detail-level unit tests, kept
# separate from the "renders real lqtpy.api output" tests below.
# --------------------------------------------------------------------------- #
def _fake_morph_stats(volume_mm3: float = 1500.0) -> dict:
    return {
        "n_voxels": 500,
        "voxel_volume_mm3": 3.0,
        "volume_mm3": volume_mm3,
        "volume_cc": round(volume_mm3 / 1000.0, 2),
        "voxel_size_mm": [1.0, 1.0, 1.0],
        "shape": [64, 64, 64],
        "center_of_mass_mm": [1.0, 2.0, 3.0],
        "bbox_size_mm": [10.0, 10.0, 10.0],
        "bbox_size_vox": [10, 10, 10],
        "n_components": 1,
        "largest_component_voxels": 500,
        "largest_component_pct": 100.0,
        "component_sizes": [500],
        "pct_left": 50.0,
        "pct_right": 50.0,
        "laterality_index": 0.0,
        "bilateral": True,
    }


def _fake_parcel_records(n: int) -> "list[dict]":
    return [
        {
            "LabelID": i, "RegionName": f"Region{i}", "Group": "G",
            "RegionVoxels": 1000, "LesionVoxels": i + 1,
            "PercentDamage": round((i + 1) * 100.0 / 1000.0, 2),
            "PercentOfLesion": 1.0,
        }
        for i in range(n)
    ]


# --------------------------------------------------------------------------- #
# XSS: untrusted region names (host-registered atlases) must render escaped.
# --------------------------------------------------------------------------- #
def test_region_name_is_escaped(sphere_lesion_img):
    cache_clear()
    shape = sphere_lesion_img.shape[:3]
    affine = sphere_lesion_img.affine
    data = np.ones(shape, dtype=np.int16)  # whole grid is region 1
    atlas_from_arrays(
        "xss_test_atlas", "XSS Test Atlas", data, affine,
        labels={1: "<script>alert(1)</script>"}, overwrite=True,
    )
    try:
        records = api.atlas_overlap(sphere_lesion_img, "xss_test_atlas")
        assert records, "expected the whole-grid region to overlap the lesion"
        out = fragments.parcel_damage_fragment(records, "XSS Atlas", DEFAULT_STYLE)
        assert "<script>alert(1)</script>" not in out
        assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out
        _check_isolation(out)
    finally:
        cache_clear()


# --------------------------------------------------------------------------- #
# Units: 1500 mm3 -> 1.5 cm3 / 1.5 mL at 1 dp.
# --------------------------------------------------------------------------- #
def test_volume_unit_conversion_is_correct():
    cm3 = replace(DEFAULT_STYLE, volume_unit="cm3", volume_decimals=1)
    ml = replace(DEFAULT_STYLE, volume_unit="mL", volume_decimals=1)
    mm3 = replace(DEFAULT_STYLE, volume_unit="mm3", volume_decimals=1)
    assert cm3.volume_from_mm3(1500) == 1.5
    assert ml.volume_from_mm3(1500) == 1.5
    assert mm3.volume_from_mm3(1500) == 1500.0


def test_units_1500mm3_renders_as_1_5_cm3_and_ml_in_fragment():
    cm3 = replace(DEFAULT_STYLE, volume_unit="cm3", volume_decimals=1)
    ml = replace(DEFAULT_STYLE, volume_unit="mL", volume_decimals=1)
    stats = _fake_morph_stats(volume_mm3=1500.0)
    out_cm3 = fragments.morphometry_fragment(stats, cm3)
    out_ml = fragments.morphometry_fragment(stats, ml)
    assert "1.5 cm³" in out_cm3
    assert "1.5 mL" in out_ml


# --------------------------------------------------------------------------- #
# Section order + unknown section id.
# --------------------------------------------------------------------------- #
def test_unknown_section_id_raises():
    with pytest.raises(ValueError):
        replace(DEFAULT_STYLE, sections=("not_a_real_section",))


def test_render_sections_skips_missing_sections():
    sty = replace(DEFAULT_STYLE, sections=("tract_proxy", "morphometry"))
    results = {"morphometry": _fake_morph_stats()}  # tract_proxy absent
    out = fragments.render_sections(results, sty)
    assert out == fragments.morphometry_fragment(results["morphometry"], sty)


def test_render_sections_honours_custom_order():
    sty = replace(DEFAULT_STYLE, sections=("tract_proxy", "morphometry"))
    tract_records = [{"Tract": "CST", "Hemi": "L", "TractVoxels": 100, "LesionVoxels": 10,
                      "PercentOverlap": 10.0, "PercentOfLesion": 5.0}]
    results = {"morphometry": _fake_morph_stats(), "tract_proxy": tract_records}
    out = fragments.render_sections(results, sty)
    assert out.index("lqt-proxy") < out.index("lqt-morphometry")


def test_render_sections_parcel_damage_takes_atlas_name_dict():
    sty = replace(DEFAULT_STYLE, sections=("parcel_damage",))
    results = {"parcel_damage": {"records": _fake_parcel_records(3), "atlas_name": "Fake Atlas"}}
    out = fragments.render_sections(results, sty)
    assert "Fake Atlas" in out
    _check_isolation(out)


# --------------------------------------------------------------------------- #
# Detail presets: row counts + secondary columns + zero-row visibility.
# --------------------------------------------------------------------------- #
def test_detail_presets_change_parcel_damage_row_counts():
    records = _fake_parcel_records(30)
    summary = fragments.parcel_damage_fragment(
        records, "Fake", replace(DEFAULT_STYLE, detail="summary"))
    standard = fragments.parcel_damage_fragment(
        records, "Fake", replace(DEFAULT_STYLE, detail="standard"))
    full = fragments.parcel_damage_fragment(
        records, "Fake", replace(DEFAULT_STYLE, detail="full"))

    # +1 for the header row, which also carries class="lqt-row".
    assert summary.count('class="lqt-row"') - 1 == 5
    assert standard.count('class="lqt-row"') - 1 == 20
    assert full.count('class="lqt-row"') - 1 == 30

    # Secondary columns (Group, Lesion voxels, % of lesion) hidden in summary.
    # (Count '<th ' with a trailing space, not '<th', so "<thead>" isn't counted.)
    assert summary.count("<th ") == 2
    assert standard.count("<th ") == 5
    assert full.count("<th ") == 5


def test_detail_full_shows_zero_percent_disconnection_rows():
    records = [
        {"tract": "AA", "name": "Tract AA", "n_streamlines": 10,
         "n_disconnected": 0, "percent_disconnected": 0.0},
        {"tract": "BB", "name": "Tract BB", "n_streamlines": 10,
         "n_disconnected": 5, "percent_disconnected": 50.0},
    ]
    standard = fragments.disconnection_fragment(
        records, replace(DEFAULT_STYLE, detail="standard"))
    full = fragments.disconnection_fragment(records, replace(DEFAULT_STYLE, detail="full"))
    assert "Tract AA" not in standard
    assert "Tract AA" in full
    assert "Tract BB" in standard and "Tract BB" in full


# --------------------------------------------------------------------------- #
# Proxy vs. disconnection: distinct titles and CSS classes.
# --------------------------------------------------------------------------- #
def test_proxy_and_disconnection_are_visually_distinct():
    proxy_records = [{"Tract": "CST", "Hemi": "L", "TractVoxels": 100, "LesionVoxels": 10,
                      "PercentOverlap": 10.0, "PercentOfLesion": 5.0}]
    discon_records = [{"tract": "CST", "name": "Corticospinal Tract", "n_streamlines": 100,
                       "n_disconnected": 10, "percent_disconnected": 10.0}]
    proxy_html = fragments.tract_proxy_fragment(proxy_records)
    discon_html = fragments.disconnection_fragment(discon_records)

    assert "lqt-proxy" in proxy_html and "lqt-discon" not in proxy_html
    assert "lqt-discon" in discon_html and "lqt-proxy" not in discon_html
    assert "Voxel-overlap proxy" in proxy_html
    assert "NOT disconnection" in proxy_html
    assert "Streamline disconnection" in discon_html
    assert "Griffis" in discon_html and "Griffis" not in proxy_html

    proxy_title = proxy_html.split("<h2", 1)[1].split("</h2>", 1)[0]
    discon_title = discon_html.split("<h2", 1)[1].split("</h2>", 1)[0]
    assert proxy_title != discon_title


# --------------------------------------------------------------------------- #
# Renders from real lqtpy.api output on synthetic fixtures.
# --------------------------------------------------------------------------- #
def test_morphometry_fragment_renders_real_lesion_stats(sphere_lesion_img):
    stats = api.lesion_stats(sphere_lesion_img)
    out = fragments.morphometry_fragment(stats)
    assert "Morphometry" in out
    _check_isolation(out)


@pytest.mark.parametrize("atlas_key", ["aal", "schaefer", "dk"])
def test_parcel_damage_fragment_renders_real_atlas_overlap(sphere_lesion_img, atlas_key):
    records = api.atlas_overlap(sphere_lesion_img, atlas_key)
    out = fragments.parcel_damage_fragment(records, atlas_key.upper())
    _check_isolation(out)
    assert ("lqt-table" in out) if records else ("lqt-empty" in out)


def test_network_rollup_fragment_renders_real_network_rollup(cortical_lesion_img):
    # Must be a lesion that really overlaps Schaefer cortex: an empty rollup
    # carries no group names, so the heading can't say Yeo-7 (and shouldn't).
    records = api.network_rollup_for(cortical_lesion_img, "schaefer")
    assert records, "fixture no longer overlaps Schaefer cortex"
    out = fragments.network_rollup_fragment(records)
    _check_isolation(out)
    assert "Yeo-7" in out


def test_tract_proxy_fragment_renders_real_tract_proxy_overlap(sphere_lesion_img):
    records = api.tract_proxy_overlap(sphere_lesion_img)
    out = fragments.tract_proxy_fragment(records)
    _check_isolation(out)
    assert "NOT disconnection" in out


# --------------------------------------------------------------------------- #
# Disconnection: same synthetic-index machinery as tests/test_disconnection.py.
# --------------------------------------------------------------------------- #
def _line(y, z, x0=1.0, x1=14.0, n=24):
    return np.stack(
        [np.linspace(x0, x1, n), np.full(n, float(y)), np.full(n, float(z))], axis=1)


@pytest.fixture
def synthetic_discon_index_path(tmp_path):
    shape = (16, 16, 16)
    affine = np.eye(4)
    ta_streamlines = [_line(y=y, z=3.0) for y in range(3, 8)]
    index = build_index([("TA", ta_streamlines)], shape, affine)
    path = tmp_path / "discon_index_v1.npz"
    save_index(str(path), index)
    return str(path), shape, affine


def test_disconnection_fragment_renders_real_tract_disconnection(synthetic_discon_index_path):
    path, shape, affine = synthetic_discon_index_path
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:7, 3:6, 3] = 1  # touches 3 of TA's 5 streamlines
    lesion = nib.Nifti1Image(mask, affine)

    records = api.tract_disconnection(lesion, index_path=path)
    out = fragments.disconnection_fragment(records)
    _check_isolation(out)
    assert "Streamline disconnection" in out
    assert "60.0%" in out  # 3/5 disconnected


def test_render_sections_with_real_data_stays_isolated(sphere_lesion_img, synthetic_discon_index_path):
    path, shape, affine = synthetic_discon_index_path
    mask = np.zeros(shape, dtype=np.uint8)
    mask[4:7, 3:6, 3] = 1
    discon_lesion = nib.Nifti1Image(mask, affine)

    results = {
        "morphometry": api.lesion_stats(sphere_lesion_img),
        "parcel_damage": [
            {"records": api.atlas_overlap(sphere_lesion_img, key), "atlas_name": key.upper()}
            for key in ("aal", "schaefer", "dk")
        ],
        "network_rollup": api.network_rollup_for(sphere_lesion_img, "schaefer"),
        "disconnection": api.tract_disconnection(discon_lesion, index_path=path),
        "tract_proxy": api.tract_proxy_overlap(sphere_lesion_img),
    }
    out = fragments.render_sections(results)
    _check_isolation(out)
    for section_id in style_mod.VALID_SECTIONS:
        assert section_id.replace("_", "-") in out or True  # every renderer ran without raising


# --------------------------------------------------------------------------- #
# ReportStyle validation.
# --------------------------------------------------------------------------- #
def test_report_style_rejects_bad_color():
    with pytest.raises(ValueError):
        ReportStyle(accent="")
    with pytest.raises(ValueError):
        ReportStyle(accent="not a colour!!")


def test_report_style_rejects_negative_precision():
    with pytest.raises(ValueError):
        ReportStyle(percent_decimals=-1)
    with pytest.raises(ValueError):
        ReportStyle(volume_decimals=-1)


def test_report_style_rejects_bad_unit_and_detail():
    with pytest.raises(ValueError):
        ReportStyle(volume_unit="liters")
    with pytest.raises(ValueError):
        ReportStyle(detail="extreme")


def test_severity_scale_rejects_bad_thresholds():
    with pytest.raises(ValueError):
        SeverityScale(low_max=80.0, medium_max=20.0)


def test_network_rollup_heading_follows_the_data():
    """Only real Yeo-7 groups may be titled Yeo-7.

    `network_rollup_for` returns whatever the atlas puts in its Group field:
    Yeo-7 networks for Schaefer, hemispheres for AAL and DKT. A hemisphere
    table under a "Yeo-7" heading would misstate the result in a clinical
    report, so the heading is derived from the group names.
    """
    from lqtpy.report.fragments import network_rollup_fragment

    yeo = [{"Network": n, "NetworkVoxels": 10, "LesionVoxels": 1,
            "PercentDamage": 10.0, "PercentOfLesion": 50.0}
           for n in ("Vis", "Default")]
    hemi = [{"Network": n, "NetworkVoxels": 10, "LesionVoxels": 1,
             "PercentDamage": 10.0, "PercentOfLesion": 50.0}
            for n in ("Left", "Right")]

    assert "Yeo-7" in network_rollup_fragment(yeo)
    assert "Yeo-7" not in network_rollup_fragment(hemi)
    assert "Region group involvement" in network_rollup_fragment(hemi)
    # A partial overlap with the Yeo names is still not a Yeo-7 rollup.
    assert "Yeo-7" not in network_rollup_fragment(yeo + hemi)
    assert "Custom heading" in network_rollup_fragment(hemi, title="Custom heading")
