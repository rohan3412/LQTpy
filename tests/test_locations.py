"""Border zones, key-structure distances, tract cross-sections, mirror image,
and the wording fixes (midline share, bilateral territory, lacunar size)."""
import numpy as np
import pytest

import lqtpy
import synthetic as syn
from lqtpy import _format as fmt


def test_border_zone_found_and_core_not():
    from lqtpy.watershed import border_zones
    core = border_zones(syn.img(syn.sphere((-45, 5, 10), 10)))
    assert not core["fits"] and core["pct_in_border_zones"] < 20
    aca_mca = border_zones(syn.img(syn.sphere((-22, 14, 45), 6)))
    assert aca_mca["fits"] and aca_mca["zones"][0]["name"] == "ACA-MCA"
    assert aca_mca["zones"][0]["hemisphere"] == "left"


def test_internal_chain_pattern():
    from lqtpy.watershed import border_zones
    chain = syn.img(syn.blobs([((-22, -16, 27), 2.5), ((-22, -6, 27), 2.5), ((-22, 4, 28), 2.5)]))
    b = border_zones(chain)
    assert b["fits"] and b["zones"][0]["kind"] == "internal"


def test_description_mentions_border_zone_only_when_it_fits():
    d = lqtpy.describe(syn.img(syn.sphere((-22, 14, 45), 6)))
    assert "border-zone (watershed) pattern" in d.text
    d2 = lqtpy.describe(syn.img(syn.sphere((-45, 5, 10), 10)))
    assert "watershed" not in d2.text


def test_key_structure_distances():
    from lqtpy.structures import distances
    capsule = syn.img(syn.ellipsoid((-24, -12, 8), (4, 6, 8)))
    rows = {r["Structure"]: r for r in distances(capsule)}
    assert rows["left corticospinal tract"]["Reaches"]
    assert rows["right corticospinal tract"]["DistanceMM"] > 20
    assert all(r["DistanceMM"] >= 0 for r in rows.values())


def test_cross_section_load_exceeds_voxel_overlap():
    from lqtpy.structures import cross_section_load
    capsule = syn.img(syn.ellipsoid((-24, -12, 8), (4, 6, 8)))
    cst = next(r for r in cross_section_load(capsule) if r["Tract"] == "Cortico Spinal Tract (L)")
    assert cst["MaxCrossSectionLoad"] > 5 * cst["PercentOverlap"]


def test_mirror_is_exact_and_homologue_named():
    from lqtpy.mirror import homologue, mirror
    les = syn.img(syn.sphere((-50, 15, 10), 8))
    m = lqtpy.lesion_stats(mirror(les))
    assert m["center_of_mass_mm"][0] == pytest.approx(50, abs=1)
    h = homologue(les)
    assert h["side"] == "right" and h["self_overlap_pct"] < 5
    assert any("pars opercularis" in r["name"] for r in h["regions"])


def test_midline_share_is_never_dropped():
    d = lqtpy.describe(syn.img(syn.sphere((0, -28, -30), 8)))
    summary = dict(d.sections)["Summary"]
    assert "on the midline" in summary and "left" in summary and "right" in summary
    terr = d.facts["territories"][0]
    assert terr["hemisphere"] == "bilateral" and terr["territory"] == "VB"
    assert fmt.territory_short("bilateral", "VB") == "bilateral VB"
    assert "In the brainstem it mainly involves the brainstem" not in d.text


def test_lacunar_needs_small_volume():
    big = lqtpy.describe(syn.img(syn.ellipsoid((-24, 4, 8), (8, 9, 12))))  # 3.6 cm3, 19 mm across
    assert "lacunar" not in big.text
    small = lqtpy.describe(syn.img(syn.ellipsoid((-24, 4, 8), (4, 4, 6))))
    assert "lacunar-sized" in small.text


def test_one_volume_format():
    assert fmt.volume(15250) == "15.2 cm³" or fmt.volume(15250) == "15.3 cm³"
    assert fmt.volume(152_000) == "152 cm³"
    assert fmt.volume(420) == "0.42 cm³"
    assert fmt.volume(35) == "0.035 cm³"
    assert fmt.quality_label("warn") == "Check" and fmt.quality_label(None) == "OK"
    assert fmt.lobe_word("Subcortical") == "deep grey matter"


def test_unnamed_tissue_split_in_facts():
    d = lqtpy.describe(syn.img(syn.blobs([((-40, 10, 10), 14), ((-35, -10, 15), 10)])))
    by_side = d.facts["unlabelled_tissue_by_side"]
    assert "left" in by_side and "white_matter" in by_side["left"]
