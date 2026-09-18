"""Lobe-wise overlap (``RegionMeta.Lobe``, ``lqtpy.overlap.lobe_rollup``,
``lqtpy.api.lobe_overlap``).

No patient data: every lesion here is a synthetic sphere built with numpy
(see ``tests/conftest.py``).
"""
from __future__ import annotations

import pytest

from lqtpy import api
from lqtpy.atlases import load_atlas
from lqtpy.errors import NoLobeMappingError
from lqtpy.io import load_lesion
from lqtpy.overlap import atlas_has_lobes, lobe_rollup, region_overlap

#: The lobe set documented in data/aal_labels.csv's header comment (see also
#: the module docstring in lqtpy/atlases.py).
AAL_LOBES = {"Frontal", "Parietal", "Temporal", "Occipital", "Limbic",
            "Insula", "Subcortical", "Cerebellum"}

#: The (larger) lobe set documented in data/dkt31_cma_labels.csv's header --
#: AAL_LOBES plus the CMA-only Ventricle/Brainstem catch-alls.
DK_LOBES = AAL_LOBES | {"Ventricle", "Brainstem"}


# --------------------------------------------------------------------------- #
# Every AAL region has a non-empty, documented lobe.
# --------------------------------------------------------------------------- #
def test_every_aal_region_has_a_lobe():
    atlas = load_atlas("aal")
    assert len(atlas.labels) == 116
    missing = [m.RegionName for m in atlas.labels.values() if not m.Lobe]
    assert missing == [], f"AAL regions with no lobe: {missing}"
    unknown = {m.Lobe for m in atlas.labels.values()} - AAL_LOBES
    assert unknown == set(), f"AAL regions with an undocumented lobe: {unknown}"


def test_every_dk_region_has_a_lobe():
    atlas = load_atlas("dk")
    assert len(atlas.labels) == 96
    missing = [m.RegionName for m in atlas.labels.values() if not m.Lobe]
    assert missing == [], f"DK regions with no lobe: {missing}"
    unknown = {m.Lobe for m in atlas.labels.values()} - DK_LOBES
    assert unknown == set(), f"DK regions with an undocumented lobe: {unknown}"


def test_schaefer_has_no_lobe_mapping():
    """schaefer only has Yeo-7 networks -- no lobe mapping is defined."""
    atlas = load_atlas("schaefer")
    assert not atlas_has_lobes(atlas)
    assert all(m.Lobe is None for m in atlas.labels.values())


# --------------------------------------------------------------------------- #
# lobe_overlap/lobe_rollup consistency with atlas_overlap/region_overlap.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("atlas_key", ["aal", "dk"])
def test_lobe_lesion_voxels_sum_matches_region_lesion_voxels(
        atlas_key, cortical_lesion_img):
    region_records = api.atlas_overlap(cortical_lesion_img, atlas_key)
    lobe_records = api.lobe_overlap(cortical_lesion_img, atlas_key)
    region_total = sum(r["LesionVoxels"] for r in region_records)
    lobe_total = sum(r["LesionVoxels"] for r in lobe_records)
    assert lobe_total == region_total


@pytest.mark.parametrize("atlas_key", ["aal", "dk"])
def test_lobe_overlap_records_sorted_by_percent_damage_desc(atlas_key, cortical_lesion_img):
    records = api.lobe_overlap(cortical_lesion_img, atlas_key)
    percents = [r["PercentDamage"] for r in records]
    assert percents == sorted(percents, reverse=True)


# --------------------------------------------------------------------------- #
# A lesion placed in one lobe reports that lobe dominant.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("atlas_key", ["aal", "dk"])
def test_lesion_in_frontal_cortex_reports_frontal_dominant(atlas_key, cortical_lesion_img):
    # cortical_lesion_img (see conftest.py) sits over right precentral
    # gyrus -- squarely Frontal in both the aal and dk lobe mappings.
    records = api.lobe_overlap(cortical_lesion_img, atlas_key)
    assert records, "expected at least one damaged lobe"
    top = records[0]
    assert top["Lobe"] == "Frontal"
    # Frontal should carry the large majority of the lesion, not just a
    # narrow plurality, given where the fixture is actually placed.
    assert top["PercentOfLesion"] > 50.0


# --------------------------------------------------------------------------- #
# An atlas with no lobe mapping raises the typed error.
# --------------------------------------------------------------------------- #
def test_lobe_rollup_raises_for_atlas_without_lobes(sphere_lesion_img):
    atlas = load_atlas("schaefer")
    lesion_img, _ = load_lesion(sphere_lesion_img, threshold=0.5)
    overlap = region_overlap(lesion_img, atlas)
    with pytest.raises(NoLobeMappingError):
        lobe_rollup(overlap, atlas)


def test_lobe_overlap_raises_for_atlas_without_lobes(sphere_lesion_img):
    with pytest.raises(NoLobeMappingError):
        api.lobe_overlap(sphere_lesion_img, "schaefer")
