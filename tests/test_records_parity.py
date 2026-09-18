"""Structural-invariant checks for the records-based (post-pandas) API.

These don't need the external golden-file comparison done once during the
pandas removal -- they just pin down invariants that must hold regardless of
which synthetic fixture is used: column order, row-ordering rules, and row
count per atlas. Downstream consumers (including MRLatte) parse the CSVs
these records feed, so column order and sort order are load-bearing.
"""
from __future__ import annotations

import pytest

import lqtpy
from lqtpy.atlases import load_atlas
from lqtpy.overlap import (region_overlap, network_rollup, homolog_symmetry,
                           _COLUMNS as _DAMAGE_COLUMNS,
                           _NETWORK_COLUMNS, _HOMOLOG_COLUMNS)
from lqtpy.tracts import tract_overlap, tracts_available, _COLUMNS as _TRACT_COLUMNS


# --------------------------------------------------------------------------- #
# Atlas.labels: row count per atlas
# --------------------------------------------------------------------------- #
def test_aal_label_count():
    assert len(load_atlas("aal").labels) == 116


def test_schaefer_label_count():
    assert len(load_atlas("schaefer").labels) == 100


def test_dk_label_count():
    n = len(load_atlas("dk").labels)
    # 62 cortical DKT31 regions (hemisphere-doubled) + 34 CMA subcortical/
    # ventricle/cerebellar structures = 96 (Mindboggle-101 joint-fusion atlas).
    assert 50 <= n <= 120


def test_atlas_labels_is_plain_dict_of_region_meta():
    atlas = load_atlas("aal")
    assert isinstance(atlas.labels, dict)
    lid, meta = next(iter(atlas.labels.items()))
    assert isinstance(lid, int)
    assert hasattr(meta, "RegionName") and hasattr(meta, "Group")


# --------------------------------------------------------------------------- #
# region_overlap: column order + descending PercentDamage sort
# --------------------------------------------------------------------------- #
def test_region_overlap_records_shape_and_order(sphere_lesion_img):
    atlas = load_atlas("aal")
    records = region_overlap(sphere_lesion_img, atlas)
    assert isinstance(records, list)
    assert records, "sphere fixture should overlap at least one AAL region"
    for r in records:
        assert list(r.keys()) == _DAMAGE_COLUMNS
    damages = [r["PercentDamage"] for r in records]
    assert damages == sorted(damages, reverse=True)
    # Row count is bounded by the number of labelled regions in the atlas.
    assert len(records) <= len(atlas.labels)


def test_region_overlap_frame_pandas_matches_records(sphere_lesion_img):
    pd = pytest.importorskip("pandas")
    atlas = load_atlas("aal")
    records = region_overlap(sphere_lesion_img, atlas)
    df = region_overlap(sphere_lesion_img, atlas, frame="pandas")
    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == _DAMAGE_COLUMNS
    assert df.to_dict("records") == records


# --------------------------------------------------------------------------- #
# network_rollup / homolog_symmetry: column order + sort order
# --------------------------------------------------------------------------- #
def test_network_rollup_records_shape_and_order(two_blob_lesion_img):
    # None of the synthetic fixtures land on the (cortex-only) Schaefer
    # atlas's ribbon -- they're all small subcortical-space blobs -- so this
    # only pins down column order and sort order, not non-emptiness.
    atlas = load_atlas("schaefer")
    overlap = region_overlap(two_blob_lesion_img, atlas)
    net = network_rollup(overlap, atlas)
    for r in net:
        assert list(r.keys()) == _NETWORK_COLUMNS
    damages = [r["PercentDamage"] for r in net]
    assert damages == sorted(damages, reverse=True)


def test_homolog_symmetry_records_shape_and_order(two_blob_lesion_img):
    atlas = load_atlas("aal")
    overlap = region_overlap(two_blob_lesion_img, atlas)
    homolog = homolog_symmetry(overlap)
    assert homolog, "two-blob fixture should produce at least one homolog pair"
    for r in homolog:
        assert list(r.keys()) == _HOMOLOG_COLUMNS
    maxes = [max(r["LeftPercent"], r["RightPercent"]) for r in homolog]
    assert maxes == sorted(maxes, reverse=True)


# --------------------------------------------------------------------------- #
# tract_overlap: column order + descending PercentOverlap sort
# --------------------------------------------------------------------------- #
def test_tract_overlap_records_shape_and_order(sphere_lesion_img):
    if not tracts_available():
        pytest.skip("tract index not bundled in this checkout")
    records = tract_overlap(sphere_lesion_img)
    for r in records:
        assert list(r.keys()) == _TRACT_COLUMNS
    overlaps = [r["PercentOverlap"] for r in records]
    assert overlaps == sorted(overlaps, reverse=True)


# --------------------------------------------------------------------------- #
# analyze_single / analyze_group: records by default, frame="pandas" opt-in.
#
# analyze_single/analyze_group unconditionally generate plots (nilearn +
# matplotlib, plus the third-party `ggseg` package for the aal/dk flat maps)
# regardless of `write` -- pre-existing behaviour, unchanged by this wave.
# These tests exercise the full pipeline, so they need those extras and skip
# cleanly (not error) when they're absent, matching the core-only CI job.
# --------------------------------------------------------------------------- #
def _require_plot_stack():
    pytest.importorskip("nilearn")
    pytest.importorskip("matplotlib")
    pytest.importorskip("ggseg")


def test_analyze_single_returns_records_by_default(write_nifti, sphere_lesion_img):
    _require_plot_stack()
    path = write_nifti(sphere_lesion_img)
    result = lqtpy.analyze_single(path, atlases=["aal"], write=False, verbose=False)
    entry = result["atlases"][0]
    assert isinstance(entry["df"], list)
    if entry["df"]:
        assert list(entry["df"][0].keys()) == _DAMAGE_COLUMNS


def test_analyze_single_frame_pandas(write_nifti, sphere_lesion_img):
    _require_plot_stack()
    pd = pytest.importorskip("pandas")
    path = write_nifti(sphere_lesion_img)
    result = lqtpy.analyze_single(path, atlases=["aal"], write=False,
                                  verbose=False, frame="pandas")
    entry = result["atlases"][0]
    assert isinstance(entry["df"], pd.DataFrame)
    assert list(entry["df"].columns) == _DAMAGE_COLUMNS


def test_analyze_group_region_agg_sort_and_columns(write_nifti, sphere_lesion_img,
                                                    two_blob_lesion_img):
    _require_plot_stack()
    paths = [("a", write_nifti(sphere_lesion_img, "a.nii.gz")),
             ("b", write_nifti(two_blob_lesion_img, "b.nii.gz"))]
    result = lqtpy.analyze_group(paths, atlases=["aal"], write=False, verbose=False)
    entry = result["atlases"][0]
    assert isinstance(entry["df"], list)
    records = entry["df"]
    if records:
        assert list(records[0].keys()) == [
            "RegionName", "Group", "NSubjects", "MeanPercentDamage", "MaxPercentDamage"]
        keys = [(-r["NSubjects"], -r["MeanPercentDamage"]) for r in records]
        assert keys == sorted(keys)
