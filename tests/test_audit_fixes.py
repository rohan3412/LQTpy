"""Regression tests for the 2026-09-25 audit findings (docs/ROADMAP.md, Phase 0).

Each test failed before its fix.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

import lqtpy
from lqtpy import _discon_index as di
from lqtpy import api
from lqtpy.errors import GeometryError, IndexFormatError

MNI_AFF = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])


def _cortical_mask():
    data = np.zeros((182, 218, 182), dtype=np.uint8)
    data[120:132, 130:142, 80:92] = 1
    return data


def test_bool_array_is_accepted():
    mask = _cortical_mask()
    assert (api.lesion_stats((mask.astype(bool), MNI_AFF))["n_voxels"]
            == api.lesion_stats((mask, MNI_AFF))["n_voxels"])


def test_singular_affine_still_geometry_error():
    with pytest.raises(GeometryError):
        api.lesion_stats((_cortical_mask().astype(bool), np.zeros((4, 4))))


def test_group_accepts_a_single_file(tmp_path):
    path = tmp_path / "one.nii.gz"
    nib.save(nib.Nifti1Image(_cortical_mask(), MNI_AFF), str(path))
    for src in (str(path), path):
        res = lqtpy.analyze_group(src, atlases=["aal"], tracts=False, write=False)
        assert res["n_subjects"] == 1
        assert res["subjects"][0]["name"] == "one"


def test_atlases_as_bare_string():
    res = lqtpy.analyze_single((_cortical_mask(), MNI_AFF), atlases="aal",
                               tracts=False, write=False)
    assert [a["key"] for a in res["atlases"]] == ["aal"]


def test_group_report_escapes_region_names(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    pytest.importorskip("pandas")
    shape = (40, 40, 40)
    aff = np.diag([2.0, 2, 2, 1]); aff[:3, 3] = -40
    lab = np.zeros(shape, np.int16); lab[:20] = 1; lab[20:] = 2
    evil = '<img src=x onerror=alert(1)>'
    lqtpy.atlas_from_arrays("xss_test", "XSS", lab, aff, {1: evil, 2: "b"}, overwrite=True)
    try:
        les = np.zeros(shape, np.uint8); les[5:10, 18:22, 18:22] = 1
        res = lqtpy.analyze_group([("s1", (les, aff))], atlases=["xss_test"],
                                  tracts=False, plots=True, out_dir=str(tmp_path))
        html = open(res["report_path"], encoding="utf-8").read()
        assert evil not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html
    finally:
        lqtpy.atlases._CUSTOM.pop("xss_test", None)
        lqtpy.atlases._CACHE.pop("xss_test", None)


@pytest.mark.parametrize("value", [
    "rgb(0;}</style><script>alert`1`</script>)",
    "rgb(0,0,0);}body{display:none",
    "hsl(1 2 3 / 'x')",
])
def test_colour_validator_rejects_injection(value):
    from lqtpy.report import ReportStyle
    with pytest.raises(ValueError):
        ReportStyle(accent=value)


@pytest.mark.parametrize("value", [
    "#fff", "#c1443c", "rgb(10, 20, 30)", "rgba(10,20,30,.5)",
    "hsl(210deg 40% 50% / 0.8)", "oklch(62% 0.19 25)", "transparent", "currentColor",
])
def test_colour_validator_accepts_real_colours(value):
    from lqtpy.report import ReportStyle
    ReportStyle(accent=value)


def test_index_is_pickle_free(tmp_path):
    idx = di.build_index([("A", [np.array([[0.0, 0, 0], [1, 1, 1]])])], di.TRACT_SHAPE, di.TRACT_AFFINE)
    path = tmp_path / "idx.npz"
    di.save_index(str(path), idx)
    with np.load(path, allow_pickle=False) as npz:  # would raise if any object array
        assert npz["tract_codes"].dtype.kind == "U"
    assert di.load_index(str(path))["tract_codes"] == ["A"]


def test_old_or_foreign_index_is_typed_error(tmp_path):
    old = tmp_path / "old.npz"
    np.savez(old, format_version=1)
    with pytest.raises(IndexFormatError):
        di.load_index(str(old))
    junk = tmp_path / "junk.npz"
    np.savez(junk, something=np.arange(3))
    with pytest.raises(IndexFormatError):
        di.load_index(str(junk))
    pickled = tmp_path / "pickled.npz"
    np.savez(pickled, format_version=di.FORMAT_VERSION, tract_codes=np.array(["a"], dtype=object),
             shape=np.array([2, 2, 2]), layout="full")
    with pytest.raises(IndexFormatError):
        di.load_index(str(pickled))


def test_empty_index_query_returns_nothing():
    empty = di.build_index([("X", [])], di.TRACT_SHAPE, di.TRACT_AFFINE)
    assert di.gather_streamline_ids(np.array([5, 9]), empty).size == 0


def test_tract_proxy_index_loads_without_pickle():
    with np.load(lqtpy.tracts._TRACT_NPZ, allow_pickle=False) as npz:
        assert npz["names"].dtype.kind == "U"
