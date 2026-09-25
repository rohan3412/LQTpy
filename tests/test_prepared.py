"""PreparedLesion: load-once semantics, caching, and resampling parity."""
from __future__ import annotations

import threading

import numpy as np
import nibabel as nib
import pytest

import lqtpy
from lqtpy import _discon_index as di
from lqtpy.io import load_lesion, resample_mask_to
from lqtpy.prepared import prepare

import synthetic as S


def _grid(v, shift=0.0):
    sh = tuple(int(np.ceil(s / v)) for s in (182, 218, 182))
    a = np.diag([-v, v, v, 1.0])
    a[:3, 3] = [90 + shift, -126 + shift, -72]
    return sh, a


def _sphere_on(sh, a, c, r):
    w = S._world_grid(sh, a)
    return (np.linalg.norm(w - np.asarray(c, float), axis=-1) <= r).astype(np.uint8)


_ROT = np.deg2rad(17)
_OBL = np.eye(4)
_OBL[:3, :3] = np.array([[np.cos(_ROT), -np.sin(_ROT), 0], [np.sin(_ROT), np.cos(_ROT), 0],
                         [0, 0, 1]]) * 1.1
_OBL[:3, 3] = -_OBL[:3, :3] @ np.array([80, 100, 80.0])

GRIDS = {"1mm": _grid(1), "2mm": _grid(2), "2mm half-voxel": _grid(2, 0.5),
         "3mm": _grid(3), "0.8mm": _grid(0.8), "oblique": ((160, 200, 160), _OBL)}


@pytest.mark.parametrize("grid", list(GRIDS))
@pytest.mark.parametrize("center,r", [([-40, 10, 5], 15), ([85, -20, 0], 12), ([-60, -30, 20], 40)])
def test_on_grid_matches_full_resample(grid, center, r):
    """Cropped, global-coordinate nearest-neighbour == the vendored nilearn
    resampler, on every realistic grid (including lesions cut by the edge)."""
    sh, a = GRIDS[grid]
    m = _sphere_on(sh, a, center, r)
    if not m.any():
        pytest.skip("sphere outside this grid")
    img = nib.Nifti1Image(m, a)
    p = prepare(img)
    old, _ = load_lesion(img)
    refs = [lqtpy.load_atlas(k).img for k in ("aal", "dk")]
    refs.append(nib.Nifti1Image(np.zeros(di.TRACT_SHAPE, np.uint8), di.TRACT_AFFINE))
    for ref in refs:
        assert np.array_equal(p.on_grid(ref).mask, resample_mask_to(old, ref))


def test_prepare_is_idempotent_and_keeps_threshold():
    img = S.img(S.sphere([-40, 0, 10], 8))
    p = prepare(img, 0.3)
    assert prepare(p) is p
    assert lqtpy.atlas_overlap(p, "aal", threshold=0.9) == lqtpy.atlas_overlap(img, "aal", threshold=0.3)


def test_arrays_are_read_only():
    p = prepare(S.img(S.sphere([-40, 0, 10], 8)))
    with pytest.raises(ValueError):
        p.mask[0, 0, 0] = True
    with pytest.raises(ValueError):
        p.on_grid(lqtpy.load_atlas("aal").img).mask[0, 0, 0] = True


def test_nonfinite_treated_as_zero_and_counted():
    data = S.sphere([-40, 0, 10], 8).astype(np.float32)
    ref = int(data.sum())
    data[0, 0, 0] = np.nan
    data[1, 1, 1] = np.inf
    p = prepare(S.img(data) if False else nib.Nifti1Image(data, S.MNI_AFFINE))
    assert p.n_voxels == ref and p.n_nonfinite == 2


def test_outside_fov_is_counted_not_dropped():
    big_shape = (260, 300, 260)
    big = np.diag([-1.0, 1, 1, 1])
    big[:3, 3] = [130, -170, -110]
    m = _sphere_on(big_shape, big, [-40, 10, 5], 15) | _sphere_on(big_shape, big, [120, 0, 0], 8)
    cov = lqtpy.atlas_coverage(nib.Nifti1Image(m, big), "aal")
    assert cov["OutsideFOVVoxels"] > 0
    assert 0 < cov["PercentOutsideFOV"] < 20


def test_concurrent_use_of_one_prepared_lesion():
    p = prepare(S.img(S.sphere([-40, 10, 5], 15)))
    ref = lqtpy.atlas_overlap(p, "dk")
    out, errors = [], []

    def work():
        try:
            out.append(lqtpy.atlas_overlap(p, "dk"))
            lqtpy.lesion_stats(p)
        except Exception as e:  # pragma: no cover
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors and all(o == ref for o in out)


def test_midline_share_makes_hemispheres_add_up():
    s = lqtpy.lesion_stats(S.img(S.sphere([0, 0, 10], 5)))
    assert s["pct_left"] + s["pct_right"] + s["pct_midline"] == pytest.approx(100, abs=0.2)
