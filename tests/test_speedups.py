"""The faster code paths give exactly what the code they replaced gave.

Each test compares a fast path with the original computation (kept here as
the reference): morphology helpers, the voxel scan, separable resampling,
the outline-sensitivity redraws, hole filling for qc.fix, the checksum,
tract lookups, the streamline gather and the on-disk cache.
"""
from __future__ import annotations

import hashlib

import numpy as np
import nibabel as nib
import pytest
from scipy import ndimage

from lqtpy import _morph
from lqtpy.prepared import _flatnonzero, prepare

S26 = np.ones((3, 3, 3), bool)


def _random_masks(seed, n=60, max_side=16):
    rng = np.random.default_rng(seed)
    for _ in range(n):
        shape = tuple(int(s) for s in rng.integers(1, max_side, 3))
        if rng.random() < 0.5:
            yield rng.random(shape) < rng.uniform(0.2, 0.9)
        else:
            yield ndimage.binary_dilation(rng.random(shape) < 0.04, iterations=2)


def test_morphology_matches_scipy():
    for m in _random_masks(0):
        assert np.array_equal(_morph.fill_holes(m), ndimage.binary_fill_holes(m))
        for ax in range(3):
            ref = np.zeros_like(m)
            for k in range(m.shape[ax]):
                sl = [slice(None)] * 3
                sl[ax] = k
                ref[tuple(sl)] = ndimage.binary_fill_holes(m[tuple(sl)])
            assert np.array_equal(_morph.fill_holes(m, axis=ax), ref)
        for it in (1, 2):
            assert np.array_equal(_morph.dilate26(m, it), ndimage.binary_dilation(m, S26, iterations=it))
            assert np.array_equal(_morph.erode26(m, it), ndimage.binary_erosion(m, S26, iterations=it))


def test_flatnonzero_matches_numpy():
    rng = np.random.default_rng(1)
    for shape in [(70, 80, 90), (33, 41, 67), (5, 4, 3)]:
        for p in (0.0, 0.0005, 0.05, 1.0):
            m = rng.random(shape) < p
            for a in (m, np.asfortranarray(m), m[:, ::2], m.astype(np.uint8)):
                got, want = _flatnonzero(a), np.flatnonzero(a)
                assert got.dtype == want.dtype and np.array_equal(got, want)


def _nearest_box_reference(mask, src_affine, bbox, ref_shape, ref_affine):
    """The resampling before the axis-aligned fast path (3-D coordinates)."""
    from scipy import linalg
    M = linalg.inv(src_affine) @ ref_affine
    Minv = linalg.inv(M)
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float) - 1
    corners = np.array([[x, y, z] for x in (lo[0] - 1, hi[0] + 1) for y in (lo[1] - 1, hi[1] + 1)
                        for z in (lo[2] - 1, hi[2] + 1)])
    tvox = corners @ Minv[:3, :3].T + Minv[:3, 3]
    tlo = np.maximum(np.floor(tvox.min(0)).astype(int) - 1, 0)
    thi = np.minimum(np.ceil(tvox.max(0)).astype(int) + 2, np.asarray(ref_shape))
    if not np.all(thi > tlo):
        return None, None
    ti, tj, tk = (np.arange(a, b, dtype=float) for a, b in zip(tlo, thi))
    idx, valid = [], None
    for r in range(3):
        c = (M[r, 0] * ti[:, None, None] + M[r, 1] * tj[None, :, None]
             + M[r, 2] * tk[None, None, :] + M[r, 3])
        ok = (c >= 0) & (c <= mask.shape[r] - 1)
        valid = ok if valid is None else (valid & ok)
        idx.append(np.floor(c + 0.5).astype(np.int64))
    sub = np.zeros(valid.shape, dtype=bool)
    sub[valid] = mask[idx[0][valid], idx[1][valid], idx[2][valid]]
    return tuple(int(v) for v in tlo), sub


@pytest.mark.parametrize("src_zoom,ref_zoom,shift", [
    ((1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (0.0, 0.0, 0.0)),
    ((0.74, 0.74, 0.74), (1.0, 1.0, 1.0), (0.37, -0.5, 0.2)),
    ((-1.0, 1.0, 1.0), (2.0, 2.0, 2.0), (1.0, 0.5, -1.5)),
    ((1.0, 1.2, 0.9), (1.0, 1.0, 1.0), (0.5, 0.5, 0.5)),
])
def test_separable_resampling_matches_reference(src_zoom, ref_zoom, shift):
    from lqtpy.prepared import _nearest_box
    rng = np.random.default_rng(2)
    mask = np.zeros((40, 44, 36), bool)
    mask[8:30, 10:33, 5:28] = rng.random((22, 23, 23)) < 0.6
    src = np.diag(list(src_zoom) + [1.0])
    src[:3, 3] = [-20.0, -22.0, -18.0]
    ref = np.diag(list(ref_zoom) + [1.0])
    ref[:3, 3] = np.array([-24.0, -26.0, -20.0]) + np.asarray(shift)
    nz = np.argwhere(mask)
    bbox = (tuple(nz.min(0)), tuple(nz.max(0) + 1))
    got = _nearest_box(mask, src, bbox, (30, 34, 28), ref)
    want = _nearest_box_reference(mask, src, bbox, (30, 34, 28), ref)
    assert got[0] == want[0] and np.array_equal(got[1], want[1])


def _sensitivity_reference(p):
    """The redraws as first written: on the whole padded grid."""
    pad = np.pad(p.mask, 2)
    out = {"eroded": ndimage.binary_erosion(pad), "dilated": ndimage.binary_dilation(pad),
           "smoothed": ndimage.gaussian_filter(pad.astype(np.float32), 1.0) > 0.5}
    return {k: v[2:-2, 2:-2, 2:-2] for k, v in out.items()}


@pytest.mark.parametrize("where", ["middle", "edge"])
def test_sensitivity_redraws_match_whole_grid(where):
    from lqtpy.measures.sensitivity import _variants
    rng = np.random.default_rng(3)
    m = np.zeros((48, 50, 46), np.uint8)
    blob = ndimage.binary_dilation(rng.random((14, 15, 13)) < 0.08, iterations=2)
    if where == "middle":
        m[16:30, 17:32, 15:28] = blob
    else:                                   # touching the grid's edge on two axes
        m[0:14, 35:50, 15:28] = blob
    p = prepare((m, np.diag([2.0, 2.0, 2.0, 1.0])))
    ref = _sensitivity_reference(p)
    for name, (n, lesion) in _variants(p).items():
        assert n == int(ref[name].sum())
        if n:
            assert np.array_equal(lesion.mask, ref[name])
            assert np.array_equal(lesion.voxel_coords(), np.argwhere(ref[name]))


def test_grid_holes_match_whole_grid():
    from lqtpy import qc
    from lqtpy.qc_grade import hole_voxels
    from lqtpy.qc_tools import _grid_holes
    rng = np.random.default_rng(4)
    for case in range(6):
        m = np.zeros((40, 42, 38), bool)
        ball = np.linalg.norm(np.indices((15, 15, 15)) - 7, axis=0) <= 7
        ball &= rng.random(ball.shape) > 0.08       # pinholes and pores
        ball[5:10, 5:10, 5:10] = False              # an enclosed cavity
        lo = (0, 2, 23) if case % 2 else (12, 14, 10)
        m[lo[0]:lo[0] + 15, lo[1]:lo[1] + 15, lo[2]:lo[2] + 15] = ball
        assert np.array_equal(_grid_holes(m, qc.DEFAULTS), hole_voxels(m, qc.DEFAULTS))


def test_checksum_matches_packbits():
    from lqtpy.repro import checksum
    rng = np.random.default_rng(5)
    for _ in range(40):
        shape = tuple(int(s) for s in rng.integers(2, 20, 3))
        m = rng.random(shape) < rng.choice([0.02, 0.4])
        m.flat[0] = True
        p = prepare((m.astype(np.uint8), np.eye(4)))
        h = hashlib.sha256()
        h.update(np.packbits(p.mask).tobytes())
        h.update(np.asarray(p.affine, float).tobytes())
        assert checksum(p) == "mask:" + h.hexdigest()


def test_integer_threshold_matches_float_comparison():
    rng = np.random.default_rng(6)
    data = rng.integers(0, 4, (30, 30, 30)).astype(np.int16)
    for t in (0.5, 1.0, 1.5, 2.9, -0.5):
        p = prepare((data, np.eye(4)), t)
        assert np.array_equal(p.mask, data > t)


def test_tract_lookup_matches_gather():
    from lqtpy import tracts
    if not tracts.tracts_available():
        pytest.skip("no tract index")
    cache = tracts._load_tracts()
    m = np.zeros(cache["shape"], bool)
    m[60:90, 80:110, 60:85] = True
    p = prepare((m.astype(np.uint8), np.asarray(cache["ref"].affine)))
    gm = p.on_grid(cache["ref"])
    flat = gm.mask.reshape(-1)
    for t, pos in zip(cache["tracts"], tracts.covered(gm)):
        assert np.array_equal(pos, np.flatnonzero(flat[t["idx"]]))


def test_streamline_gather_is_np_unique():
    from lqtpy._discon_index import gather_streamline_ids
    rng = np.random.default_rng(7)
    n_vox, n_sl = 500, 300
    counts = rng.integers(0, 6, n_vox)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    indices = rng.integers(0, n_sl, int(counts.sum())).astype(np.uint32)
    index = {"layout": "full", "indptr": indptr, "indices": indices, "n_streamlines": n_sl}
    keys = rng.choice(n_vox, 60, replace=False)
    starts, ends = indptr[keys].astype(int), indptr[keys + 1].astype(int)
    want = np.unique(np.concatenate([indices[a:b] for a, b in zip(starts, ends)]))
    got = gather_streamline_ids(keys, index)
    assert got.dtype == want.dtype and np.array_equal(got, want)


def test_disk_cache_roundtrip_and_old_entries(tmp_path, monkeypatch):
    from lqtpy import _diskcache
    monkeypatch.setenv("LQTPY_CACHE_DIR", str(tmp_path))
    built = []

    def build():
        built.append(1)
        return {"big": np.arange(100_000, dtype=np.int32), "small": np.array(7)}
    a = _diskcache.cached_arrays("demo", build)
    b = _diskcache.cached_arrays("demo", build)
    assert len(built) == 1
    for d in (a, b):
        assert np.array_equal(d["big"], np.arange(100_000)) and int(d["small"]) == 7
        assert not d["big"].flags.writeable and type(d["big"]) is np.ndarray
    # an entry from lqtpy 0.7.0 (one compressed .npz) is converted, not rebuilt
    import os
    np.savez_compressed(os.path.join(_diskcache.cache_dir(), "old.npz"), q=np.ones(5, np.uint8))
    c = _diskcache.cached_arrays("old", build)
    assert len(built) == 1 and np.array_equal(c["q"], np.ones(5))
    monkeypatch.setenv("LQTPY_NO_DISK_CACHE", "1")
    assert np.array_equal(_diskcache.cached_arrays("demo", build)["big"], np.arange(100_000))
    assert len(built) == 2


def test_report_downloads_take_bytes_or_paths(tmp_path):
    from lqtpy.report._standalone import _downloads
    f = tmp_path / "a.csv"
    f.write_bytes(b"x,y\r\n1,2\r\n")
    by_path = _downloads({"a.csv": str(f)})
    by_bytes = _downloads({"a.csv": b"x,y\r\n1,2\r\n"})
    assert by_path == by_bytes and by_path[0]


def test_morphometry_header_zooms_unchanged():
    from lqtpy.morphometry import compute_morphometry
    m = np.zeros((20, 20, 20), np.uint8)
    m[5:9, 5:9, 5:9] = 1
    h = nib.Nifti1Header()
    h.set_data_shape(m.shape)
    h.set_zooms((2.0, 2.0, 2.0))
    p = prepare(nib.Nifti1Image(m, np.eye(4), h))
    # what the full uint8 image's header said, as before
    zooms = np.abs(np.asarray(p.img.header.get_zooms()[:3], float))
    mismatch = not np.allclose(zooms, np.ones(3), rtol=1e-3, atol=1e-3)
    assert compute_morphometry(p)["header_zooms_mismatch"] is mismatch
