"""Real per-tract streamline disconnection (`lqtpy.api.tract_disconnection`).

Everything here is synthetic -- a tiny made-up tractogram built directly in
code with :func:`lqtpy._discon_index.build_index`, the SAME function
``tools/build_discon_index.py`` uses for the real HCP-842 atlas. No trk
files, no patient data, nothing under ``Z:\\`` is touched by this file.
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading

import numpy as np
import nibabel as nib
import pytest

from lqtpy import api, datasets
from lqtpy._discon_index import (build_index, gather_streamline_ids,
                                 load_index, save_index,
                                 voxel_indices_for_points)
from lqtpy.errors import DataUnavailableError, InvalidLesionError

# --------------------------------------------------------------------------- #
# Synthetic tractogram: identity affine (world == voxel), small grid.
# --------------------------------------------------------------------------- #
SHAPE = (16, 16, 16)
AFFINE = np.eye(4)


def _line(y, z, x0=1.0, x1=14.0, n=24):
    """A straight streamline along x, at fixed (y, z), sampled at `n` points
    (points, not a densified path -- but enough of them that a lesion box
    anywhere along the line actually contains one)."""
    return np.stack([np.linspace(x0, x1, n),
                     np.full(n, float(y)),
                     np.full(n, float(z))], axis=1)


def _arc(y0, z0, radius=3.0, n=24):
    """A curved streamline (quarter-circle arc) in the y-z plane at a fixed
    x range, used for tract TB so it's geometrically distinct from TA's
    straight lines."""
    theta = np.linspace(0, np.pi / 2, n)
    y = y0 + radius * np.cos(theta)
    z = z0 + radius * np.sin(theta)
    x = np.linspace(2.0, 13.0, n)
    return np.stack([x, y, z], axis=1)


def _build_synthetic_tractogram():
    """Two tracts:

    - TA: 5 straight streamlines along x, at y = 3..7, z = 3 (all in the
      z=3 plane, distinguished by y).
    - TB: 4 curved streamlines, centred well away from TA's z=3 plane (in
      z = 9..12), so no TA/TB streamline ever shares a voxel.
    """
    ta_streamlines = [_line(y=y, z=3.0) for y in range(3, 8)]
    tb_streamlines = [_arc(y0=9.0 + i, z0=9.0) for i in range(4)]
    tract_streamlines = [("TA", ta_streamlines), ("TB", tb_streamlines)]
    return tract_streamlines, ta_streamlines, tb_streamlines


@pytest.fixture(scope="module")
def synthetic():
    tract_streamlines, ta, tb = _build_synthetic_tractogram()
    index = build_index(tract_streamlines, SHAPE, AFFINE)
    return {
        "tract_streamlines": tract_streamlines,
        "ta": ta,
        "tb": tb,
        "index": index,
    }


@pytest.fixture
def saved_index_path(tmp_path, synthetic):
    path = tmp_path / "discon_index_v1.npz"
    save_index(str(path), synthetic["index"])
    return str(path)


def _lesion_img_from_mask(mask, affine=AFFINE):
    return nib.Nifti1Image(mask.astype(np.uint8), affine)


def _voxel_set_for_streamline(pts):
    vox = voxel_indices_for_points(pts, AFFINE, SHAPE)
    return {tuple(v) for v in vox.tolist()}


# --------------------------------------------------------------------------- #
# Core percentage semantics.
# --------------------------------------------------------------------------- #
def test_whole_tract_lesion_gives_100_percent(synthetic, saved_index_path):
    mask = np.zeros(SHAPE, dtype=bool)
    for pts in synthetic["ta"]:
        for v in voxel_indices_for_points(pts, AFFINE, SHAPE):
            mask[tuple(v)] = True
    lesion = _lesion_img_from_mask(mask)

    records = api.tract_disconnection(lesion, index_path=saved_index_path)
    by_code = {r["tract"]: r for r in records}

    assert by_code["TA"]["percent_disconnected"] == 100.0
    assert by_code["TA"]["n_disconnected"] == by_code["TA"]["n_streamlines"] == 5
    assert by_code["TB"]["percent_disconnected"] == 0.0
    assert by_code["TB"]["n_disconnected"] == 0


def test_lesion_missing_everything_gives_zero_for_all(synthetic, saved_index_path):
    # A single voxel far from every TA/TB streamline (verified below).
    mask = np.zeros(SHAPE, dtype=bool)
    mask[15, 0, 0] = True
    all_touched = set()
    for _code, streamlines in synthetic["tract_streamlines"]:
        for pts in streamlines:
            all_touched |= _voxel_set_for_streamline(pts)
    assert (15, 0, 0) not in all_touched

    lesion = _lesion_img_from_mask(mask)
    records = api.tract_disconnection(lesion, index_path=saved_index_path)
    assert len(records) == 2
    for r in records:
        assert r["percent_disconnected"] == 0.0
        assert r["n_disconnected"] == 0


def test_lesion_touching_exactly_k_of_n_streamlines(synthetic, saved_index_path):
    # TA streamlines live at y = 3..7 (z=3). A box covering y in [3, 5]
    # touches exactly the k=3 streamlines with y in {3, 4, 5}, out of n=5.
    mask = np.zeros(SHAPE, dtype=bool)
    mask[4:7, 3:6, 3] = True
    lesion = _lesion_img_from_mask(mask)

    records = api.tract_disconnection(lesion, index_path=saved_index_path)
    by_code = {r["tract"]: r for r in records}
    assert by_code["TA"]["n_disconnected"] == 3
    assert by_code["TA"]["n_streamlines"] == 5
    assert by_code["TA"]["percent_disconnected"] == pytest.approx(100.0 * 3 / 5, abs=1e-6)


def test_records_are_sorted_by_tract_code_and_include_zero_percent(synthetic, saved_index_path):
    mask = np.zeros(SHAPE, dtype=bool)
    mask[15, 0, 0] = True
    lesion = _lesion_img_from_mask(mask)
    records = api.tract_disconnection(lesion, index_path=saved_index_path)
    codes = [r["tract"] for r in records]
    assert codes == sorted(codes)
    assert {"tract", "name", "n_streamlines", "n_disconnected", "percent_disconnected"} <= set(records[0])


# --------------------------------------------------------------------------- #
# Equivalence with dipy.tracking.utils.target().
# --------------------------------------------------------------------------- #
def test_matches_dipy_target_on_random_lesions(synthetic):
    dipy_utils = pytest.importorskip("dipy.tracking.utils")

    tract_streamlines = synthetic["tract_streamlines"]
    index = synthetic["index"]
    # Flat ordered list of (streamline_points) in the same global-id order
    # build_index assigns (tract order, then within-tract order).
    all_streamlines = [pts for _code, sls in tract_streamlines for pts in sls]
    id_by_identity = {id(s): i for i, s in enumerate(all_streamlines)}

    rng = np.random.default_rng(12345)
    for _trial in range(20):
        mask = rng.random(SHAPE) < 0.03  # sparse random lesion
        if not mask.any():
            continue
        lesion_keys = np.flatnonzero(mask.ravel())

        ours = set(gather_streamline_ids(lesion_keys, index).tolist())

        dipy_selected = list(dipy_utils.target(all_streamlines, affine=AFFINE,
                                               target_mask=mask, include=True))
        theirs = {id_by_identity[id(s)] for s in dipy_selected}

        assert ours == theirs


# --------------------------------------------------------------------------- #
# Half-voxel tie: floor(x + 0.5) (dipy's convention), not np.round.
# --------------------------------------------------------------------------- #
def test_half_voxel_tie_matches_dipy_round_half_up():
    dipy_utils = pytest.importorskip("dipy.tracking.utils")

    # A point exactly on a .5 voxel boundary in this identity-affine grid.
    point = np.array([[4.5, 4.5, 4.5]])
    ours = voxel_indices_for_points(point, AFFINE, SHAPE)
    assert ours.shape == (1, 3)
    # floor(4.5 + 0.5) = 5, for every axis -- NOT np.round(4.5) == 4
    # (round-half-to-even).
    assert tuple(ours[0]) == (5, 5, 5)
    assert tuple(np.round(point[0]).astype(int)) == (4, 4, 4)  # the wrong answer

    streamline = [point[0]]
    mask_ours = np.zeros(SHAPE, dtype=bool)
    mask_ours[5, 5, 5] = True
    mask_theirs_wrong = np.zeros(SHAPE, dtype=bool)
    mask_theirs_wrong[4, 4, 4] = True

    selected_at_5 = list(dipy_utils.target([streamline], affine=AFFINE,
                                           target_mask=mask_ours, include=True))
    selected_at_4 = list(dipy_utils.target([streamline], affine=AFFINE,
                                           target_mask=mask_theirs_wrong, include=True))
    assert len(selected_at_5) == 1  # dipy also resolves the tie to voxel 5, not 4
    assert len(selected_at_4) == 0


# --------------------------------------------------------------------------- #
# Errors: empty lesion, missing index.
# --------------------------------------------------------------------------- #
def test_empty_lesion_raises_invalid_lesion_error(saved_index_path):
    mask = np.zeros(SHAPE, dtype=np.uint8)
    lesion = _lesion_img_from_mask(mask)
    with pytest.raises(InvalidLesionError):
        api.tract_disconnection(lesion, index_path=saved_index_path)


def test_missing_index_raises_data_unavailable_error_with_actionable_message(
        tmp_path, monkeypatch):
    monkeypatch.setenv("LQTPY_DATA_HOME", str(tmp_path / "empty_data_home"))
    datasets.discon_index_cache_clear()

    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[5, 5, 5] = 1
    lesion = _lesion_img_from_mask(mask)

    with pytest.raises(DataUnavailableError) as excinfo:
        api.tract_disconnection(lesion)
    message = str(excinfo.value)
    assert "build_discon_index.py" in message
    assert "LQTPY_DATA_HOME" in message
    datasets.discon_index_cache_clear()


# --------------------------------------------------------------------------- #
# import lqtpy.api + a disconnection query must not import dipy/matplotlib/
# nilearn/pandas.
# --------------------------------------------------------------------------- #
def test_disconnection_query_does_not_import_heavy_deps(saved_index_path):
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import nibabel as nib\n"
        "import lqtpy.api as api\n"
        f"index_path = {saved_index_path!r}\n"
        "mask = np.zeros((16, 16, 16), dtype=np.uint8)\n"
        "mask[4:7, 3:6, 3] = 1\n"
        "img = nib.Nifti1Image(mask, np.eye(4))\n"
        "records = api.tract_disconnection(img, index_path=index_path)\n"
        "assert isinstance(records, list) and len(records) == 2, records\n"
        "bad = sorted(m for m in ('dipy', 'matplotlib', 'nilearn', 'pandas') "
        "if m in sys.modules)\n"
        "assert bad == [], bad\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------- #
# Thread safety: concurrent first calls load the index exactly once.
# --------------------------------------------------------------------------- #
def test_concurrent_first_calls_load_index_once(tmp_path, synthetic, monkeypatch):
    data_home = tmp_path / "data_home"
    data_home.mkdir()
    save_index(str(data_home / "discon_index_v1.npz"), synthetic["index"])
    monkeypatch.setenv("LQTPY_DATA_HOME", str(data_home))
    datasets.discon_index_cache_clear()

    mask = np.zeros(SHAPE, dtype=np.uint8)
    mask[4:7, 3:6, 3] = 1
    lesion = _lesion_img_from_mask(mask)

    n_threads = 16
    results = [None] * n_threads
    errors = []
    barrier = threading.Barrier(n_threads)

    def worker(i):
        try:
            barrier.wait(timeout=30)
            api.tract_disconnection(lesion)
            results[i] = datasets._INDEX_CACHE
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)

    assert not errors, errors
    ids = {id(r) for r in results if r is not None}
    assert len(ids) == 1, "disconnection index was built more than once under concurrency"

    datasets.discon_index_cache_clear()


# --------------------------------------------------------------------------- #
# load_index() format-version check, and index arrays are frozen once cached.
# --------------------------------------------------------------------------- #
def test_full_and_occupied_layouts_agree(synthetic):
    """gather_streamline_ids must give identical results whether the index
    was stored in the (usually smaller) 'occupied' layout or the 'full'
    layout -- both are built from the same underlying `indices` array, just
    addressed differently. The real HCP-842 build always picks 'occupied'
    (measured smaller), so this reconstructs a 'full' index from the same
    payload to exercise that branch too."""
    index = synthetic["index"]
    assert index["layout"] == "occupied"  # sanity: this synthetic case picks occupied

    shape = index["shape"]
    n_voxels = int(np.prod(shape))
    voxel_keys = index["voxel_keys"].astype(np.int64)
    indptr_occ = index["indptr"].astype(np.int64)
    counts = np.diff(indptr_occ)
    indptr_full = np.zeros(n_voxels + 1, dtype=np.int64)
    indptr_full[voxel_keys + 1] = counts
    indptr_full = np.cumsum(indptr_full)
    full_index = {"layout": "full", "indptr": indptr_full, "indices": index["indices"]}

    rng = np.random.default_rng(999)
    for _ in range(10):
        mask = rng.random(SHAPE) < 0.05
        lesion_keys = np.flatnonzero(mask.ravel())
        occ_result = set(gather_streamline_ids(lesion_keys, index).tolist())
        full_result = set(gather_streamline_ids(lesion_keys, full_index).tolist())
        assert occ_result == full_result


def test_load_index_roundtrip(saved_index_path, synthetic):
    loaded = load_index(saved_index_path)
    assert loaded["shape"] == SHAPE
    assert sorted(loaded["tract_codes"]) == ["TA", "TB"]
    assert loaded["n_streamlines"] == synthetic["index"]["n_streamlines"]


def test_cached_index_arrays_are_read_only(tmp_path, synthetic, monkeypatch):
    data_home = tmp_path / "data_home2"
    data_home.mkdir()
    save_index(str(data_home / "discon_index_v1.npz"), synthetic["index"])
    monkeypatch.setenv("LQTPY_DATA_HOME", str(data_home))
    datasets.discon_index_cache_clear()

    index = datasets._load_discon_index()
    assert index["indices"].flags.writeable is False
    with pytest.raises(ValueError):
        index["indices"][0] = 0

    datasets.discon_index_cache_clear()
