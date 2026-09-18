"""Thread safety of the atlas caches (see atlases.py's module docstring).

Concurrent first callers of ``load_atlas``/``preload`` must build each
atlas exactly once and all end up sharing the identical object; the cached
label arrays and per-label voxel counts they build must be read-only.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from lqtpy import api
from lqtpy.atlases import cache_clear, list_atlases, load_atlas, preload

N_THREADS = 16


def _run_threads(worker, n=N_THREADS):
    errors = []

    def guarded(i):
        try:
            worker(i)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors


def test_concurrent_load_atlas_builds_once_and_shares_identity():
    cache_clear()
    keys = list_atlases()
    barrier = threading.Barrier(N_THREADS)
    results = [None] * N_THREADS

    def worker(i):
        barrier.wait(timeout=30)
        results[i] = {k: load_atlas(k) for k in keys}

    _run_threads(worker)

    for k in keys:
        objs = {id(r[k]) for r in results if r is not None}
        assert len(objs) == 1, f"atlas '{k}' was built more than once under concurrency"


def test_concurrent_preload_and_load_atlas_and_api_call(sphere_lesion_img):
    cache_clear()
    keys = list_atlases()
    barrier = threading.Barrier(N_THREADS)
    api_results = [None] * N_THREADS

    def worker(i):
        barrier.wait(timeout=30)
        if i % 3 == 0:
            preload()
        elif i % 3 == 1:
            for k in keys:
                load_atlas(k)
        else:
            api_results[i] = api.atlas_overlap(sphere_lesion_img, "aal")

    _run_threads(worker)

    baseline = api.atlas_overlap(sphere_lesion_img, "aal")
    for r in api_results:
        if r is not None:
            assert r == baseline

    # Every thread that touched an atlas must have converged on the same
    # cached objects.
    ref = {k: load_atlas(k) for k in keys}
    for k in keys:
        assert load_atlas(k) is ref[k]


def test_cached_label_arrays_are_read_only():
    cache_clear()
    for k in list_atlases():
        atlas = load_atlas(k)
        arr = atlas.label_array()
        assert arr.flags.writeable is False
        with pytest.raises(ValueError):
            arr[0, 0, 0] = 999

        counts = atlas.region_voxels()
        assert counts.flags.writeable is False
        with pytest.raises(ValueError):
            counts[0] = -1


def test_cache_clear_forces_rebuild_with_identical_results():
    cache_clear()
    atlas1 = load_atlas("aal")
    arr1 = atlas1.label_array()

    cache_clear()
    atlas2 = load_atlas("aal")
    arr2 = atlas2.label_array()

    assert atlas1 is not atlas2
    assert arr1.dtype == arr2.dtype
    assert np.array_equal(arr1, arr2)


def test_preload_builds_every_atlas_cache():
    cache_clear()
    preload()
    for k in list_atlases():
        atlas = load_atlas(k)
        # already built -- label_array()/region_voxels() must not rebuild.
        assert atlas._label_array is not None
        assert atlas._region_voxels is not None
