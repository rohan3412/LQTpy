"""Core index construction/query primitives for real per-tract streamline
disconnection.

This module is deliberately tiny and dependency-light (``numpy`` only -- no
``nibabel``, no ``dipy``) so it can be:

- imported by :mod:`lqtpy.datasets` / :mod:`lqtpy.api` at *query* time with
  no extra cost, and
- imported by ``tools/build_discon_index.py`` (the real HCP-842 build) *and*
  by ``tests/test_disconnection.py`` (a synthetic tractogram), so the test
  suite exercises the exact code that builds the real index rather than a
  reimplementation of it.

Method
------
A streamline counts as disconnected by a lesion if **any** of its points
falls in a lesion voxel -- whole-streamline, binary, no partial credit and
no minimum-overlap-length (this matches
``dipy.tracking.utils.target(..., include=True)`` semantics on the
streamline's raw points, not its densified path). See
``lqtpy/api.py::tract_disconnection`` for the full method description and
citation.
"""

################################################################################
# THE ~120 MM TRAP -- READ THIS BEFORE TOUCHING ANYTHING BELOW.              #
################################################################################
# The HCP-842 population-atlas .trk.gz bundles store `vox_to_ras =
# diag(-1,-1,1)` with **ZERO translation** in their TRK header. That header
# is what `nibabel.streamlines.load(...).streamlines` (or the "obvious"
# `load_tractogram(path, reference='same', to_space=Space.RASMM)`) uses to
# place points in "world" space -- so trusting it puts every streamline
# about **120 mm** away from where it actually belongs relative to the
# (157, 189, 136) HCP-842 tractography atlas grid this package ships
# (`lqtpy/data/hcp842_tracts.npz` / `TRACT_AFFINE` below). A lesion mask
# resampled onto that grid and intersected against "world" points read the
# naive way will silently select ZERO streamlines for every bundle -- not an
# error, not a crash, just quietly wrong output that looks plausible until
# someone notices every tract reads 0% disconnected.
#
# The validated fix (see the geometry study this was built from -- 62/70
# bundles, median Dice 0.78 in the correct frame, Dice EXACTLY 0.0 in the
# naive frame and both axis-flip controls, for every single tract):
#
#     world_correct = raw_points_from_trk + TRACT_ATLAS_ORIGIN_SHIFT
#     voxel         = inv(TRACT_AFFINE) @ world_correct
#
# Algebraically this is identical to just trusting the TRK's own internal
# voxel grid and ignoring its (wrong) translation entirely -- i.e. the TRK
# files were rasterized directly onto the atlas's own (157,189,136) grid,
# and only the header metadata lost the offset. `TRACT_ATLAS_ORIGIN_SHIFT`
# below is exactly the translation column of `TRACT_AFFINE`.
#
# DO NOT "SIMPLIFY" THIS BY CALLING `load_tractogram(path, reference='same',
# to_space=Space.RASMM)` AND USING ITS OUTPUT DIRECTLY. That call is the
# single most likely way for a future edit to silently reintroduce this bug:
# it runs without error, returns a plausible-looking set of 3D points, and
# is off by ~120 mm. Always go through `voxel_indices_for_points` below,
# which encodes the correction, and never trust a TRK-derived affine for
# this atlas.
################################################################################

from __future__ import annotations

import numpy as np

#: On-disk/format version for the .npz produced by build_index/save_index.
#: Bump this whenever the array layout or its semantics change, and teach
#: load_index to either handle old versions or raise a clear error.
#: v2: tract codes stored as a fixed-width unicode array, so the file loads
#: with ``allow_pickle=False`` (v1 used an object array, which needs pickle
#: and would let a crafted .npz run code on load).
FORMAT_VERSION = 2

#: The validated frame for the HCP-842 population tractography atlas (see
#: the module docstring's trap warning above). The trk files' internal
#: voxel grid *is* this atlas's own grid; only the translation needs adding
#: back in before any voxel lookup.
TRACT_AFFINE = np.array([
    [-1.0, 0.0, 0.0, 78.0],
    [0.0, -1.0, 0.0, 76.0],
    [0.0, 0.0, 1.0, -50.0],
    [0.0, 0.0, 0.0, 1.0],
])
TRACT_SHAPE = (157, 189, 136)
#: Translation column of TRACT_AFFINE, applied to a TRK's raw (translation-
#: less) points before the affine inverse -- see the trap warning above.
TRACT_ATLAS_ORIGIN_SHIFT = np.array([78.0, 76.0, -50.0])


def _smallest_uint_dtype(max_value: int) -> np.dtype:
    """Smallest unsigned integer numpy dtype that can hold ``max_value``."""
    max_value = max(int(max_value), 0)
    for dt in (np.uint8, np.uint16, np.uint32, np.uint64):
        if max_value <= np.iinfo(dt).max:
            return np.dtype(dt)
    return np.dtype(np.uint64)


def voxel_indices_for_points(points_world: np.ndarray, affine: np.ndarray,
                             shape) -> np.ndarray:
    """Map an ``(N, 3)`` array of WORLD-space points to in-grid voxel
    indices, using DIPY's own rounding convention.

    ``dipy.tracking.utils.target()`` maps a point to a voxel via
    ``floor(inv(affine) @ point + 0.5)`` (see
    ``dipy.tracking.utils._to_voxel_coordinates``) -- i.e. **round-half-up**,
    not ``np.round``'s round-half-to-even. The two conventions only disagree
    on a point that lands exactly on a ``.5`` voxel boundary, but when they
    disagree this is the one that matches dipy -- see
    ``tests/test_disconnection.py``'s half-voxel tie test, which pins this
    down with a point constructed to land exactly on ``.5``.

    Returns an ``(M, 3)`` int64 array, ``M <= N`` (out-of-grid points are
    dropped, matching ``target()``'s own behaviour of never selecting a
    point outside the target mask's grid).
    """
    points_world = np.asarray(points_world, dtype=np.float64)
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise ValueError(f"points_world must be (N, 3), got shape {points_world.shape}")
    inv = np.linalg.inv(np.asarray(affine, dtype=np.float64))
    ones = np.ones((points_world.shape[0], 1))
    homogeneous = np.hstack([points_world, ones])
    vox = (inv @ homogeneous.T).T[:, :3]
    idx = np.floor(vox + 0.5).astype(np.int64)  # round-half-up, matches dipy
    shape_arr = np.asarray(shape, dtype=np.int64)
    valid = np.all((idx >= 0) & (idx < shape_arr), axis=1)
    return idx[valid]


def build_index(tract_streamlines, shape, affine):
    """Build the voxel -> streamline-id disconnection index.

    Parameters
    ----------
    tract_streamlines : list of (tract_code, list[ndarray (N_i, 3)])
        One entry per tract, in the tract-code order the caller wants
        (this becomes the tract id order stored in the index). Each
        streamline's points must already be in the SAME world space
        ``affine`` maps voxel indices into -- for the real HCP-842 data
        that means ``raw_trk_points + TRACT_ATLAS_ORIGIN_SHIFT`` (see the
        module docstring's trap warning); callers building a synthetic
        tractogram for tests can just place points directly in that space.
    shape : 3-tuple of int
        The reference grid shape (e.g. ``TRACT_SHAPE``).
    affine : (4, 4) array
        Voxel-to-world affine for ``shape`` (e.g. ``TRACT_AFFINE``).

    Returns
    -------
    dict with keys:
        format_version, shape, affine, tract_codes (list[str]),
        tract_ref_counts (per-tract streamline count, the fixed
        denominator), streamline_tract (per-streamline tract id),
        n_streamlines, indices (CSR streamline-id payload, shared by both
        layouts below), and, chosen by whichever measured smaller:
          - layout='full':     indptr sized n_voxels_total + 1.
          - layout='occupied': indptr sized n_occupied + 1, plus
                                voxel_keys (sorted occupied flat voxel
                                indices).
        Also carries '_full_bytes'/'_occupied_bytes' (measured sizes of
        each candidate layout, for reporting -- not needed at query time).
    """
    shape = tuple(int(s) for s in shape)
    n_voxels_total = int(np.prod(shape))
    affine = np.asarray(affine, dtype=np.float64)

    tract_codes = [code for code, _ in tract_streamlines]
    if len(set(tract_codes)) != len(tract_codes):
        raise ValueError(f"Duplicate tract codes in tract_streamlines: {tract_codes}")

    streamline_tract_list = []
    voxel_key_parts = []
    sid_parts = []
    sid = 0
    for tid, (_code, streamlines) in enumerate(tract_streamlines):
        for pts in streamlines:
            vox = voxel_indices_for_points(np.asarray(pts), affine, shape)
            if vox.shape[0]:
                keys = np.ravel_multi_index((vox[:, 0], vox[:, 1], vox[:, 2]), shape)
                keys = np.unique(keys)  # dedupe within-streamline revisits
                voxel_key_parts.append(keys)
                sid_parts.append(np.full(keys.shape[0], sid, dtype=np.int64))
            streamline_tract_list.append(tid)
            sid += 1

    n_streamlines = sid
    ref_counts = np.bincount(np.asarray(streamline_tract_list, dtype=np.int64),
                             minlength=len(tract_codes))

    if voxel_key_parts:
        all_keys = np.concatenate(voxel_key_parts)
        all_sids = np.concatenate(sid_parts)
    else:
        all_keys = np.empty(0, dtype=np.int64)
        all_sids = np.empty(0, dtype=np.int64)

    order = np.argsort(all_keys, kind="stable")
    sorted_keys = all_keys[order]
    sorted_sids = all_sids[order]

    occupied_keys, counts = np.unique(sorted_keys, return_counts=True)
    n_occupied = int(occupied_keys.shape[0])
    nnz = int(sorted_sids.shape[0])

    sid_dtype = _smallest_uint_dtype(max(n_streamlines - 1, 0))
    indices = sorted_sids.astype(sid_dtype)

    indptr_dtype = _smallest_uint_dtype(nnz)
    indptr_occupied = np.zeros(n_occupied + 1, dtype=np.int64)
    indptr_occupied[1:] = np.cumsum(counts)
    indptr_occupied = indptr_occupied.astype(indptr_dtype)

    key_dtype = _smallest_uint_dtype(n_voxels_total - 1)
    voxel_keys = occupied_keys.astype(key_dtype)

    # Full-grid layout shares the identical `indices` payload (both are
    # built from the same ascending-voxel-key sort); only indptr differs.
    indptr_full = np.zeros(n_voxels_total + 1, dtype=np.int64)
    indptr_full[occupied_keys + 1] = counts
    indptr_full = np.cumsum(indptr_full).astype(indptr_dtype)

    occupied_bytes = (voxel_keys.nbytes + indptr_occupied.nbytes + indices.nbytes)
    full_bytes = (indptr_full.nbytes + indices.nbytes)

    tract_id_dtype = _smallest_uint_dtype(max(len(tract_codes) - 1, 0))
    streamline_tract = np.asarray(streamline_tract_list, dtype=tract_id_dtype)

    result = {
        "format_version": FORMAT_VERSION,
        "shape": np.asarray(shape, dtype=np.int32),
        "affine": affine,
        "tract_codes": np.asarray([str(c) for c in tract_codes]),
        "tract_ref_counts": ref_counts.astype(np.int32),
        "streamline_tract": streamline_tract,
        "n_streamlines": n_streamlines,
        "indices": indices,
        "_full_bytes": full_bytes,
        "_occupied_bytes": occupied_bytes,
    }
    if occupied_bytes <= full_bytes:
        result["layout"] = "occupied"
        result["indptr"] = indptr_occupied
        result["voxel_keys"] = voxel_keys
    else:
        result["layout"] = "full"
        result["indptr"] = indptr_full
    return result


def gather_streamline_ids(lesion_voxel_keys: np.ndarray, index: dict) -> np.ndarray:
    """Union of streamline ids touching any voxel in ``lesion_voxel_keys``.

    ``lesion_voxel_keys`` is a 1-D array of flat voxel indices (into the
    index's ``shape``, C order, matching ``np.ravel_multi_index``) -- need
    not be sorted or unique. Returns a sorted array of unique streamline
    ids (``np.unique``), vectorized (no per-voxel Python loop) so this
    stays fast for lesion masks with many thousands of voxels.
    """
    lesion_voxel_keys = np.asarray(lesion_voxel_keys, dtype=np.int64)
    indptr = index["indptr"]
    indices = index["indices"]

    if index["layout"] == "full":
        keys = lesion_voxel_keys
    else:
        voxel_keys = index["voxel_keys"]
        if voxel_keys.shape[0] == 0:
            return np.empty(0, dtype=indices.dtype)
        pos = np.searchsorted(voxel_keys, lesion_voxel_keys)
        pos = np.clip(pos, 0, voxel_keys.shape[0] - 1)
        found = voxel_keys[pos] == lesion_voxel_keys
        keys = pos[found]

    if keys.shape[0] == 0:
        return np.empty(0, dtype=indices.dtype)

    starts = indptr[keys].astype(np.int64)
    ends = indptr[keys + 1].astype(np.int64)
    counts = ends - starts
    total = int(counts.sum())
    if total == 0:
        return np.empty(0, dtype=indices.dtype)

    # Vectorized ragged gather: for each voxel with `c` hits starting at
    # `s`, emit s, s+1, ..., s+c-1, without a Python-level loop over voxels.
    cum = np.cumsum(counts)
    group_start_in_output = cum - counts
    output_pos = np.arange(total)
    # which voxel-group does each output slot belong to
    group_of_slot = np.repeat(np.arange(keys.shape[0]), counts)
    offset_within_group = output_pos - np.repeat(group_start_in_output, counts)
    gather_at = starts[group_of_slot] + offset_within_group

    hits = indices[gather_at]
    return np.unique(hits)


def save_index(path: str, index: dict, *, compress: bool = True) -> None:
    """Save a ``build_index()`` result to ``path`` as ``.npz``."""
    payload = {k: v for k, v in index.items() if not k.startswith("_")}
    saver = np.savez_compressed if compress else np.savez
    saver(path, **payload)


def load_index(path: str) -> dict:
    """Load and validate an index built by :func:`save_index`.

    Loads with ``allow_pickle=False``: an index is plain numeric/unicode
    arrays, so nothing in a well-formed file needs unpickling, and refusing
    pickle means a crafted file can't execute code. Raises
    :class:`~lqtpy.errors.IndexFormatError` for an older format or a file
    that isn't an index.

    Returned arrays are NOT made read-only here -- callers that cache the
    result (see ``lqtpy/datasets.py``) are responsible for that, consistent
    with how ``lqtpy/atlases.py`` freezes its own cached arrays.
    """
    from .errors import IndexFormatError

    try:
        with np.load(path, allow_pickle=False) as npz:
            if "format_version" not in npz.files:
                raise IndexFormatError(
                    f"'{path}' is not a lqtpy disconnection index (no format_version).")
            version = int(npz["format_version"])
            if version != FORMAT_VERSION:
                raise IndexFormatError(
                    f"Disconnection index at '{path}' has format_version={version}, "
                    f"this lqtpy expects {FORMAT_VERSION}. Rebuild it with "
                    "tools/build_discon_index.py."
                )
            out = {k: npz[k] for k in npz.files}
    except IndexFormatError:
        raise
    except FileNotFoundError:
        raise
    except (ValueError, OSError, KeyError) as exc:
        raise IndexFormatError(
            f"Could not read disconnection index at '{path}': {exc}") from exc
    out["shape"] = tuple(int(s) for s in out["shape"])
    out["tract_codes"] = [str(c) for c in out["tract_codes"]]
    out["layout"] = str(out["layout"])
    out["format_version"] = version
    out["n_streamlines"] = int(out["n_streamlines"])
    return out
