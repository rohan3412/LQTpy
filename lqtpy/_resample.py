"""Vendored ``nilearn.image.resample_to_img``.

This file is vendored from MRLatte's ``backend/worker_common.py`` (the
section headed "Vendored nilearn.image.resample_to_img", lines ~107-297 as of
this copy). The two copies are intentionally **not** consolidated into a
shared package yet -- lqtpy and MRLatte are separate repos/projects, so this
is a deliberate duplication rather than an oversight. If a bug is found here,
check whether it also needs fixing in MRLatte's copy (and vice versa).
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

# ── Vendored `nilearn.image.resample_to_img` ────────────────────────────────
# Phase 1b: `nilearn` was pulled into the core install for exactly ONE symbol,
# `resample_to_img`, and it hard-requires pandas + scikit-learn — 127 MB of
# wheels for one 60-line function. The port below is a faithful, line-for-line
# reimplementation of nilearn 0.12's `resample_to_img` -> `resample_img` path
# on top of `scipy.ndimage.affine_transform` + `nibabel`, restricted to the
# behaviour the callers actually use (3D/4D source, 3D target grid, no
# clipping, fill 0).
#
# Fidelity notes — every one of these is load-bearing for bit-identical output
# and was confirmed by backend/tests/test_resample_parity.py (in MRLatte):
#
#   * `scipy.linalg.inv` (NOT `numpy.linalg.inv`) for the source-affine
#     inverse: the two disagree in the last ULP on some affines, and that
#     propagates into voxel-centre coordinates and flips nearest-neighbour
#     ties.
#   * `A = np.diag(A)` when A is diagonal: scipy dispatches a 1-D matrix to
#     `zoom_shift` and a 2-D one to the general geometric transform, which are
#     not bitwise equal.
#   * dtype: preserved, except signed-int + 'continuous' which nilearn casts
#     to the same-width float (int16 -> float32, not float16).
#   * int64/uint64 output is downcast to int32 when it fits, and bool -> uint8
#     — nilearn's `new_img_like` does this, and callers see the dtype.
#   * degenerate case (same affine within `allclose` AND same 3D shape) is a
#     passthrough, matching nilearn's `_resampling_not_needed`.
#
# Deliberate divergence: nilearn extrapolates NaN/inf out of the data before
# resampling (via `nilearn.masking.extrapolate_out_mask`) and restores the NaN
# mask afterwards. No call site here ever passes non-finite data (lesion
# masks, integer atlases, integer density maps), so that path is not ported;
# a non-finite input raises instead of silently taking a different code path.
#
# `force_resample` is accepted and ignored: it selected nilearn's
# pure-translation "padding optimization", which nilearn itself is retiring in
# 0.13 (`force_resample=True` becomes the default) because that shortcut
# mis-places data whenever the source extends past the target origin. This
# port always takes the real resampling path.
_INTERPOLATION_ORDER = {"continuous": 3, "linear": 1, "nearest": 0}


class BoundingBoxError(ValueError):
    """Target grid's field of view contains none of the source data."""


def _to_matrix_vector(transform):
    """Split a homogeneous 4x4 transform into its 3x3 matrix and 3-vector."""
    ndimin = transform.shape[0] - 1
    ndimout = transform.shape[1] - 1
    return transform[0:ndimin, 0:ndimout], transform[0:ndimin, ndimout]


def _get_bounds(shape, affine):
    """World-space bounds of the corner-voxel CENTRES of `shape` under
    `affine`. Only used for the empty-field-of-view check."""
    adim, bdim, cdim = shape
    adim -= 1
    bdim -= 1
    cdim -= 1
    box = np.array([
        [0.0, 0, 0, 1], [adim, 0, 0, 1], [0, bdim, 0, 1], [0, 0, cdim, 1],
        [adim, bdim, 0, 1], [adim, 0, cdim, 1], [0, bdim, cdim, 1],
        [adim, bdim, cdim, 1],
    ]).T
    box = np.dot(affine, box)[:3]
    return list(zip(box.min(axis=-1), box.max(axis=-1)))


def _new_img_like(ref_img, data, affine, copy_header=False):
    """Local port of `nilearn.image.new_img_like` for the NIfTI case."""
    if data.dtype == bool:
        data = data.astype(np.uint8)
    if data.dtype in (np.dtype(np.int64), np.dtype(np.uint64)) and data.size:
        info = np.iinfo(np.int32)
        if info.min <= data.min() and data.max() <= info.max:
            data = data.astype("int32")
    header = None
    if copy_header and ref_img.header is not None:
        header = ref_img.header.copy()
        try:
            "something" in header  # noqa: B015 — probes dict-like headers
        except TypeError:
            pass
        else:
            if "scl_slope" in header:
                header["scl_slope"] = 0.0
            if "scl_inter" in header:
                header["scl_inter"] = 0.0
            if "glmax" in header:
                header["glmax"] = 0.0
            if "cal_max" in header:
                header["cal_max"] = np.max(data) if data.size > 0 else 0.0
            if "cal_min" in header:
                header["cal_min"] = np.min(data) if data.size > 0 else 0.0
    klass = ref_img.__class__
    if klass is nib.Nifti1Pair:
        # Nifti1Pair has no to_filename; nilearn promotes it the same way.
        klass = nib.Nifti1Image
    return klass(data, affine, header=header)


def resample_to_img(source_img, target_img, interpolation="continuous",
                    fill_value=0, clip=False, order="F", copy=True,
                    copy_header=False, force_resample=None):
    """Resample `source_img` onto `target_img`'s voxel grid.

    Drop-in replacement for `nilearn.image.resample_to_img` for 3D/4D NIfTI
    inputs. No registration is performed — the images must already be aligned
    in world space.

    interpolation : "nearest" (order 0), "linear" (order 1) or "continuous"
                    (order 3, the default, matching nilearn).
    fill_value    : value for target voxels falling outside the source FOV.
    clip          : clamp the output into [min(source, 0), max(source, 0)].
    order         : memory layout of the output array ("F" like nilearn).
    copy / force_resample : accepted for signature compatibility; ignored
                    (this implementation never aliases the source data and
                    always takes the real resampling path).
    """
    from scipy import linalg
    from scipy.ndimage import affine_transform

    if interpolation not in _INTERPOLATION_ORDER:
        raise ValueError(
            f"interpolation must be one of {tuple(_INTERPOLATION_ORDER)}.\n"
            f" Got '{interpolation}' instead.")

    affine = np.asarray(source_img.affine)
    source_shape = source_img.shape
    target_affine = np.asarray(target_img.affine)
    # A 4D target only contributes its 3D grid (nilearn does the same).
    target_shape = tuple(int(s) for s in target_img.shape[:3])

    # ── Degenerate case: already on the target grid → passthrough ───────────
    if (np.array_equal(np.asarray(target_shape), np.asarray(source_shape[:3]))
            and np.allclose(target_affine, affine)):
        return source_img

    data = np.asanyarray(source_img.dataobj)
    if data.dtype.kind == "f" and not np.all(np.isfinite(data)):
        raise ValueError(
            "resample_to_img: non-finite values (NaN/inf) in the source image. "
            "The vendored resampler does not implement nilearn's out-of-mask "
            "extrapolation for them.")

    # Empty-field-of-view check, identical to nilearn's.
    bounds = _get_bounds(data.shape[:3],
                         np.linalg.inv(target_affine).dot(affine))
    if bounds[0][1] < 0 or bounds[1][1] < 0 or bounds[2][1] < 0:
        raise BoundingBoxError("The field of view given by the target affine "
                               "does not contain any of the data")

    if np.all(target_affine == affine):
        # More numerically stable than inv(affine) @ affine.
        transform_affine = np.eye(4)
    else:
        transform_affine = np.dot(linalg.inv(affine), target_affine)
    A, b = _to_matrix_vector(transform_affine)

    resampled_dtype = data.dtype
    if interpolation == "continuous" and data.dtype.kind == "i":
        aux = data.dtype.name.replace("int", "float")
        aux = aux.replace("ufloat", "float").replace("floatc", "float")
        if aux in ("float8", "float16"):
            aux = "float32"
        resampled_dtype = np.dtype(aux)

    # A diagonal matrix passed as 1-D lets scipy pick its faster (and
    # numerically distinct) zoom_shift path — nilearn relies on this.
    if np.all(np.diag(np.diag(A)) == A):
        A = np.diag(A)

    other_shape = list(data.shape[3:])
    resampled_data = np.zeros(list(target_shape) + other_shape,
                              order=order, dtype=resampled_dtype)
    all_img = (slice(None),) * 3
    # The interpolation is separable in the trailing dimensions, so a 4D
    # source is resampled one 3D volume at a time.
    for ind in np.ndindex(*other_shape):
        affine_transform(
            data[all_img + ind], A, offset=b, output_shape=target_shape,
            output=resampled_data[all_img + ind], cval=fill_value,
            order=_INTERPOLATION_ORDER[interpolation],
        )

    if clip:
        vmin = min(data.min(), 0) if data.size else 0
        vmax = max(data.max(), 0) if data.size else 0
        resampled_data.clip(vmin, vmax, out=resampled_data)

    return _new_img_like(source_img, resampled_data, target_affine,
                         copy_header=copy_header)
