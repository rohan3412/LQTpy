"""Loading, binarizing, and resampling lesion masks."""
from __future__ import annotations

import os

import numpy as np
import nibabel as nib
from ._resample import resample_to_img
from .errors import GeometryError, InvalidLesionError, LQTpyError


def as_lesion_image(src) -> nib.Nifti1Image:
    """Coerce ``src`` into a nibabel image.

    Accepts a **path** (``str`` or ``os.PathLike``), a **nibabel image**, or
    an **``(array, affine)``** tuple, and returns a nibabel image. Validates:

    - 3D, or 4D with a singleton last axis (squeezed to 3D);
    - a finite, non-singular affine;
    - a numeric dtype.

    Raises :class:`~lqtpy.errors.InvalidLesionError` (unreadable/wrong-shape/
    wrong-dtype) or :class:`~lqtpy.errors.GeometryError` (bad affine) --
    both caller-mistake errors (HTTP 4xx).
    """
    if isinstance(src, (str, os.PathLike)):
        path = os.fspath(src)
        try:
            from .formats import load_image
            img = load_image(path)
        except FileNotFoundError:
            raise
        except LQTpyError:
            raise
        except Exception as exc:
            raise InvalidLesionError(
                f"Could not read lesion image at '{path}': {exc}") from exc
    elif hasattr(src, "affine") and hasattr(src, "get_fdata") and hasattr(src, "shape"):
        img = src
    elif isinstance(src, (tuple, list)) and len(src) == 2:
        array, affine = src
        array = np.asarray(array)
        affine = np.asarray(affine)
        if affine.shape != (4, 4):
            raise InvalidLesionError(
                f"(array, affine) affine must be a 4x4 matrix, got shape {affine.shape}.")
        if array.dtype.kind not in "iufb":
            # Checked before construction: nibabel's own dtype rejection is a
            # generic ValueError indistinguishable from an affine problem.
            raise InvalidLesionError(
                f"Lesion array must have a numeric dtype, got {array.dtype}.")
        if array.dtype == bool:
            # nibabel can't store bool; without this cast the rejection
            # surfaced as a misleading GeometryError about the affine.
            array = array.astype(np.uint8)
        try:
            # nibabel decomposes the affine into a qform (SVD) at construction
            # time and raises its own (undocumented-type) error for a
            # singular/degenerate one before our own affine checks below ever
            # run -- catch that here and fold it into the same GeometryError.
            img = nib.Nifti1Image(array, affine)
        except Exception as exc:
            raise GeometryError(
                f"Could not build an image from the given affine:\n{affine}\n({exc})") from exc
    else:
        raise InvalidLesionError(
            "Unsupported lesion input: expected a path, a nibabel image, or "
            f"an (array, affine) tuple, got {type(src)!r}.")

    shape = img.shape
    if len(shape) == 4:
        if shape[3] != 1:
            raise InvalidLesionError(
                "4D lesion image must have a singleton last axis, got shape "
                f"{shape}.")
        data = np.asarray(img.get_fdata())[..., 0]
        header = img.header.copy() if getattr(img, "header", None) is not None else None
        img = nib.Nifti1Image(data, img.affine, header)
    elif len(shape) != 3:
        raise InvalidLesionError(
            "Lesion image must be 3D (or 4D with a singleton last axis), "
            f"got shape {shape}.")

    affine = np.asarray(img.affine, dtype=float)
    if not np.all(np.isfinite(affine)):
        raise GeometryError(f"Lesion affine contains non-finite values:\n{affine}")
    det = np.linalg.det(affine[:3, :3])
    if not np.isfinite(det) or abs(det) < 1e-10:
        raise GeometryError(
            f"Lesion affine is singular or near-singular (determinant={det!r}).")

    if hasattr(img, "get_data_dtype"):
        dtype = np.dtype(img.get_data_dtype())
    else:
        dtype = np.asarray(img.dataobj).dtype
    if dtype.kind not in "iufb":
        raise InvalidLesionError(f"Lesion image must have a numeric dtype, got {dtype}.")

    return img


def load_lesion(src, threshold: float = 0.5):
    """Load/coerce a lesion source and return ``(img, was_binary)``.

    ``src`` is anything :func:`as_lesion_image` accepts (a path, a nibabel
    image, or an ``(array, affine)`` tuple). Values > ``threshold`` become 1,
    everything else (including NaN/inf) 0. Kept for backward compatibility;
    new code should use :func:`lqtpy.prepare`, which does the same work once
    and caches everything derived from it.
    """
    from .prepared import prepare

    p = prepare(src, threshold)
    return p.img, p.was_binary


def resample_mask_to(lesion_img, ref_img):
    """Resample a binary mask onto ``ref_img``'s grid (nearest-neighbour).

    Returns a boolean ndarray on the reference grid. If the grids already
    match, resampling is skipped.
    """
    same_shape = lesion_img.shape[:3] == ref_img.shape[:3]
    same_affine = np.allclose(lesion_img.affine, ref_img.affine, atol=1e-3)
    if same_shape and same_affine:
        return np.asarray(lesion_img.get_fdata()) > 0
    res = resample_to_img(
        lesion_img, ref_img, interpolation="nearest",
        force_resample=True, copy_header=True,
    )
    return np.asarray(res.get_fdata()) > 0
