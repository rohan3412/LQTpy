"""Native-space masks: warp to MNI, and check quality where gaps are clearest.

A lesion drawn on a patient's thick-slice scan shows its drawing problems
(gaps between drawn slices, single-slice drawings, unfilled outlines) most
clearly on that scan's own grid. After warping to MNI they are smeared into
thin dips and slabs that :mod:`lqtpy.qc` has to infer. Giving LQTpy the
native mask as well lets those checks run on the original drawing.

Two uses:

- ``analyze_single(mni_mask, native=native_mask)``: analyse the MNI mask as
  usual, but run the drawing checks (``sparse_slices``, ``flat``,
  ``hollow``, ``specks``) on the native mask.
- ``analyze_single(native_mask, transform=T)``: warp the native mask to
  MNI with :func:`warp_to_mni` first, then do the same.

Transforms (``transform=``):

- a 4 x 4 matrix (array, or a text file of 4 rows) mapping **native world
  mm to MNI world mm** (RAS). For FSL FLIRT ``.mat`` files, which work in
  scaled voxel coordinates, convert first (e.g. ``fsl.wrappers`` or
  ``c3d_affine_tool -fsl2ras``). ANTs/ITK ``.mat`` files are LPS and must be
  converted to RAS the same way.
- a **deformation field** image on the MNI grid whose last axis holds, for
  every MNI voxel, the native world coordinate (mm, RAS) it comes from:
  the convention of SPM's ``y_*.nii`` "inverse" fields. Shape (X, Y, Z, 3)
  or (X, Y, Z, 1, 3).
"""
from __future__ import annotations

import os

import numpy as np
import nibabel as nib
from scipy import ndimage

from .errors import GeometryError, InvalidLesionError

__all__ = ["warp_to_mni", "load_transform", "DRAWING_CHECKS", "native_qc"]

#: QC checks that describe the drawing itself, and so read best natively.
DRAWING_CHECKS = ("sparse_slices", "flat", "hollow", "specks")


def load_transform(transform):
    """``("affine", 4x4)`` or ``("field", image)`` from an array, a text
    file, or an image path/object."""
    if isinstance(transform, np.ndarray) or isinstance(transform, (list, tuple)):
        m = np.asarray(transform, float)
        if m.shape != (4, 4):
            raise GeometryError(f"An affine transform must be 4 x 4, got {m.shape}.")
        return "affine", m
    if hasattr(transform, "dataobj"):
        return "field", transform
    path = os.fspath(transform)
    low = path.lower()
    if low.endswith((".nii", ".nii.gz", ".mgz")):
        return "field", nib.load(path)
    try:
        m = np.loadtxt(path)
    except Exception as exc:
        raise GeometryError(f"Could not read a 4 x 4 matrix from '{path}': {exc}") from exc
    if m.shape != (4, 4):
        raise GeometryError(f"'{path}' holds a {m.shape} matrix; expected 4 x 4.")
    return "affine", m


def _mni_grid():
    from .atlases import load_atlas
    at = load_atlas("tissue")
    return at.img.shape[:3], np.asarray(at.img.affine, float)


def warp_to_mni(native, transform, *, threshold: float = 0.5, reference=None,
                order: int = 1) -> nib.Nifti1Image:
    """Resample a native-space mask into MNI space (FSL MNI152 1 mm grid by
    default, or ``reference``'s grid). Linear interpolation then
    re-thresholding at 0.5, the way registration pipelines warp masks.
    Returns a uint8 image."""
    from .prepared import prepare
    p = prepare(native, threshold)
    if reference is None:
        shape, raff = _mni_grid()
    else:
        shape, raff = reference.shape[:3], np.asarray(reference.affine, float)
    kind, T = load_transform(transform)
    ijk = np.indices(shape, dtype=np.float32).reshape(3, -1)
    if kind == "affine":
        # MNI voxel -> MNI world -> native world -> native voxel.
        M = np.linalg.inv(p.affine) @ np.linalg.inv(T) @ raff
        src = M[:3, :3] @ ijk + M[:3, 3:4]
    else:
        field = np.asarray(T.dataobj, dtype=np.float32)
        if field.ndim == 5:
            field = field[..., 0, :]
        if field.ndim != 4 or field.shape[-1] != 3:
            raise GeometryError(f"A deformation field needs shape (X, Y, Z, 3), got {field.shape}.")
        if field.shape[:3] != tuple(shape) or not np.allclose(T.affine, raff, atol=1e-3):
            shape, raff = field.shape[:3], np.asarray(T.affine, float)
        world = field.reshape(-1, 3).T
        inv = np.linalg.inv(p.affine)
        src = inv[:3, :3] @ world + inv[:3, 3:4]
    vals = ndimage.map_coordinates(p.mask.astype(np.float32), src, order=order, mode="constant",
                                   cval=0.0, prefilter=False)
    out = (vals.reshape(shape) >= 0.5).astype(np.uint8)
    if not out.any():
        raise InvalidLesionError("The warped mask is empty: check the transform's direction "
                                 "(it must map native mm to MNI mm) and units.")
    return nib.Nifti1Image(out, raff)


def native_qc(native, *, threshold: float = 0.5) -> "list":
    """The drawing checks run on the native mask, messages prefixed so a
    reader knows where they came from."""
    from . import qc as _qc
    flags = _qc.check(native, threshold=threshold, mni=False)
    # Copies: check() results are cached on the prepared lesion.
    return [_qc.Flag(f.id, f.severity,
                     "In the native-space drawing: " + f.message[:1].lower() + f.message[1:],
                     {**(f.evidence or {}), "space": "native"}, f.suggestion)
            for f in flags if f.id in DRAWING_CHECKS]
