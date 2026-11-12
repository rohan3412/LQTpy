"""ANTs / ANTsPy transforms as input to :func:`lqtpy.native.warp_to_mni`.

Pass the transform list ANTs gives for warping the native scan to MNI, in
the order ANTs uses it: ANTsPy's ``registration(fixed=mni, moving=native)
["fwdtransforms"]``, or the ``-t`` arguments of ``antsApplyTransforms``,
typically ``["…1Warp.nii.gz", "…0GenericAffine.mat"]``.

    lqtpy.native.warp_to_mni("lesion_native.nii.gz", ants_fwdtransforms)
    lqtpy single lesion_native.nii.gz --transform 1Warp.nii.gz --transform 0GenericAffine.mat

Supported files:

- affine: ITK ``.mat`` (MATLAB v4, ANTs' ``*GenericAffine.mat``) or ITK
  text (``.txt`` / ``.tfm``, "#Insight Transform File");
- displacement field: a NIfTI whose name ends in ``Warp.nii[.gz]`` (ANTs
  naming), 5-D ``(X, Y, Z, 1, 3)`` or 4-D ``(X, Y, Z, 3)``.

Not supported: composite ``.h5`` files (write them unpacked with
``write_composite_transform=False``) and inverse-warp lists (those go the
other way, MNI -> native).

Conventions (ITK): every transform maps a point of the **fixed** space (MNI)
to the **moving** space (native), in LPS millimetres, which is exactly the
direction a pull-back resampling needs. Points pass through the list in
order: the first transform listed is applied first to an MNI point. An
affine maps ``y = A (x - c) + t + c`` (``c``: the fixed centre); a
displacement field maps ``y = x + u(x)``.
"""
from __future__ import annotations

import os

import numpy as np

from .errors import GeometryError

__all__ = ["is_ants_transform", "load_ants", "map_points"]

_LPS = np.array([-1.0, -1.0, 1.0])


def _is_warp_name(path: str) -> bool:
    low = os.path.basename(path).lower()
    return low.endswith(("warp.nii.gz", "warp.nii"))


def is_ants_transform(path) -> bool:
    """True for a file ANTs writes (``.mat``, ITK text, ``*Warp.nii.gz``)."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    p = os.fspath(path)
    low = p.lower()
    if low.endswith(".mat"):
        # FSL-style text matrices are .mat too: a readable 4 x 4 text file is not ANTs.
        try:
            return np.loadtxt(p).shape != (4, 4)
        except Exception:
            return True
    if _is_warp_name(p) or low.endswith(".tfm"):
        return True
    if low.endswith(".txt"):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                return fh.readline().startswith("#Insight Transform File")
        except OSError:
            return False
    if low.endswith(".h5"):
        return True
    return False


def _affine_from_params(params, fixed):
    params = np.asarray(params, float).ravel()
    if params.size != 12:
        raise GeometryError(f"An ITK affine needs 12 parameters, got {params.size}.")
    A = params[:9].reshape(3, 3)
    t = params[9:]
    c = np.zeros(3) if fixed is None else np.asarray(fixed, float).ravel()[:3]
    return ("affine", A, t, c)


def _read_mat(path):
    from scipy.io import loadmat
    try:
        m = loadmat(path)
    except Exception as exc:
        raise GeometryError(f"Could not read the ITK transform '{path}': {exc}") from exc
    key = next((k for k in m if not k.startswith("__") and k != "fixed"
                and ("Transform" in k)), None)
    if key is None:
        raise GeometryError(f"'{path}' holds no ITK affine transform (keys: "
                            f"{[k for k in m if not k.startswith('__')]}).")
    return _affine_from_params(m[key], m.get("fixed"))


def _read_txt(path):
    params = fixed = None
    kind = ""
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Transform:"):
                kind = line.split(":", 1)[1].strip()
                if params is not None:
                    raise GeometryError(f"'{path}' holds several transforms; pass them as a list.")
            elif line.startswith("Parameters:"):
                params = [float(v) for v in line.split(":", 1)[1].split()]
            elif line.startswith("FixedParameters:"):
                fixed = [float(v) for v in line.split(":", 1)[1].split()]
    if params is None:
        raise GeometryError(f"'{path}' has no Parameters line.")
    if "Affine" not in kind and "MatrixOffset" not in kind and "Rigid" not in kind \
            and "Euler" not in kind and "Similarity" not in kind:
        raise GeometryError(f"'{path}': transform type {kind!r} isn't supported "
                            "(affine / rigid only).")
    if "Euler3D" in kind or "Rigid3D" in kind:
        raise GeometryError(f"'{path}': convert {kind} to an affine first "
                            "(antsApplyTransforms -o Linear[affine.mat]).")
    return _affine_from_params(params, fixed)


def _read_warp(path):
    import nibabel as nib
    img = nib.load(path)
    u = np.asarray(img.dataobj, dtype=np.float32)
    if u.ndim == 5:
        u = u[..., 0, :]
    if u.ndim != 4 or u.shape[-1] != 3:
        raise GeometryError(f"An ANTs warp needs shape (X, Y, Z, 1, 3), got {img.shape}.")
    return ("field", u, np.asarray(img.affine, float))


def load_ants(transforms) -> list:
    """Read a list of ANTs transform files (ANTs order) into steps."""
    if isinstance(transforms, (str, os.PathLike)):
        transforms = [transforms]
    steps = []
    for t in transforms:
        p = os.fspath(t)
        low = p.lower()
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        if low.endswith(".h5"):
            raise GeometryError(f"'{p}': composite .h5 transforms aren't supported; save the "
                                "registration with write_composite_transform=False.")
        if low.endswith(".mat"):
            steps.append(_read_mat(p))
        elif _is_warp_name(p):
            if "inversewarp" in os.path.basename(low):
                raise GeometryError(f"'{p}' is an inverse warp (MNI -> native); lqtpy needs the "
                                    "forward transforms (fwdtransforms).")
            steps.append(_read_warp(p))
        elif low.endswith((".txt", ".tfm")):
            steps.append(_read_txt(p))
        else:
            raise GeometryError(f"'{p}' isn't an ANTs transform file (.mat, .txt, *Warp.nii.gz).")
    if not steps:
        raise GeometryError("No ANTs transforms given.")
    return steps


def map_points(steps, points_ras: np.ndarray) -> np.ndarray:
    """MNI points (3, N; RAS mm) -> native points (3, N; RAS mm)."""
    from scipy import ndimage
    p = np.asarray(points_ras, np.float64) * _LPS[:, None]          # to LPS
    for st in steps:
        if st[0] == "affine":
            _, A, t, c = st
            p = A @ (p - c[:, None]) + (t + c)[:, None]
        else:
            _, u, aff = st
            # The field's grid is in RAS (nibabel); its vectors are LPS.
            inv = np.linalg.inv(aff)
            ras = p * _LPS[:, None]
            ijk = inv[:3, :3] @ ras + inv[:3, 3:4]
            # Outside its grid a field moves nothing (ITK's convention).
            disp = np.stack([ndimage.map_coordinates(u[..., k], ijk, order=1, mode="constant",
                                                     cval=0.0, prefilter=False) for k in range(3)])
            p = p + disp
    return p * _LPS[:, None]                                         # back to RAS
