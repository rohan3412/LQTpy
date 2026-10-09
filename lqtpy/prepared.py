"""Prepared lesions: load and binarize once, reuse everywhere.

Every :mod:`lqtpy.api` function used to reload, re-threshold and re-resample
the lesion it was given. Calling five of them on one lesion paid that cost
five times. :func:`prepare` does the work once and returns a
:class:`PreparedLesion` that every API function accepts in place of a raw
lesion:

    les = lqtpy.prepare("lesion_MNI.nii.gz")
    lqtpy.lesion_stats(les)
    lqtpy.atlas_overlap(les, "aal")
    lqtpy.lobe_overlap(les, "dk")

A prepared lesion caches the mask resampled onto each atlas grid it meets,
so two atlases sharing a grid share the resampling too. It is immutable
from the caller's point of view and safe to share between threads.
"""
from __future__ import annotations

import os
import threading

import numpy as np
import nibabel as nib

from .errors import InvalidLesionError
from .io import as_lesion_image

__all__ = ["PreparedLesion", "prepare", "GridMask"]


class GridMask:
    """A lesion mask resampled onto one reference grid.

    ``mask`` is a read-only boolean array with the reference grid's shape.
    ``outside_fov_voxels`` counts lesion voxels (on the lesion's *own*
    grid) whose centres fall outside the reference grid's field of view:
    they can't appear in ``mask`` at all, and are reported rather than
    dropped silently.
    """

    __slots__ = ("mask", "n_voxels", "outside_fov_voxels", "outside_fov_fraction")

    def __init__(self, mask, outside_fov_voxels, n_native):
        mask.flags.writeable = False
        self.mask = mask
        self.n_voxels = int(np.count_nonzero(mask))
        self.outside_fov_voxels = int(outside_fov_voxels)
        self.outside_fov_fraction = (outside_fov_voxels / n_native) if n_native else 0.0


def _grid_key(shape, affine) -> tuple:
    return (tuple(int(s) for s in shape[:3]),
            tuple(np.round(np.asarray(affine, dtype=float).ravel(), 6)))


class PreparedLesion:
    """A binarized lesion plus caches. Build with :func:`prepare`.

    Attributes
    ----------
    img : nibabel.Nifti1Image
        The binary (uint8, 0/1) lesion on its own grid.
    mask : ndarray of bool
        The same, as a read-only boolean array.
    affine : ndarray (4, 4)
    threshold : float
        Voxels with value strictly greater than this are lesion.
    was_binary : bool
        True if every finite input value was exactly 0 or 1.
    n_nonfinite : int
        NaN/inf voxels in the input. They are treated as 0 (not lesion).
    name : str
        A label for outputs: the file stem for a path, else ``"lesion"``.
    n_voxels : int
    bbox : (lo, hi)
        Inclusive-exclusive voxel bounds of the lesion on its own grid.
    """

    def __init__(self, *, img, mask, threshold, was_binary, n_nonfinite, name, source):
        mask.flags.writeable = False
        self.img = img
        self.mask = mask
        self.affine = np.asarray(img.affine, dtype=float)
        self.shape = tuple(int(s) for s in mask.shape)
        self.threshold = float(threshold)
        self.was_binary = bool(was_binary)
        self.n_nonfinite = int(n_nonfinite)
        self.name = name
        self.source = source
        nz = np.nonzero(mask)
        self.n_voxels = int(nz[0].size)
        self._nonzero = tuple(a.astype(np.int32) for a in nz)
        self.bbox = (tuple(int(a.min()) for a in nz), tuple(int(a.max()) + 1 for a in nz))
        self._grids: "dict[tuple, GridMask]" = {}
        self._cache: dict = {}
        # Re-entrant: a cached value (morphometry) may be built from another
        # cached value (components) while the lock is held.
        self._lock = threading.RLock()
        #: qform/sform facts about the original header (see _header_info).
        self.header_info: dict = {}
        #: the input's values before thresholding: dtype, distinct values,
        #: integer or not, and (for fractional data) counts above 0.3/0.5/0.7.
        self.value_info: dict = {}

    def __repr__(self) -> str:
        return (f"PreparedLesion(name={self.name!r}, n_voxels={self.n_voxels}, "
                f"shape={self.shape}, threshold={self.threshold})")

    # ------------------------------------------------------------------ #
    def voxel_coords(self) -> np.ndarray:
        """(N, 3) int array of lesion voxel indices on the lesion's own grid."""
        return np.column_stack(self._nonzero)

    def world_coords(self) -> np.ndarray:
        """(N, 3) float array of lesion voxel centres in world mm."""
        ijk = self.voxel_coords().astype(float)
        return ijk @ self.affine[:3, :3].T + self.affine[:3, 3]

    def cropped(self, pad: int = 1):
        """``(sub_mask, offset)``: the mask cropped to its bounding box plus
        ``pad`` voxels (clipped to the grid), and the voxel offset of the
        crop's origin. Whole-grid algorithms (labelling, distance maps)
        run on this instead of the full volume."""
        lo = [max(0, a - pad) for a in self.bbox[0]]
        hi = [min(s, b + pad) for s, b in zip(self.shape, self.bbox[1])]
        sub = self.mask[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        return sub, np.asarray(lo)

    def cached(self, key, build):
        """Thread-safe memo for derived quantities (morphometry, components...)."""
        value = self._cache.get(key)
        if value is None:
            with self._lock:
                value = self._cache.get(key)
                if value is None:
                    value = build()
                    self._cache[key] = value
        return value

    # ------------------------------------------------------------------ #
    def on_grid(self, ref_img) -> GridMask:
        """The lesion resampled (nearest-neighbour) onto ``ref_img``'s grid,
        cached per grid. Accepts a nibabel image or a ``(shape, affine)``
        pair."""
        if isinstance(ref_img, tuple):
            ref_shape, ref_affine = ref_img
        else:
            ref_shape, ref_affine = ref_img.shape[:3], ref_img.affine
        key = _grid_key(ref_shape, ref_affine)
        gm = self._grids.get(key)
        if gm is None:
            with self._lock:
                gm = self._grids.get(key)
                if gm is None:
                    gm = self._resample(tuple(int(s) for s in ref_shape[:3]),
                                        np.asarray(ref_affine, dtype=float))
                    self._grids[key] = gm
        return gm

    def _outside_fov(self, ref_shape, ref_affine) -> int:
        """Lesion voxels whose centres map outside the reference grid, using
        the same round-to-nearest rule nearest-neighbour resampling uses."""
        vox = np.linalg.inv(ref_affine) @ np.vstack(
            [self.world_coords().T, np.ones(self.n_voxels)])
        idx = np.round(vox[:3]).astype(np.int64)
        inside = np.all((idx >= 0) & (idx < np.asarray(ref_shape)[:, None]), axis=0)
        return int(self.n_voxels - np.count_nonzero(inside))

    def _resample(self, ref_shape, ref_affine) -> GridMask:
        same = (ref_shape == self.shape
                and np.allclose(ref_affine, self.affine, atol=1e-3))
        if same:
            return GridMask(self.mask.copy(), 0, self.n_voxels)
        return GridMask(_nearest_on_grid(self.mask, self.affine, self.bbox,
                                         ref_shape, ref_affine),
                        self._outside_fov(ref_shape, ref_affine), self.n_voxels)


def _nearest_on_grid(mask, src_affine, bbox, ref_shape, ref_affine) -> np.ndarray:
    """Nearest-neighbour resample of a boolean ``mask`` onto a reference grid.

    For each reference voxel the source coordinate is
    ``inv(src_affine) @ ref_affine @ ijk`` and the source voxel is
    ``floor(coord + 0.5)`` -- SciPy's order-0 rule (round half up), so this
    matches :func:`lqtpy._resample.resample_to_img` with
    ``interpolation="nearest"`` except where a coordinate lands within
    floating-point error of a half-voxel tie.

    Only the reference voxels the lesion's bounding box can reach are
    evaluated, and coordinates are always computed from *global* reference
    indices, so the result doesn't depend on that cropping. This is what
    makes resampling cost scale with lesion size instead of grid size.
    """
    from scipy import linalg

    ref_shape = tuple(int(s) for s in ref_shape)
    out = np.zeros(ref_shape, dtype=bool)
    M = linalg.inv(src_affine) @ ref_affine          # ref voxel -> src voxel
    Minv = linalg.inv(M)                             # src voxel -> ref voxel
    lo, hi = np.asarray(bbox[0], float), np.asarray(bbox[1], float) - 1
    corners = np.array([[x, y, z] for x in (lo[0] - 1, hi[0] + 1)
                        for y in (lo[1] - 1, hi[1] + 1)
                        for z in (lo[2] - 1, hi[2] + 1)])
    tvox = corners @ Minv[:3, :3].T + Minv[:3, 3]
    tlo = np.maximum(np.floor(tvox.min(0)).astype(int) - 1, 0)
    thi = np.minimum(np.ceil(tvox.max(0)).astype(int) + 2, np.asarray(ref_shape))
    if not np.all(thi > tlo):
        return out
    ti, tj, tk = (np.arange(a, b, dtype=float) for a, b in zip(tlo, thi))
    src_shape = mask.shape
    idx = []
    valid = None
    for r in range(3):
        c = (M[r, 0] * ti[:, None, None] + M[r, 1] * tj[None, :, None]
             + M[r, 2] * tk[None, None, :] + M[r, 3])
        # SciPy's mode="constant": a coordinate beyond the outermost voxel
        # centres (outside [0, n-1]) reads the fill value, even where
        # rounding would land on an edge voxel.
        ok = (c >= 0) & (c <= src_shape[r] - 1)
        valid = ok if valid is None else (valid & ok)
        idx.append(np.floor(c + 0.5).astype(np.int64))
    sub = np.zeros(valid.shape, dtype=bool)
    sub[valid] = mask[idx[0][valid], idx[1][valid], idx[2][valid]]
    out[tlo[0]:thi[0], tlo[1]:thi[1], tlo[2]:thi[2]] = sub
    return out


def _stem(src) -> str:
    if not isinstance(src, (str, os.PathLike)):
        return "lesion"
    from .formats import strip_extension
    return strip_extension(os.path.basename(os.fspath(src)))


def prepare(lesion, threshold: float = 0.5, *, name: "str | None" = None) -> PreparedLesion:
    """Load, validate and binarize a lesion once.

    ``lesion`` is anything :func:`lqtpy.io.as_lesion_image` accepts, or an
    existing :class:`PreparedLesion` (returned unchanged; its own threshold
    stands). A voxel is lesion when its value is strictly greater than
    ``threshold``. NaN and inf voxels count as 0 and are reported in
    ``n_nonfinite``.

    Raises :class:`~lqtpy.errors.InvalidLesionError` for an empty mask.
    """
    if isinstance(lesion, PreparedLesion):
        return lesion
    img = as_lesion_image(lesion)
    # dataobj applies scl_slope/inter and keeps the stored dtype otherwise,
    # so a uint8 mask is thresholded as uint8 -- no float64 copy of the grid.
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    n_nonfinite = 0
    if data.dtype.kind == "f":
        finite = np.isfinite(data)
        n_nonfinite = int(data.size - np.count_nonzero(finite))
        if n_nonfinite:
            data = np.where(finite, data, 0)
    mask = data > threshold
    value_info = {"dtype": str(data.dtype)}
    if data.dtype.kind == "b":
        was_binary = True
        value_info["values"] = [1]
    else:
        n_nonzero = np.count_nonzero(data)
        was_binary = bool(n_nonzero == np.count_nonzero(data == 1))
        if not was_binary and n_nonzero:
            nzv = data[data != 0]
            uniq = np.unique(nzv[:2_000_000])
            integer = bool(np.all(np.mod(uniq[:64], 1) == 0))
            value_info.update(values=[float(v) for v in uniq[:16]], n_values=int(min(len(uniq), 10 ** 6)),
                              integer=integer, max=float(nzv.max()), min=float(nzv.min()))
            if not integer:
                # Volume if the threshold were 0.3 or 0.7 instead: how much the
                # result depends on it.
                value_info["n_above"] = {str(t): int(np.count_nonzero(nzv > t)) for t in (0.3, 0.5, 0.7)}
    label = name or _stem(lesion)
    if not mask.any():
        where = os.fspath(lesion) if isinstance(lesion, (str, os.PathLike)) else "<in-memory lesion>"
        vmax = float(np.max(data)) if data.size else 0.0
        raise InvalidLesionError(
            f"Lesion '{where}' is empty after thresholding at {threshold} "
            f"(max value in image = {vmax}).")
    header = img.header.copy() if getattr(img, "header", None) is not None else None
    bin_img = nib.Nifti1Image(mask.astype(np.uint8), img.affine, header)
    bin_img.set_data_dtype(np.uint8)
    if hasattr(bin_img.header, "set_slope_inter"):
        bin_img.header.set_slope_inter(1, 0)
    prepared = PreparedLesion(img=bin_img, mask=mask, threshold=threshold,
                              was_binary=was_binary, n_nonfinite=n_nonfinite,
                              name=label, source=lesion)
    prepared.header_info = _header_info(img)
    prepared.value_info = value_info
    return prepared


def _header_info(img) -> dict:
    """qform/sform state of the *original* header, for QC. An in-memory
    (array, affine) input has an explicit affine and nothing to check."""
    hdr = getattr(img, "header", None)
    if hdr is None or not hasattr(hdr, "get_sform"):
        return {}
    try:
        _, scode = hdr.get_sform(coded=True)
        _, qcode = hdr.get_qform(coded=True)
        info = {"sform_code": int(scode), "qform_code": int(qcode),
                "no_orientation": int(scode) == 0 and int(qcode) == 0}
        if scode and qcode:
            q, s_ = hdr.get_qform(), hdr.get_sform()
            corners = np.array([[i, j, k, 1.0] for i in (0, img.shape[0] - 1)
                                for j in (0, img.shape[1] - 1) for k in (0, img.shape[2] - 1)])
            diff = np.abs((corners @ q.T)[:, :3] - (corners @ s_.T)[:, :3]).max()
            info["qform_sform_mm"] = round(float(diff), 2)
        return info
    except Exception:
        return {}
