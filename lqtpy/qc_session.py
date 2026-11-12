"""A quality-check session for a mask being drawn: ``lqtpy.qc.Session``.

A drawing tool keeps one session per open drawing in its backend and sends
only what changed after each stroke (the slices touched, or the voxels
added and removed) instead of the whole volume. :meth:`Session.check`
returns the graded result, re-using the last one when nothing changed.

    s = lqtpy.qc.Session(shape, affine, context={"drawing_axis": 2})
    s.set_slice(2, 107, slice_array)        # after a stroke on axial slice 107
    r = s.check(tier="fast")                # QCResult, ~35 ms for a 20 cc lesion

The check itself is the ordinary :func:`lqtpy.qc.assess` on the session's
mask, so results are identical to checking a saved file of the same
voxels. What the session saves is reading, sending and re-parsing the image
on every stroke, and repeat checks of an unchanged drawing.
"""
from __future__ import annotations

import threading

import numpy as np

__all__ = ["Session"]


class Session:
    """One drawing. ``shape``/``affine``: the drawing's grid (or pass
    ``mask=`` an existing array / image instead of ``shape``). ``context``:
    a :class:`~lqtpy.qc_model.DrawingContext` or its dict; ``expected_side``
    and other :func:`~lqtpy.qc.assess` options go in ``options``."""

    def __init__(self, shape=None, affine=None, *, mask=None, context=None, name: str = "drawing",
                 **options):
        if mask is not None:
            if hasattr(mask, "dataobj"):
                affine = mask.affine if affine is None else affine
                mask = np.asanyarray(mask.dataobj)
            mask = np.asarray(mask) != 0
            if mask.ndim == 4:
                mask = mask[..., 0]
        elif shape is not None:
            mask = np.zeros(tuple(int(s) for s in shape[:3]), bool)
        else:
            raise ValueError("Session needs shape= or mask=")
        if affine is None:
            raise ValueError("Session needs the drawing's affine")
        self.affine = np.asarray(affine, float)
        self._mask = np.ascontiguousarray(mask, dtype=bool)
        self.context = context
        self.options = options
        self.name = name
        self._lock = threading.RLock()
        self._version = 0
        self._result = None          # (version, tier, QCResult)
        self.changed_slices: "dict[int, set]" = {0: set(), 1: set(), 2: set()}

    # ---------------------------------------------------------------- edits
    @property
    def mask(self) -> np.ndarray:
        """A read-only view of the current drawing."""
        v = self._mask.view()
        v.flags.writeable = False
        return v

    @property
    def version(self) -> int:
        """Goes up by one with every edit that changed a voxel."""
        return self._version

    def _touch(self, axis=None, index=None):
        self._version += 1
        if axis is not None:
            self.changed_slices[axis].update(np.atleast_1d(index).tolist())

    def set_slice(self, axis: int, index: int, data) -> bool:
        """Replace one slice (``axis`` 0/1/2, array axis). Returns True if
        any voxel changed."""
        data = np.asarray(data) != 0
        with self._lock:
            sl = [slice(None)] * 3
            sl[axis] = int(index)
            cur = self._mask[tuple(sl)]
            if cur.shape != data.shape:
                raise ValueError(f"slice shape {data.shape} != {cur.shape}")
            if np.array_equal(cur, data):
                return False
            self._mask[tuple(sl)] = data
            self._touch(axis, index)
            return True

    def apply(self, added=None, removed=None) -> bool:
        """Add and/or remove voxels: boolean masks of the full grid, or
        ``(N, 3)`` voxel index arrays. Returns True if any voxel changed."""
        with self._lock:
            before = self._version
            for vals, on in ((removed, False), (added, True)):
                if vals is None:
                    continue
                vals = np.asarray(vals)
                if vals.shape == self._mask.shape:
                    sel = vals.astype(bool)
                    if (self._mask[sel] != on).any():
                        self._mask[sel] = on
                        self._version += 1
                        idx = np.argwhere(sel)
                        for a in range(3):
                            self.changed_slices[a].update(np.unique(idx[:, a]).tolist())
                elif vals.size:
                    ijk = vals.reshape(-1, 3).astype(int)
                    t = tuple(ijk.T)
                    if (self._mask[t] != on).any():
                        self._mask[t] = on
                        self._version += 1
                        for a in range(3):
                            self.changed_slices[a].update(np.unique(ijk[:, a]).tolist())
            return self._version != before

    def set_mask(self, mask) -> bool:
        """Replace the whole drawing (e.g. after Undo of many strokes)."""
        m = np.asarray(mask) != 0
        with self._lock:
            if m.shape != self._mask.shape:
                raise ValueError(f"mask shape {m.shape} != {self._mask.shape}")
            if np.array_equal(m, self._mask):
                return False
            diff = np.argwhere(m != self._mask)
            self._mask[...] = m
            self._version += 1
            for a in range(3):
                self.changed_slices[a].update(np.unique(diff[:, a]).tolist())
            return True

    # ---------------------------------------------------------------- checks
    def check(self, *, tier: str = "fast"):
        """The :class:`~lqtpy.qc_model.QCResult` for the current drawing
        (``tier`` "fast" or "full"). Re-uses the last result when nothing
        changed since. An empty drawing gives ``None``."""
        from . import qc
        from .prepared import PreparedLesion
        with self._lock:
            if self._result and self._result[0] == self._version and self._result[1] == tier:
                return self._result[2]
            if not self._mask.any():
                return None
            p = PreparedLesion(mask=self._mask.copy(), affine=self.affine, threshold=0.5,
                               was_binary=True, n_nonfinite=0, name=self.name, source=None)
            version = self._version
        res = qc.assess(p, tier=tier, context=self.context, **self.options)
        with self._lock:
            if version == self._version:
                self._result = (version, tier, res)
                for a in self.changed_slices:
                    self.changed_slices[a].clear()
        return res

    def image(self):
        """The drawing as a uint8 NIfTI image."""
        import nibabel as nib
        return nib.Nifti1Image(self._mask.astype(np.uint8), self.affine)
