"""Exact, faster equivalents of a few scipy.ndimage binary operations.

- :func:`fill_holes`: ``ndimage.binary_fill_holes`` (holes are background
  pieces that don't reach the array's edge), by one labelling pass instead
  of a dilation repeated until nothing changes. ``axis`` fills each slice
  across that axis on its own, all slices at once.
- :func:`dilate26` / :func:`erode26`: ``binary_dilation`` /
  ``binary_erosion`` with the 3 x 3 x 3 cube (and the default
  ``border_value=0``), as three one-voxel shifts per pass.

Each returns exactly what the scipy call returns (tests/test_speedups.py).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def fill_holes(mask: np.ndarray, axis: "int | None" = None) -> np.ndarray:
    """``ndimage.binary_fill_holes(mask)``; with ``axis``, the same for each
    slice ``mask[..., k, ...]`` along ``axis``."""
    mask = np.asarray(mask, bool)
    if axis is None:
        structure = None
        faces = range(mask.ndim)
    else:
        structure = ndimage.generate_binary_structure(mask.ndim, 1)
        sl = [slice(None)] * mask.ndim
        sl[axis] = [0, 2]
        structure[tuple(sl)] = False      # no connections between slices
        faces = [a for a in range(mask.ndim) if a != axis]
    lab, n = ndimage.label(~mask, structure=structure)
    outside = np.zeros(n + 1, bool)
    for a in faces:
        outside[np.take(lab, 0, axis=a)] = True
        outside[np.take(lab, -1, axis=a)] = True
    outside[0] = True                     # label 0 is the mask itself
    return mask | ~outside[lab]


def _shift_pass(m: np.ndarray, axis: int, grow: bool) -> np.ndarray:
    a = np.moveaxis(m, axis, 0)
    out = a.copy()
    if grow:
        out[1:] |= a[:-1]
        out[:-1] |= a[1:]
    else:
        out[1:] &= a[:-1]
        out[:-1] &= a[1:]
        out[0] = False                    # a neighbour outside the array counts as empty
        out[-1] = False
    return np.moveaxis(out, 0, axis)


def dilate26(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """``ndimage.binary_dilation(mask, np.ones((3, 3, 3)), iterations)``."""
    out = np.asarray(mask, bool)
    for _ in range(iterations):
        for a in range(out.ndim):
            out = _shift_pass(out, a, True)
    return np.ascontiguousarray(out)


def erode26(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    """``ndimage.binary_erosion(mask, np.ones((3, 3, 3)), iterations)``."""
    out = np.asarray(mask, bool)
    for _ in range(iterations):
        for a in range(out.ndim):
            out = _shift_pass(out, a, False)
    return np.ascontiguousarray(out)
