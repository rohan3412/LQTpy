"""Synthetic lesion library for tests, calibration and demos.

No patient data: every lesion is built with numpy. Two families:

- MNI-space shapes drawn directly on the 1 mm FSL MNI152 grid
  (:func:`sphere`, :func:`ellipsoid`, :func:`blobs`).
- **Native drawings warped to MNI** (:func:`native_drawing`): a lesion
  drawn on a thick-slice native acquisition -- optionally on every k-th
  slice only, on a single slice, or as an unfilled outline -- then
  resampled to MNI with trilinear interpolation and re-thresholded at 0.5,
  the way registration pipelines do. This reproduces the smeared artefacts
  real MNI masks show, which is what the QC detectors must handle.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

MNI_SHAPE = (182, 218, 182)
MNI_AFFINE = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])


def _world_grid(shape, affine):
    ijk = np.indices(shape).reshape(3, -1).T.astype(float)
    return (ijk @ affine[:3, :3].T + affine[:3, 3]).reshape(*shape, 3)


_MNI_WORLD = None


def mni_world():
    global _MNI_WORLD
    if _MNI_WORLD is None:
        _MNI_WORLD = _world_grid(MNI_SHAPE, MNI_AFFINE)
    return _MNI_WORLD


def img(mask, affine=MNI_AFFINE):
    return nib.Nifti1Image(np.asarray(mask, dtype=np.uint8), affine)


def ellipsoid(center_mm, radii_mm, *, world=None):
    w = mni_world() if world is None else world
    d = (w - np.asarray(center_mm, float)) / np.asarray(radii_mm, float)
    return (d ** 2).sum(-1) <= 1.0


def sphere(center_mm, r_mm, *, world=None):
    return ellipsoid(center_mm, (r_mm, r_mm, r_mm), world=world)


def blobs(specs):
    """Union of spheres: ``specs`` = [(center_mm, r_mm), ...]."""
    m = np.zeros(MNI_SHAPE, bool)
    for c, r in specs:
        m |= sphere(c, r)
    return m


def _rot(deg_x=0.0, deg_z=0.0):
    ax, az = np.deg2rad(deg_x), np.deg2rad(deg_z)
    rx = np.array([[1, 0, 0], [0, np.cos(ax), -np.sin(ax)], [0, np.sin(ax), np.cos(ax)]])
    rz = np.array([[np.cos(az), -np.sin(az), 0], [np.sin(az), np.cos(az), 0], [0, 0, 1]])
    return rz @ rx


def native_drawing(center_mm, radii_mm, *, slice_mm=5.0, inplane_mm=0.9,
                   draw_every=1, only_slice=None, outline=False, tilt_deg=(8.0, 5.0),
                   phase=0):
    """A lesion hand-drawn on native axial slices, then warped to MNI.

    ``draw_every=k`` keeps every k-th slice's drawing (the others empty);
    ``only_slice`` keeps just the slice through the centre (+offset);
    ``outline=True`` keeps only each slice's 2-D boundary. The native grid
    is tilted by ``tilt_deg`` (about x, then z) relative to MNI, as a real
    acquisition would be. Returns a boolean MNI-grid mask.
    """
    from scipy import ndimage

    from lqtpy._resample import resample_to_img

    R = _rot(*tilt_deg)
    zooms = np.array([inplane_mm, inplane_mm, slice_mm])
    shape = tuple(int(np.ceil(200 / z)) for z in zooms)
    lin = R * zooms  # columns scaled
    centre_vox = np.array(shape) / 2.0
    aff = np.eye(4)
    aff[:3, :3] = lin
    aff[:3, 3] = np.asarray(center_mm, float) - lin @ centre_vox
    w = _world_grid(shape, aff)
    # Lesion defined in the native frame's own orientation around the centre.
    local = (w - np.asarray(center_mm, float)) @ R
    m = ((local / np.asarray(radii_mm, float)) ** 2).sum(-1) <= 1.0
    zs = np.flatnonzero(m.any(axis=(0, 1)))
    if only_slice is not None:
        keep = np.zeros(shape[2], bool)
        keep[int(round(centre_vox[2])) + int(only_slice)] = True
        m &= keep[None, None, :]
    elif draw_every > 1:
        keep = np.zeros(shape[2], bool)
        keep[zs[phase % draw_every::draw_every]] = True
        m &= keep[None, None, :]
    if outline:
        for k in range(shape[2]):
            sl = m[:, :, k]
            if sl.any():
                m[:, :, k] = sl & ~ndimage.binary_erosion(sl)
    native = nib.Nifti1Image(m.astype(np.float32), aff)
    ref = nib.Nifti1Image(np.zeros(MNI_SHAPE, np.uint8), MNI_AFFINE)
    warped = resample_to_img(native, ref, interpolation="linear")
    return np.asarray(warped.dataobj) > 0.5
