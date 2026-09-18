"""Synthetic lesion fixtures, built entirely in code as nibabel images.

No patient data is used anywhere in this test suite -- every "lesion" here is
a sphere/blob generated with numpy on a made-up grid. Fixtures that need a
file on disk write into ``tmp_path`` (pytest's per-test temp dir), never into
the repo.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest
from scipy import ndimage


def _sphere_mask(shape, center, radius):
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]
    dist2 = (zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2
    return dist2 <= radius ** 2


def _rotation_affine(voxel_mm, shape, angle_deg=12.0):
    """A 1-mm-ish oblique affine: isotropic voxel size, rotated about the
    z-axis, with the volume centred near world origin (MNI-like)."""
    theta = np.deg2rad(angle_deg)
    rot = np.array([
        [np.cos(theta), -np.sin(theta), 0.0],
        [np.sin(theta), np.cos(theta), 0.0],
        [0.0, 0.0, 1.0],
    ])
    mat = rot * voxel_mm
    center_vox = np.array(shape, dtype=float) / 2.0
    translation = -mat @ center_vox
    affine = np.eye(4)
    affine[:3, :3] = mat
    affine[:3, 3] = translation
    return affine


def _centered_affine(voxel_mm, shape):
    """Simple axis-aligned affine, isotropic ``voxel_mm``, volume centred on
    world origin (MNI-like: origin roughly in the middle of the brain)."""
    affine = np.diag([voxel_mm, voxel_mm, voxel_mm, 1.0])
    center_vox = np.array(shape, dtype=float) / 2.0
    affine[:3, 3] = -voxel_mm * center_vox
    return affine


# --------------------------------------------------------------------------- #
# Binary sphere on a 1 mm MNI-like grid.
# --------------------------------------------------------------------------- #
@pytest.fixture
def sphere_shape():
    return (64, 64, 64)


@pytest.fixture
def sphere_affine(sphere_shape):
    return _centered_affine(1.0, sphere_shape)


@pytest.fixture
def sphere_lesion_img(sphere_shape, sphere_affine):
    """A single binary sphere, radius 10 vox, on a 1 mm isotropic grid."""
    mask = _sphere_mask(sphere_shape, np.array(sphere_shape) / 2.0, 10)
    data = mask.astype(np.uint8)
    return nib.Nifti1Image(data, sphere_affine)


# --------------------------------------------------------------------------- #
# Off-grid: non-integer voxel size.
# --------------------------------------------------------------------------- #
@pytest.fixture
def offgrid_lesion_img():
    shape = (48, 48, 48)
    voxel_mm = 0.737
    affine = _centered_affine(voxel_mm, shape)
    mask = _sphere_mask(shape, np.array(shape) / 2.0, 8)
    data = mask.astype(np.uint8)
    return nib.Nifti1Image(data, affine)


# --------------------------------------------------------------------------- #
# Oblique affine (rotated, non-axis-aligned).
# --------------------------------------------------------------------------- #
@pytest.fixture
def oblique_lesion_img():
    shape = (56, 56, 56)
    affine = _rotation_affine(1.0, shape, angle_deg=12.0)
    mask = _sphere_mask(shape, np.array(shape) / 2.0, 9)
    data = mask.astype(np.uint8)
    return nib.Nifti1Image(data, affine)


# --------------------------------------------------------------------------- #
# Two disjoint blobs (connected-components).
# --------------------------------------------------------------------------- #
@pytest.fixture
def two_blob_lesion_img(sphere_shape, sphere_affine):
    a = _sphere_mask(sphere_shape, (18, 18, 18), 5)
    b = _sphere_mask(sphere_shape, (46, 46, 46), 6)
    data = (a | b).astype(np.uint8)
    return nib.Nifti1Image(data, sphere_affine)


# --------------------------------------------------------------------------- #
# Empty lesion (all zero).
# --------------------------------------------------------------------------- #
@pytest.fixture
def empty_lesion_img(sphere_shape, sphere_affine):
    data = np.zeros(sphere_shape, dtype=np.uint8)
    return nib.Nifti1Image(data, sphere_affine)


# --------------------------------------------------------------------------- #
# Probabilistic lesion: values spanning [0, 1], including values in (0, 0.5).
# --------------------------------------------------------------------------- #
@pytest.fixture
def probabilistic_lesion_img(sphere_shape, sphere_affine):
    core = _sphere_mask(sphere_shape, np.array(sphere_shape) / 2.0, 10).astype(float)
    smoothed = ndimage.gaussian_filter(core, sigma=3.0)
    smoothed = smoothed / smoothed.max()
    # Guarantee at least one voxel squarely in (0, 0.5) and one at/near 1.0.
    assert np.any((smoothed > 0) & (smoothed < 0.5))
    assert smoothed.max() > 0.9
    return nib.Nifti1Image(smoothed.astype(np.float32), sphere_affine)


# --------------------------------------------------------------------------- #
# File-on-disk helper: writes any of the above images into tmp_path.
# --------------------------------------------------------------------------- #
@pytest.fixture
def write_nifti(tmp_path):
    def _write(img, name="lesion.nii.gz"):
        path = tmp_path / name
        nib.save(img, str(path))
        return str(path)
    return _write


@pytest.fixture
def cortical_lesion_img():
    """A lesion placed in cortex, so atlas rollups actually return groups.

    The `sphere_lesion_img` fixture sits where Schaefer has no parcels, which
    is fine for most tests but makes a network rollup come back empty.
    """
    shape = (182, 218, 182)
    affine = np.array([[-1.0, 0, 0, 90.0],
                       [0, 1.0, 0, -126.0],
                       [0, 0, 1.0, -72.0],
                       [0, 0, 0, 1.0]])
    data = _sphere_mask(shape, (91 - 34, 109 + 4, 91 + 28), 12).astype(np.float32)
    return nib.Nifti1Image(data, affine)
