"""Prove ``lqtpy._resample.resample_to_img`` is a faithful drop-in for
``nilearn.image.resample_to_img`` (the function it replaces core's nilearn
dependency for).

Skips cleanly (via ``pytest.importorskip``) when nilearn isn't installed, so
this test does not turn nilearn into a hard test-time dependency either.
"""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

from lqtpy._resample import resample_to_img as lqtpy_resample

nilearn_image = pytest.importorskip("nilearn.image")


def _target_img(shape, affine):
    return nib.Nifti1Image(np.zeros(shape, dtype=np.uint8), affine)


@pytest.mark.parametrize("interpolation", ["nearest", "continuous"])
def test_parity_offgrid(offgrid_lesion_img, sphere_shape, sphere_affine,
                        interpolation):
    target = _target_img(sphere_shape, sphere_affine)
    ours = lqtpy_resample(offgrid_lesion_img, target, interpolation=interpolation)
    theirs = nilearn_image.resample_to_img(
        offgrid_lesion_img, target, interpolation=interpolation,
        force_resample=True, copy_header=True,
    )
    assert ours.shape == theirs.shape
    assert np.allclose(ours.affine, theirs.affine)
    assert np.allclose(np.asarray(ours.get_fdata()), np.asarray(theirs.get_fdata()))


@pytest.mark.parametrize("interpolation", ["nearest", "continuous"])
def test_parity_oblique(oblique_lesion_img, sphere_shape, sphere_affine,
                        interpolation):
    target = _target_img(sphere_shape, sphere_affine)
    ours = lqtpy_resample(oblique_lesion_img, target, interpolation=interpolation)
    theirs = nilearn_image.resample_to_img(
        oblique_lesion_img, target, interpolation=interpolation,
        force_resample=True, copy_header=True,
    )
    assert ours.shape == theirs.shape
    assert np.allclose(ours.affine, theirs.affine)
    assert np.allclose(np.asarray(ours.get_fdata()), np.asarray(theirs.get_fdata()))


@pytest.mark.parametrize("interpolation", ["nearest", "continuous"])
def test_parity_sphere_onto_shifted_grid(sphere_lesion_img, sphere_shape,
                                         interpolation):
    """Same voxel size, different origin/shape -- a plain translation case,
    distinct from the oblique/off-grid cases above."""
    shifted_affine = np.diag([1.0, 1.0, 1.0, 1.0])
    shifted_affine[:3, 3] = [-30.0, -30.0, -30.0]
    target = _target_img((70, 70, 70), shifted_affine)
    ours = lqtpy_resample(sphere_lesion_img, target, interpolation=interpolation)
    theirs = nilearn_image.resample_to_img(
        sphere_lesion_img, target, interpolation=interpolation,
        force_resample=True, copy_header=True,
    )
    assert ours.shape == theirs.shape
    assert np.allclose(ours.affine, theirs.affine)
    assert np.allclose(np.asarray(ours.get_fdata()), np.asarray(theirs.get_fdata()))


@pytest.mark.parametrize("interpolation", ["nearest", "continuous"])
def test_parity_probabilistic(probabilistic_lesion_img, offgrid_lesion_img,
                              interpolation):
    """A float-valued (probabilistic) source resampled onto an off-grid target."""
    target = offgrid_lesion_img
    ours = lqtpy_resample(probabilistic_lesion_img, target, interpolation=interpolation)
    theirs = nilearn_image.resample_to_img(
        probabilistic_lesion_img, target, interpolation=interpolation,
        force_resample=True, copy_header=True,
    )
    assert ours.shape == theirs.shape
    assert np.allclose(ours.affine, theirs.affine)
    assert np.allclose(np.asarray(ours.get_fdata()), np.asarray(theirs.get_fdata()))
