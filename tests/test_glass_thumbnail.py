"""Glass-brain thumbnails drawn from the lesion's slab must be byte-identical
to drawing the whole volume (lqtpy.plotting.glass_brain_png)."""
from __future__ import annotations

import numpy as np
import nibabel as nib
import pytest

import synthetic as S

pytest.importorskip("nilearn")
pytest.importorskip("matplotlib")

from lqtpy import plotting  # noqa: E402

plotting.use_agg()


def _both(img_fn, theme, monkeypatch):
    calls = []
    real = plotting._restore_full_frame

    def spy(*a):
        r = real(*a)
        calls.append(r)
        return r
    monkeypatch.setattr(plotting, "_restore_full_frame", spy)
    new = plotting.glass_brain_png(img_fn(), theme=theme)
    monkeypatch.setattr(plotting, "_slab", lambda img, margin=2: None)
    old = plotting.glass_brain_png(img_fn(), theme=theme)
    return new, old, calls


CASES = {
    "small sphere": lambda: S.img(S.sphere([30, -20, 10], 4)),
    # reaches past the glass-brain outline at the occipital pole: the panel
    # limits come from the lesion's bounding box
    "occipital pole": lambda: S.img(S.sphere([10, -112, 5], 8)),
    "float values just under 1": lambda: nib.Nifti1Image(
        np.where(S.sphere([-35, 5, 15], 10), np.float32(0.99999994), np.float32(0)), S.MNI_AFFINE),
}


@pytest.mark.parametrize("name", list(CASES))
def test_slab_thumbnail_is_byte_identical(name, monkeypatch):
    new, old, calls = _both(CASES[name], "light", monkeypatch)
    assert new is not None and new == old
    assert calls == [True]                       # the slab path was used, not the fallback


def test_dark_theme_identical(monkeypatch):
    new, old, calls = _both(CASES["occipital pole"], "dark", monkeypatch)
    assert new == old and calls == [True]


def test_oblique_grid_draws_whole_volume():
    aff = S.MNI_AFFINE.copy()
    aff[:3, :3] = aff[:3, :3] @ np.array([[1, 0.1, 0], [0, 1, 0], [0, 0, 1]])
    img = nib.Nifti1Image(S.sphere([0, 0, 0], 6).astype(np.uint8), aff)
    assert plotting._slab(img) is None


def test_non_image_input_is_not_cropped():
    assert plotting._slab("lesion.nii.gz") is None
