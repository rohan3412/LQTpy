"""Every bundled data file the code relies on must actually be present in the
installed package, resolved the same way the library itself resolves them
(``importlib.resources``) -- not just present in the repo checkout.
"""
from __future__ import annotations

from importlib import resources

import pytest

import lqtpy
from lqtpy.atlases import list_atlases, load_atlas

_ATLAS_FILES = [
    "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz",
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.nii.gz",
    "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.csv",
    "aal.nii.gz",
]

_DATA_FILES = [
    "aal_labels.csv",
    "dkt31_cma_labels.csv",
    "ggseg_dk.csv.gz",
    "ggseg_schaefer100.csv.gz",
    "hcp842_tracts.npz",
]


@pytest.mark.parametrize("filename", _ATLAS_FILES)
def test_bundled_atlas_file_exists(filename):
    ref = resources.files("lqtpy").joinpath("data", "atlases", filename)
    assert ref.is_file(), f"missing bundled atlas file: data/atlases/{filename}"


@pytest.mark.parametrize("filename", _DATA_FILES)
def test_bundled_data_file_exists(filename):
    ref = resources.files("lqtpy").joinpath("data", filename)
    assert ref.is_file(), f"missing bundled data file: data/{filename}"


@pytest.mark.parametrize("key", list_atlases())
def test_atlas_loads_via_load_atlas(key):
    atlas = load_atlas(key)
    assert atlas.key == key
    assert atlas.img is not None
    assert len(atlas.labels) > 0
