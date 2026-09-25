"""Label-table schema v2 and atlas registration validation."""
from __future__ import annotations

import numpy as np
import pytest

import lqtpy
from lqtpy.atlases import OPTIONAL_FIELDS, RegionMeta, load_atlas
from lqtpy.errors import InvalidAtlasError

_TISSUES = {"Cortex", "Deep grey matter", "Cerebellum", "White matter", "CSF", "Brainstem", "Other"}


@pytest.mark.parametrize("key", ["aal", "schaefer", "dk", "jhu", "arterial"])
def test_bundled_atlas_tables_are_complete(key):
    atlas = load_atlas(key)
    vol = set(np.unique(atlas.label_array()).tolist()) - {0}
    assert vol == set(atlas.labels), key
    for lid, m in atlas.labels.items():
        assert m.DisplayName, (key, lid)
        assert m.Hemisphere in ("L", "R", "Midline"), (key, m)
        if key == "arterial":
            assert m.Territory in ("ACA", "MCA", "PCA", "VB", "Ventricle"), m
            continue
        assert m.Tissue in _TISSUES, (key, m)
        if key != "schaefer":
            assert m.Lobe, (key, m)


def test_arterial_atlas_origin_correction_is_pinned():
    """The released arterial atlas has no origin in its header; the bundled
    copy carries FSL's MNI152 affine shifted +2 mm (y) and +1 mm (z) -- see
    tools/build_extra_atlases.py. Without the fix the atlas sits ~120 mm off."""
    atlas = load_atlas("arterial")
    expected = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -124], [0, 0, 1, -71], [0, 0, 0, 1]])
    assert np.allclose(atlas.img.affine, expected)
    # Its lateral ventricles overlap DKT's lateral ventricles on the shared grid.
    from lqtpy.prepared import prepare
    import nibabel as nib
    art = atlas.label_array()
    v = nib.Nifti1Image(np.isin(art, [31, 32]).astype(np.uint8), atlas.img.affine)
    dk = load_atlas("dk")
    on = prepare(v).on_grid(dk.img).mask
    dkv = np.isin(dk.label_array(), [4, 43])
    assert 2 * (on & dkv).sum() / (on.sum() + dkv.sum()) > 0.68


def test_hemisphere_matches_region_centroid():
    for key in ("aal", "dk", "schaefer", "jhu", "arterial"):
        atlas = load_atlas(key)
        arr = atlas.label_array()
        aff = atlas.img.affine
        ijk = np.argwhere(arr > 0)
        labs = arr[arr > 0]
        xw = ijk @ aff[0, :3] + aff[0, 3]
        sums = np.bincount(labs, weights=xw)
        cnt = np.bincount(labs)
        for lid, m in atlas.labels.items():
            cx = sums[lid] / cnt[lid]
            if m.Hemisphere == "L":
                assert cx < 0, (key, m.RegionName)
            elif m.Hemisphere == "R":
                assert cx > 0, (key, m.RegionName)


def test_functional_labels_are_conservative():
    aal = load_atlas("aal")
    by = {m.RegionName: m for m in aal.labels.values()}
    assert by["Frontal_Inf_Oper_L"].Function == "Broca's area"
    assert by["Frontal_Inf_Oper_R"].Function is None
    assert by["Precentral_R"].Function == "primary motor cortex"
    assert by["Precentral_L"].display() == "left precentral gyrus"


def test_regionmeta_positional_backward_compat():
    m = RegionMeta("A", "G", "Frontal")
    assert (m.RegionName, m.Group, m.Lobe) == ("A", "G", "Frontal")
    assert all(getattr(m, f) is None for f in OPTIONAL_FIELDS[1:])


def _vol():
    shp = (20, 20, 20)
    aff = np.diag([2.0, 2, 2, 1])
    aff[:3, 3] = -20
    lab = np.zeros(shp, np.float32)
    lab[:10] = 1
    lab[10:] = 2
    return lab, aff


def test_register_rejects_negative_labels():
    lab, aff = _vol()
    lab[0, 0, 0] = -1
    with pytest.raises(InvalidAtlasError):
        lqtpy.atlas_from_arrays("neg_t", "neg", lab, aff, {1: "a", 2: "b"}, overwrite=True)


def test_register_rounds_float_labels_and_rejects_non_integers():
    lab, aff = _vol()
    lab[lab == 2] = 1.9999
    lqtpy.atlas_from_arrays("flt_t", "flt", lab, aff, {1: "a", 2: "b"}, overwrite=True)
    try:
        assert set(np.unique(load_atlas("flt_t").label_array())) == {1, 2}
    finally:
        lqtpy.atlases._CUSTOM.pop("flt_t", None)
        lqtpy.atlases._CACHE.pop("flt_t", None)
    lab[0, 0, 0] = 1.5
    with pytest.raises(InvalidAtlasError):
        lqtpy.atlas_from_arrays("flt_t", "flt", lab, aff, {1: "a"}, overwrite=True)


def test_register_keeps_schema_fields_from_dicts():
    lab, aff = _vol()
    lqtpy.atlas_from_arrays("dict_t", "d", lab, aff,
                            {1: {"RegionName": "A_L", "Group": "g", "Lobe": "Frontal",
                                 "Function": "primary motor cortex"}, 2: "B"}, overwrite=True)
    try:
        m = load_atlas("dict_t").labels[1]
        assert (m.Lobe, m.Function, m.Hemisphere) == ("Frontal", "primary motor cortex", "L")
        assert load_atlas("dict_t").labels[2].Hemisphere is None
    finally:
        lqtpy.atlases._CUSTOM.pop("dict_t", None)
        lqtpy.atlases._CACHE.pop("dict_t", None)
