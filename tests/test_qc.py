"""QC detectors against realistic synthetic artefacts (tests/synthetic.py).

Each artefact is drawn on thick native slices and warped to MNI with
trilinear interpolation, as registration pipelines do, so the detectors are
tested on smeared artefacts, not idealised ones. The negative cases matter
as much: clean, overdrawn, multi-lesion and stacked-round lesions must not
raise warnings.
"""
from __future__ import annotations

import numpy as np
import pytest

import lqtpy
from lqtpy import qc

import synthetic as S

C = [-35, 5, 15]
R = (14, 18, 16)


def _ids(lesion, **kw):
    return {f.id for f in qc.check(S.img(lesion) if isinstance(lesion, np.ndarray) else lesion, **kw)
            if f.severity != "info"}


@pytest.fixture(scope="module")
def drawings():
    return {
        "full5": S.native_drawing(C, R),
        "every2_5": S.native_drawing(C, R, draw_every=2),
        "every3_5": S.native_drawing(C, R, draw_every=3),
        "every3_3": S.native_drawing(C, R, slice_mm=3, draw_every=3),
        "single5": S.native_drawing(C, R, only_slice=0),
        "outline5": S.native_drawing(C, R, outline=True),
    }


@pytest.mark.parametrize("key", ["every2_5", "every3_5", "every3_3"])
def test_sparse_slice_drawings_are_flagged(drawings, key):
    flags = qc.check(S.img(drawings[key]))
    sparse = [f for f in flags if f.id == "sparse_slices"]
    assert sparse, [f.id for f in flags]
    assert sparse[0].evidence["plane"] == "axial"
    assert not any(f.id == "flat" for f in flags)  # slabs aren't reported twice


def test_full_thick_slice_drawing_is_clean(drawings):
    assert _ids(drawings["full5"]) == set()


def test_single_slice_is_flat(drawings):
    flags = qc.check(S.img(drawings["single5"]))
    flat = [f for f in flags if f.id == "flat"]
    assert flat and flat[0].evidence["components"][0]["plane"] == "axial"


def test_outline_is_hollow_not_specks(drawings):
    ids = _ids(drawings["outline5"])
    assert "hollow" in ids
    assert "specks" not in ids


@pytest.mark.parametrize("name,mask", [
    ("sphere", lambda: S.sphere(C, 15)),
    ("lacune", lambda: S.sphere([-12, -18, 8], 3)),
    ("elongated", lambda: S.ellipsoid([-25, -10, 25], (4, 20, 4))),
    ("two lesions", lambda: S.sphere([-30, 0, 40], 8) | S.sphere([-30, 0, 5], 8)),
    ("stacked round", lambda: S.blobs([([-25, -10, z], 3) for z in (10, 18, 26, 34)])),
    ("overdrawn rim", lambda: S.sphere([-60, 0, 10], 12)),
    ("joined blobs", lambda: S.blobs([([31, -29, 14], 9), ([31, -30, 7], 10), ([51, -22, 12], 9)])),
])
def test_no_false_alarms(name, mask):
    assert _ids(mask()) == set(), name


def test_specks_are_info_for_a_couple_of_stray_marks():
    m = S.sphere(C, 12) | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)])
    flags = qc.check(S.img(m))
    specks = [f for f in flags if f.id == "specks"]
    assert specks and specks[0].severity == "info" and specks[0].evidence["n_specks"] == 2


def test_chunk_outside_brain_is_flagged():
    m = S.sphere(C, 12) | S.sphere([0, -100, -55], 8)
    assert "outside_brain" in _ids(m)


def test_chunk_on_the_image_edge_is_reported_once():
    """A chunk touching the image border is an edge artefact; it isn't also
    counted as lesion outside the brain."""
    m = S.sphere(C, 12) | S.sphere([0, -100, -65], 8)
    ids = _ids(m)
    assert "edge_artifact" in ids and "outside_brain" not in ids


def test_not_in_mni_space():
    far = S.MNI_AFFINE.copy()
    far[:3, 3] += [0, 180, 0]
    flags = qc.check(S.img(S.sphere([0, 0, 0], 15), far))
    assert flags[0].id == "not_mni" and flags[0].severity == "error"


def test_expected_side_mismatch():
    m = S.sphere([40, 0, 10], 10)  # right hemisphere
    assert "side_mismatch" in _ids(m, expected_side="left")
    assert "side_mismatch" not in _ids(m, expected_side="right")


def test_header_zoom_mismatch_flagged_and_volume_uses_affine():
    import nibabel as nib
    im = nib.Nifti1Image(S.sphere(C, 10).astype(np.uint8), S.MNI_AFFINE)
    im.header.set_zooms((2.0, 2.0, 2.0))
    assert "header" in _ids(im)
    assert lqtpy.lesion_stats(im)["voxel_volume_mm3"] == 1.0


def test_gap_clusters_merge_slabs_but_not_separate_lesions(drawings):
    gapped = lqtpy.prepare(S.img(drawings["every2_5"]))
    assert lqtpy.lesion_stats(gapped)["n_components"] > 1
    assert len(qc.clusters(gapped)["sizes"]) == 1
    two = lqtpy.prepare(S.img(S.sphere([-30, 0, 40], 8) | S.sphere([-30, 0, 5], 8)))
    assert len(qc.clusters(two)["sizes"]) == 2


def test_repair_gaps_restores_volume_and_is_opt_in(drawings):
    true = lqtpy.prepare(S.img(drawings["full5"])).mask
    p = lqtpy.prepare(S.img(drawings["every2_5"]))
    before = p.n_voxels
    img, rep = qc.repair_gaps(p)
    assert p.n_voxels == before            # the input is untouched
    repaired = np.asarray(img.dataobj) > 0
    assert (repaired | p.mask).sum() == repaired.sum()  # superset of the input
    dice = 2 * (repaired & true).sum() / (repaired.sum() + true.sum())
    assert rep["repaired"] and dice > 0.85
    assert "sparse_slices" not in _ids(img)


def test_repair_is_a_no_op_without_gaps(drawings):
    img, rep = qc.repair_gaps(S.img(drawings["full5"]))
    assert rep["repaired"] is False and rep["voxels_added"] == 0


def test_unknown_option_rejected():
    with pytest.raises(TypeError):
        qc.check(S.img(S.sphere(C, 5)), not_an_option=1)


def test_image_edge_fragments_are_artefacts_and_cleanable():
    """Seen in real exports: fragments on the outermost slice of the image,
    outside the brain. Flagged, excluded from the other detectors, and
    removable only on request."""
    m = S.sphere(C, 12)
    m[40:60, -1, 80:100] = True          # last coronal slice: a flat edge sheet
    m[120:130, -1, 80:90] = True
    flags = {f.id: f for f in qc.check(S.img(m))}
    assert "edge_artifact" in flags and flags["edge_artifact"].evidence["n_fragments"] == 2
    assert "flat" not in flags and "specks" not in flags
    img, rep = qc.clean(S.img(m))
    assert rep["removed_edge_fragments"] == 2
    assert int(np.asarray(img.dataobj).sum()) == int(S.sphere(C, 12).sum())
    assert "edge_artifact" not in _ids(img)


def test_lesion_touching_edge_inside_brain_is_not_an_artefact():
    import nibabel as nib
    # A grid cropped through the brain: a lesion reaching the crop edge is real.
    aff = S.MNI_AFFINE.copy()
    m = S.sphere([-30, 0, 10], 10)[:, :126, :]   # crop at y = 0 mm, through the lesion
    assert "edge_artifact" not in {f.id for f in qc.check(nib.Nifti1Image(m.astype(np.uint8), aff))}


def test_header_disagreement_is_info_when_sform_is_right():
    import nibabel as nib
    im = nib.Nifti1Image(S.sphere(C, 8).astype(np.uint8), S.MNI_AFFINE)
    q = S.MNI_AFFINE.copy()
    q[:3, 3] += [0, 0, 110]
    im.set_qform(q, code=1)
    im.set_sform(S.MNI_AFFINE, code=2)
    h = [f for f in qc.check(im) if f.id == "header"]
    assert h and h[0].severity == "info"


# ---------------------------------------------------------------- strokes / slab
def _crosshair_drawing():
    """A lesion drawn on one coronal slab, joined to a thin sagittal sheet and a
    thin axial sheet (strokes on the wrong planes), like a real mask that
    passed every check before 0.5.1."""
    w = S.mni_world()
    x, y, z = w[..., 0], w[..., 1], w[..., 2]
    disc = (np.abs(y + 51) <= 3.5) & (((x + 12) / 26) ** 2 + ((z - 5) / 30) ** 2 <= 1)
    sagittal = (np.abs(x + 10) <= 0.5) & (y >= -75) & (y <= -30) & (z >= -30) & (z <= 35)
    axial = (np.abs(z - 2) <= 0.5) & (y >= -60) & (y <= -40) & (x >= -35) & (x <= 15)
    return disc | sagittal | axial


def test_strokes_and_slab_found():
    from lqtpy.qc import check, clean
    m = _crosshair_drawing()
    flags = {f.id: f for f in check(S.img(m))}
    assert "strokes" in flags and "flat" in flags
    assert "coronal" in flags["flat"].message and "front to back" in flags["flat"].message
    img, rep = clean(S.img(m), strokes=True)
    assert rep["removed_strokes"] >= 1
    # what's left is still a single-slab drawing
    assert "flat" in {f.id for f in check(img)}


def test_no_strokes_or_slab_on_normal_lesions():
    from lqtpy.qc import check
    for m in (S.sphere((-40, 10, 10), 10), S.blobs([((-40, 10, 10), 14), ((-35, -10, 15), 10)]),
              S.native_drawing((-40, 10, 10), (12, 10, 9)),
              S.ellipsoid((-30, -20, 20), (4, 18, 12))):   # thin but real (8 mm)
        ids = {f.id for f in check(S.img(m))}
        assert "strokes" not in ids, ids


def test_jagged_slice_by_slice_drawing_is_not_strokes():
    """A lesion drawn axial slice by axial slice with outlines that jump
    between slices leaves wide one-voxel rims in the drawing plane; those
    are ordinary drawing, not strokes."""
    from lqtpy.qc import check
    rng = np.random.default_rng(3)
    w = S.mni_world()
    m = np.zeros(S.MNI_SHAPE, bool)
    for z in range(0, 30):
        rx, ry = rng.uniform(8, 22), rng.uniform(8, 22)
        zz = w[..., 2] == z
        m |= zz & (((w[..., 0] + 40) / rx) ** 2 + ((w[..., 1] - 5) / ry) ** 2 <= 1)
    assert "strokes" not in {f.id for f in check(S.img(m))}
