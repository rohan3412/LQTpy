"""Comb-like, blocky and partly drawn slices (QC rules 1.2), the add-up
verdict, and the speed helpers behind 0.7.0 (each must give exactly what it
replaces)."""
from __future__ import annotations

import threading

import numpy as np
import pytest
from scipy import ndimage

import lqtpy
from lqtpy import qc
from lqtpy.qc_model import Finding, verdict_and_score

import synthetic as S

CENTRE, RADII = (-35.0, 0.0, 20.0), (18.0, 22.0, 16.0)


def _ellipsoid(scale=None, z_of=None):
    """An ellipsoid drawn slice by slice along the axial axis (array axis 2):
    ``scale(k)`` scales slice k's outline, ``z_of(k)`` picks which height's
    outline slice k gets (a copied outline: blocky)."""
    w = S.mni_world()
    out = np.zeros(S.MNI_SHAPE, bool)
    zk = w[0, 0, :, 2]
    for k in range(S.MNI_SHAPE[2]):
        z = z_of(k) if z_of else zk[k]
        dz = (z - CENTRE[2]) / RADII[2]
        if abs(dz) > 1:
            continue
        f = scale(k) if scale else 1.0
        dx = (w[:, :, k, 0] - CENTRE[0]) / (RADII[0] * f)
        dy = (w[:, :, k, 1] - CENTRE[1]) / (RADII[1] * f)
        out[:, :, k] = dx ** 2 + dy ** 2 + dz ** 2 <= 1
    return out


def _find(res, fid):
    return next((f for f in res.findings if f.id == fid), None)


@pytest.fixture(scope="module")
def smooth():
    return qc.assess(S.img(_ellipsoid()))


def test_smooth_lesion_has_no_drawing_style_findings(smooth):
    assert not {f.id for f in smooth.findings} & {"comb", "blocky", "partial_gaps"}


def test_comb_like_drawing():
    rng = np.random.default_rng(3)
    jitter = rng.uniform(0.7, 1.3, S.MNI_SHAPE[2])
    res = qc.assess(S.img(_ellipsoid(scale=lambda k: jitter[k])))
    f = _find(res, "comb")
    assert f is not None and f.extent["plane"] == "axial"
    assert f.extent["comb_ratio"] > 2.5 and f.grade in ("minor", "moderate")
    assert f.locations and f.locations[0]["plane"] != "axial"   # look across the slices


def test_blocky_drawing():
    w = S.mni_world()
    zk = w[0, 0, :, 2]
    block = 8                                       # 8 mm blocks on the 1 mm grid

    def z_of(k):
        k0 = (k // block) * block + block // 2
        return zk[min(k0, len(zk) - 1)]
    res = qc.assess(S.img(_ellipsoid(z_of=z_of)))
    f = _find(res, "blocky")
    assert f is not None and f.extent["plane"] == "axial"
    assert f.extent["blocky_pct"] >= 35 and f.extent["block_mm"] >= 4
    assert f.grade in ("minor", "moderate")


def test_partly_drawn_slices_are_found_and_fixed():
    m = _ellipsoid()
    k = int(np.flatnonzero(m.any(axis=(0, 1))).mean())
    x_of = S.mni_world()[:, 0, 0, 0]
    half = x_of < CENTRE[0]                        # the left half of two slices left undrawn
    for kk in (k, k + 1):
        m[half, :, kk] = False
    res = qc.assess(S.img(m))
    f = _find(res, "partial_gaps")
    assert f is not None and f.severity == "warn" and f.grade is not None
    assert f.extent["plane"] == "axial" and {k, k + 1} <= {loc["index"] for loc in f.locations}
    fixed, rep = qc.fix(S.img(m), ["partial_gaps"])
    assert "partial_gaps" in rep["applied"] and rep["voxels_added"] > 0
    assert _find(qc.assess(fixed), "partial_gaps") is None


def _f(grade, severity="warn"):
    return Finding("x", severity, "", grade=grade)


@pytest.mark.parametrize("grades, verdict", [
    (["moderate"], "usable with caution"),
    (["moderate", "minor"], "usable with caution"),
    (["minor", "minor", "minor"], "usable with caution"),
    (["moderate", "moderate"], "fix before use"),
    (["moderate", "minor", "minor"], "fix before use"),
    (["minor"] * 4, "fix before use"),
])
def test_checks_add_up(grades, verdict):
    assert verdict_and_score([_f(g) for g in grades])[0] == verdict


def test_notes_never_add_up():
    assert verdict_and_score([_f(None, "info")] * 10)[0] == "usable"


# --------------------------------------------------------------------------- #
# Speed helpers: same results as what they replace
# --------------------------------------------------------------------------- #
def test_atlas_counts_before_labels_do_not_deadlock():
    """region_voxels() on a fresh atlas builds the label array under the same
    lock (a hang in 0.7.0 development, latent in 0.6.1)."""
    lqtpy.cache_clear()
    done = []
    t = threading.Thread(target=lambda: done.append(lqtpy.load_atlas("jhu").region_voxels().sum()))
    t.start()
    t.join(timeout=120)
    assert not t.is_alive() and done


def test_grid_mask_voxel_list_matches_mask():
    p = lqtpy.prepare(S.img(S.sphere((-40, 10, 5), 12)))
    at = lqtpy.load_atlas("aal")                    # a different grid: resampled
    gm = p.on_grid(at.img)
    assert np.array_equal(gm.flat, np.flatnonzero(gm.mask))
    assert np.array_equal(gm.ijk(), np.argwhere(gm.mask))
    labels = at.label_array()
    assert np.array_equal(gm.take(labels), labels[gm.mask])
    same = p.on_grid(lqtpy.load_atlas("dk").img)    # the lesion's own grid
    assert np.array_equal(same.flat, np.flatnonzero(p.mask)) and same.n_voxels == p.n_voxels


def test_part_helpers_match_full_scans():
    from lqtpy.morphometry import component_boxes, components, label_centres, part_voxels
    p = lqtpy.prepare(S.img(S.blobs([((-40, 10, 5), 8), ((30, -20, 10), 5), ((0, 40, 30), 3)])))
    labels, lo, sizes = components(p)
    boxes = component_boxes(p)
    for k in range(1, len(sizes) + 1):
        assert np.array_equal(part_voxels(labels, boxes, k, lo), np.argwhere(labels == k) + lo)
    want = ndimage.center_of_mass(np.ones(labels.shape), labels, range(1, len(sizes) + 1))
    assert np.array_equal(label_centres(labels, len(sizes)), np.asarray(want))


def test_derived_lesions_match_prepare():
    from lqtpy.mirror import mirror
    p = lqtpy.prepare(S.img(S.sphere((-40, 10, 5), 12)))
    at = lqtpy.load_atlas("aal")
    mirrored = lqtpy.prepare(mirror(p), threshold=0.0)
    derived = p._derived(affine=np.diag([-1.0, 1, 1, 1]) @ p.affine)
    assert np.array_equal(derived.on_grid(at.img).flat, mirrored.on_grid(at.img).flat)
    part = p.mask & (np.arange(p.shape[0]) < 120)[:, None, None]
    via_image = lqtpy.prepare(S.img(part), threshold=0.0)
    d = p._derived(part)
    assert (d.n_voxels, d.bbox) == (via_image.n_voxels, via_image.bbox)
    assert np.array_equal(d.on_grid(at.img).flat, via_image.on_grid(at.img).flat)
