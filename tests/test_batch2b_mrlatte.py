"""0.8.0 batch 2b, features for drawing tools (MRLatte) and the Registry:
capabilities, clinician advice, fix_plan, compare, QC sessions, ANTs warps.
All opt-in: nothing here changes an existing output."""
from __future__ import annotations

import json
import os
import re

import nibabel as nib
import numpy as np
import pytest

import synthetic as S
import lqtpy
from lqtpy import qc

C = [-35, 5, 15]


def _flawed():
    return S.sphere(C, 12) | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)])


def _gappy():
    m = S.sphere(C, 12)
    m[:, :, 75:95:3] = False            # every third axial slice skipped
    return m


# --------------------------------------------------------------------------- #
# capabilities
# --------------------------------------------------------------------------- #
def test_capabilities_is_json_and_lists_features():
    c = lqtpy.capabilities()
    json.dumps(c)
    assert c["lqtpy_version"] == lqtpy.__version__ and c["qc_version"]
    f = c["features"]
    for k in ("qc.assess", "qc.fix_plan", "qc.session", "qc.advice", "compare", "warp.ants"):
        assert f[k] is True, k
    assert "interpolate_gaps" in c["actions"] and "ants" in c["transform_formats"]


# --------------------------------------------------------------------------- #
# clinician advice
# --------------------------------------------------------------------------- #
def test_every_finding_id_has_advice_without_function_names():
    from lqtpy.qc_advice import ADVICE
    from lqtpy.qc_model import CATEGORY
    assert set(CATEGORY) <= set(ADVICE)
    for fid, (action, text) in ADVICE.items():
        assert "lqtpy" not in text and not re.search(r"\w\(", text), fid
        assert action and text.endswith("."), fid


def test_advice_is_opt_in_and_suggestion_unchanged():
    r = qc.assess(S.img(_gappy()))
    f = next(f for f in r.findings if f.id == "sparse_slices")
    assert f.action == "interpolate_gaps" and "lqtpy" not in f.advice
    assert "repair_gaps" in f.suggestion                         # unchanged
    plain = r.to_dict()
    assert "advice" not in plain["findings"][0]
    rich = r.to_dict(advice=True)["findings"]
    a = next(x for x in rich if x["id"] == "sparse_slices")["advice"]
    assert a == {"action": "interpolate_gaps", "text": f.advice, "fix_id": "sparse_slices",
                 "adds_voxels": True}


# --------------------------------------------------------------------------- #
# fix_plan
# --------------------------------------------------------------------------- #
def test_fix_plan_one_entry_per_fix_each_from_the_original():
    m = _gappy() | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)])
    plan = qc.fix_plan(S.img(m))
    by = {e["id"]: e for e in plan}
    assert set(by) == {"specks", "sparse_slices"}
    sp = by["specks"]
    assert sp["action"] == "remove_specks" and not sp["adds_voxels"] and sp["voxels_added"] == 0
    assert sp["removed"].sum() == sp["voxels_removed"] > 0 and m[sp["removed"]].all()
    gp = by["sparse_slices"]
    assert gp["adds_voxels"] and gp["added"].sum() == gp["voxels_added"] > 0 and not m[gp["added"]].any()
    # each entry is that fix alone: the same as qc.fix with only its id
    _, rep = qc.fix(S.img(m), ["sparse_slices"])
    assert np.array_equal(rep["added"], gp["added"])


def test_fix_plan_clean_mask_is_empty():
    assert qc.fix_plan(S.img(S.sphere(C, 12))) == []


# --------------------------------------------------------------------------- #
# compare
# --------------------------------------------------------------------------- #
def test_compare_identical_and_grown():
    a = S.img(S.sphere(C, 12))
    same = lqtpy.compare(a, a)
    assert same["overlap"]["dice"] == 1.0 and same["volume"]["change_mm3"] == 0.0
    assert same["grid"] == "own" and same["centroid_shift_mm"] == 0.0
    b = S.img(S.sphere(C, 14))
    c = lqtpy.compare(a, b)
    assert c["volume"]["change_pct"] > 0 and 0.5 < c["overlap"]["dice"] < 1
    assert c["overlap"]["only_a_mm3"] == 0.0
    json.dumps(c)


def test_compare_different_grids_uses_mni():
    a = S.img(S.sphere(C, 12))
    aff = S.MNI_AFFINE.copy()
    aff[:3, :3] *= 2
    shape = (91, 109, 91)
    ijk = np.indices(shape).reshape(3, -1).T
    w = ijk @ aff[:3, :3].T + aff[:3, 3]
    m2 = (((w - C) ** 2).sum(1) <= 144).reshape(shape)
    c = lqtpy.compare(a, nib.Nifti1Image(m2.astype(np.uint8), aff))
    assert c["grid"] == "mni_1mm" and c["overlap"]["dice"] > 0.85


def test_compare_disjoint_regions_gained_and_lost():
    c = lqtpy.compare(S.img(S.sphere(C, 8)), S.img(S.sphere([35, -20, 10], 8)))
    assert c["overlap"]["dice"] == 0.0
    assert c["regions"]["gained"] and c["regions"]["lost"]
    assert all(r["a_pct"] == 0.0 for r in c["regions"]["gained"])


# --------------------------------------------------------------------------- #
# QC session
# --------------------------------------------------------------------------- #
def test_session_matches_a_file_check_and_caches():
    m = _flawed()
    s = qc.Session(m.shape, S.MNI_AFFINE)
    assert s.check() is None                                      # empty drawing
    s.set_mask(m)
    r = s.check(tier="full")
    ref = qc.assess(S.img(m))
    assert r.verdict == ref.verdict and r.score == ref.score
    assert [f.to_dict() for f in r.findings] == [f.to_dict() for f in ref.findings]
    assert s.check(tier="full") is r                              # unchanged: cached
    v = s.version
    assert not s.set_slice(2, 0, np.zeros(m.shape[:2]))           # no change
    assert s.version == v


def test_session_edits():
    s = qc.Session(mask=S.img(S.sphere(C, 12)))
    before = s.check()
    assert before.verdict == "usable"
    speck = np.array([[60, 160, 110]])
    assert s.apply(added=speck)
    assert 110 in s.changed_slices[2]
    after = s.check()
    assert after is not before and s.changed_slices[2] == set()
    assert s.apply(removed=speck) and s.check().verdict == "usable"
    with pytest.raises(ValueError):
        s.set_slice(2, 5, np.zeros((3, 3)))
    assert not s.mask.flags.writeable


# --------------------------------------------------------------------------- #
# ANTs transforms
# --------------------------------------------------------------------------- #
def _native():
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    aff[:3, 3] = [-90, -126, -72]
    shape = (91, 109, 91)
    ijk = np.indices(shape).reshape(3, -1).T
    w = ijk @ aff[:3, :3].T + aff[:3, 3]
    m = (((w - [-30, 0, 15]) ** 2).sum(1) <= 100).reshape(shape).astype(np.uint8)
    return nib.Nifti1Image(m, aff)


def _native_to_mni():
    th = np.deg2rad(8)
    T = np.eye(4)
    T[:3, :3] = [[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]]
    T[:3, 3] = [5, -7, 3]
    return T


def _itk_affine(T, centre=(10.0, -20.0, 5.0)):
    """ITK parameters for native->MNI ``T`` (RAS): ITK maps fixed (MNI) to
    moving (native) in LPS, y = A (x - c) + t + c."""
    Ti = np.linalg.inv(T)
    F = np.diag([-1.0, -1.0, 1.0])
    A = F @ Ti[:3, :3] @ F
    c = np.asarray(centre)
    t = A @ c + F @ Ti[:3, 3] - c
    return np.concatenate([A.ravel(), t]), c


def test_ants_affine_mat_and_txt_match_the_matrix(tmp_path):
    from scipy.io import savemat
    from lqtpy.native import warp_to_mni
    nat, T = _native(), _native_to_mni()
    ref = np.asarray(warp_to_mni(nat, T).dataobj)
    params, c = _itk_affine(T)
    mat = tmp_path / "0GenericAffine.mat"
    savemat(str(mat), {"AffineTransform_float_3_3": params[:, None], "fixed": c[:, None]}, format="4")
    assert np.array_equal(np.asarray(warp_to_mni(nat, str(mat)).dataobj), ref)
    txt = tmp_path / "affine.txt"
    txt.write_text("#Insight Transform File V1.0\n#Transform 0\n"
                   "Transform: AffineTransform_double_3_3\n"
                   f"Parameters: {' '.join(map(str, params))}\n"
                   f"FixedParameters: {' '.join(map(str, c))}\n")
    assert np.array_equal(np.asarray(warp_to_mni(nat, str(txt)).dataobj), ref)


def test_ants_warp_then_affine(tmp_path):
    from scipy.io import savemat
    from lqtpy.native import warp_to_mni
    from lqtpy import cli
    nat, T = _native(), _native_to_mni()
    params, c = _itk_affine(T)
    mat = tmp_path / "0GenericAffine.mat"
    savemat(str(mat), {"AffineTransform_double_3_3": params[:, None], "fixed": c[:, None]}, format="4")
    mni = warp_to_mni(nat, T)
    u = np.zeros(mni.shape + (1, 3), np.float32)
    u[..., 0, 0] = -4.0                       # +4 mm in RAS x, stored as LPS
    warp = tmp_path / "1Warp.nii.gz"
    nib.save(nib.Nifti1Image(u, mni.affine), str(warp))
    got = np.asarray(warp_to_mni(nat, [str(warp), str(mat)]).dataobj)
    shift = np.eye(4)
    shift[0, 3] = 4
    want = np.asarray(warp_to_mni(nat, np.linalg.inv(np.linalg.inv(T) @ shift)).dataobj)
    assert np.array_equal(got, want)
    # the command line takes the same list as repeated --transform
    src = tmp_path / "native.nii.gz"
    nib.save(nat, str(src))
    out = tmp_path / "mni.nii.gz"
    assert cli.main(["warp", str(src), "--transform", str(warp), "--transform", str(mat),
                     "-o", str(out)]) == 0
    assert np.array_equal(np.asarray(nib.load(str(out)).dataobj), got)


def test_ants_refuses_inverse_and_composite(tmp_path):
    from lqtpy.errors import GeometryError
    from lqtpy.ants import load_ants
    inv = tmp_path / "1InverseWarp.nii.gz"
    nib.save(nib.Nifti1Image(np.zeros((2, 2, 2, 1, 3), np.float32), np.eye(4)), str(inv))
    with pytest.raises(GeometryError, match="inverse"):
        load_ants([str(inv)])
    h5 = tmp_path / "Composite.h5"
    h5.write_bytes(b"x")
    with pytest.raises(GeometryError, match="composite"):
        load_ants([str(h5)])


def test_fsl_style_text_mat_is_still_a_plain_matrix(tmp_path):
    from lqtpy.native import load_transform
    f = tmp_path / "ras.mat"
    np.savetxt(str(f), np.eye(4))
    kind, m = load_transform(str(f))
    assert kind == "affine" and np.array_equal(m, np.eye(4))
