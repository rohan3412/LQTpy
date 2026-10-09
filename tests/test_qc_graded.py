"""Graded QC: the synthetic library (CI subset), the result model, drawing
context, tiers, helpers, and contract tests mirroring how MRLatte and the
Lesion Registry call lqtpy."""
import json
import os

import numpy as np
import nibabel as nib
import pytest

import lqtpy
import qc_zoo
import synthetic as S
from lqtpy import qc
from lqtpy.qc_model import GRADES, QC_VERSION, DrawingContext, Finding, grade_of

_G = {None: 0, "minor": 1, "moderate": 2, "major": 3}


# ---------------------------------------------------------------- library
@pytest.mark.parametrize("case", [c for c in qc_zoo.all_cases() if c.fast], ids=lambda c: c.name)
def test_zoo(case):
    res = qc.assess(case.img(), **case.kwargs)
    ids = {f.id: f for f in res.findings}
    if case.expect and not case.undetectable:
        hit = next((ids[e] for e in case.expect if e in ids), None)
        assert hit is not None, (case.name, [f.id for f in res.findings])
        if case.grade:
            assert _G[hit.grade] >= _G[case.grade], (case.name, hit.grade)
        if case.near_mm is not None:
            assert hit.locations, case.name
            assert np.linalg.norm(np.subtract(hit.locations[0]["mm"], case.near_mm)) <= 15
    if case.forbid:
        bad = [f.id for f in res.findings if f.severity != "info" and f.id in case.forbid]
        assert not bad, (case.name, bad)


def test_grades_grow_with_extent():
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    rng = np.random.default_rng(1)
    grades = []
    for frac in (0.03, 0.15, 0.4):
        m = base & ~(rng.random(base.shape) < frac)
        f = next(f for f in qc.check(S.img(m)) if f.id == "holes")
        grades.append(_G[f.grade])
    assert grades == sorted(grades) and grades[-1] > grades[0]


# ---------------------------------------------------------------- model
def test_finding_is_backward_compatible_flag():
    f = qc.check(S.img(S.sphere((-40, 10, 10), 8) | S.sphere((20, 30, 20), 1)))[0]
    assert isinstance(f, qc.Flag)
    d = f.to_dict()
    for k in ("id", "severity", "message", "evidence", "suggestion", "category", "grade",
              "extent", "affected_pct", "locations", "fix"):
        assert k in d
    assert f.severity in ("info", "warn", "error")


def test_result_json_and_version():
    r = qc.assess(S.img(S.sphere((-40, 10, 10), 8)))
    d = json.loads(r.to_json())
    assert d["schema"] == "lqtpy.qc" and d["qc_version"] == QC_VERSION
    assert d["verdict"] in ("usable", "usable with caution", "fix before use", "unusable")
    assert 0 <= d["score"] <= 100


def test_verdicts_order():
    clean = qc.assess(S.img(S.sphere((-40, 10, 10), 8)))
    bad = qc.assess(S.img(S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE) & (np.abs(S.mni_world()[..., 2] - 15) < 0.5)))
    inverted = qc.assess(S.img(~S.sphere((-40, 10, 10), 8)))
    assert clean.verdict == "usable" and clean.score >= 95
    assert bad.verdict == "fix before use" and bad.score < clean.score
    assert inverted.verdict == "unusable" and inverted.score == 0


def test_grade_bands():
    assert grade_of("outside_brain", 0.5) is None
    assert grade_of("outside_brain", 3) == "minor"
    assert grade_of("outside_brain", 50) == "major"
    assert grade_of("fragmentation", 1000) == "moderate"  # plausibility ceiling
    assert set(GRADES) == {"minor", "moderate", "major"}


def test_check_returns_copies():
    img = S.img(S.sphere((-40, 10, 10), 8) | S.sphere((20, 30, 20), 1))
    p = lqtpy.prepare(img)
    a = qc.check(p)
    a[0].message = "changed"
    assert qc.check(p)[0].message != "changed"


# ---------------------------------------------------------------- context
def test_context_parsing_is_forgiving():
    c = DrawingContext.from_any({"drawn_slices": {"2": [3, 1, "x"], "9": [1]}, "drawing_axis": "7",
                                 "expected_side": "LEFT", "junk": 1})
    assert c is None or c.drawn_slices in (None, {2: [1, 3]}) or c.drawn_slices == {}
    c = DrawingContext.from_any({"drawn_slices": {"2": [3, 1]}, "drawing_axis": 2, "expected_side": "Left"})
    assert c.drawn_slices == {2: [1, 3]} and c.drawing_axis == 2 and c.expected_side == "left"
    assert DrawingContext.from_any("nonsense") is None


def test_context_finds_a_single_forgotten_slice_exactly():
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    zs = np.flatnonzero(base.any((0, 1)))
    gap = int(zs[len(zs) // 3])
    m = base.copy()
    m[:, :, gap] = False
    ctx = {"drawn_slices": {2: [int(z) for z in zs if z != gap]}, "drawing_axis": 2}
    f = next(f for f in qc.check(S.img(m), context=ctx) if f.id == "sparse_slices")
    assert f.evidence["gap_slices"] == [gap] and f.locations[0]["index"] == gap
    # the same slice marked as interpolated is not a gap
    ctx["interpolated_slices"] = {2: [gap]}
    m2 = base.copy()
    assert not [f for f in qc.check(S.img(m2), context=ctx) if f.id == "sparse_slices"]


def test_context_drawing_axis_protects_that_plane():
    w = S.mni_world()
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    sheet = (np.abs(w[..., 0] + 40) < 0.5) & (np.abs(w[..., 1] - 5) < 25) & (np.abs(w[..., 2] - 15) < 25)
    img = S.img(base | sheet)
    assert "strokes" in {f.id for f in qc.check(img)}
    assert "strokes" not in {f.id for f in qc.check(img, context={"drawing_axis": 0})}


# ---------------------------------------------------------------- tiers
def test_header_tier_recognises_lab_grids(tmp_path):
    for grid in qc_zoo.LAB_GRIDS:
        m, aff = qc_zoo.on_lab_grid(S.sphere((-40, 10, 10), 8), grid)
        r = qc.assess(nib.Nifti1Image(m.astype(np.uint8), aff), tier="header")
        assert r.options["grid"]["mni"], grid
        assert not [f for f in r.findings if f.id == "not_mni"]
    r = qc.assess(nib.Nifti1Image(np.zeros((64, 64, 30), np.uint8), np.eye(4)), tier="header")
    assert r.findings and r.findings[0].severity in ("warn", "error")


def test_header_tier_reports_qform_sform(tmp_path):
    img = S.img(S.sphere((-40, 10, 10), 8))
    q = img.affine.copy()
    q[1, 3] += 111
    img.set_qform(q, code=1)
    img.set_sform(img.affine, code=2)
    path = str(tmp_path / "m.nii.gz")
    nib.save(img, path)
    r = qc.assess(path, tier="header")
    f = next(f for f in r.findings if f.id == "header")
    assert f.extent["qform_sform_mm"] > 100


def test_fast_tier_is_fast():
    import time
    img = S.img(S.native_drawing(qc_zoo.C_BASE, qc_zoo.R_BASE, draw_every=2))
    qc.assess(img, tier="fast")
    t = time.perf_counter()
    qc.assess(lqtpy.prepare(S.img(S.native_drawing(qc_zoo.C_BASE, qc_zoo.R_BASE, draw_every=2))), tier="fast")
    assert time.perf_counter() - t < 1.0


# ---------------------------------------------------------------- helpers
def test_problem_mask_and_fix():
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    m = base & ~S.sphere(qc_zoo.C_BASE, 6)
    img = S.img(m)
    f = next(f for f in qc.check(img) if f.id == "holes")
    pm = qc.problem_mask(img, f)
    assert qc.PROBLEM_KIND["holes"] == "add" and pm.sum() > 500 and not (pm & m).any()
    new, rep = qc.fix(img, ["holes"])
    assert rep["applied"] == ["holes"] and rep["voxels_added"] == int(pm.sum())
    assert "holes" not in {x.id for x in qc.check(new)}
    _, rep2 = qc.fix(img, ["flat"])
    assert "flat" in rep2["skipped"]


def test_rank_prefers_the_clean_copy():
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    gappy = base.copy()
    gappy[:, :, ::2] = False
    ranked = qc.rank([S.img(gappy), S.img(base)])
    assert ranked[0]["index"] == 1 and ranked[0]["verdict"] == "usable"


def test_assess_folder_summary(tmp_path):
    nib.save(S.img(S.sphere((-40, 10, 10), 8)), str(tmp_path / "a.nii.gz"))
    nib.save(S.img(~S.sphere((-40, 10, 10), 8)), str(tmp_path / "b.nii.gz"))
    r = qc.assess_folder(str(tmp_path))
    assert r["summary"]["n"] == 2 and r["summary"]["verdicts"].get("unusable") == 1
    json.dumps(r, default=str)


# ---------------------------------------------------------------- contracts
def test_mrlatte_contract(tmp_path):
    """MRLatte's backend (backend/lesion_metrics.py:run_qc) calls
    ``lqtpy.qc.check(path, threshold=0.5, mni=True)`` on a drawing saved
    on the 0.74 mm template grid with label values 1-6, and reads
    ``f.to_dict()`` (id, severity, message, evidence, suggestion)."""
    m, aff = qc_zoo.on_lab_grid(S.native_drawing(qc_zoo.C_BASE, qc_zoo.R_BASE, draw_every=2), "0.74mm")
    lab = m.astype(np.uint8) * 3                 # pen value 3
    path = str(tmp_path / "drawing.nii.gz")
    nib.save(nib.Nifti1Image(lab, aff), path)
    flags = lqtpy.qc.check(path, threshold=0.5, mni=True)
    assert flags and all(f.severity in ("info", "warn", "error") for f in flags)
    d = flags[0].to_dict()
    for k in ("id", "severity", "message", "evidence", "suggestion"):
        assert k in d
    json.dumps([f.to_dict() for f in flags], default=str)
    gap = next(f for f in flags if f.id == "sparse_slices")
    for loc in gap.locations:
        assert all(0 <= v < s for v, s in zip(loc["ijk"], m.shape))
    assert hasattr(lqtpy, "qc") and hasattr(lqtpy.qc, "check")


def test_registry_contract(tmp_path):
    """The Lesion Registry reads analyze_single(...)["qc"] items (id,
    severity, message) and a thumbnail, and uses a disconnection-index
    path."""
    p = str(tmp_path / "les.nii.gz")
    nib.save(S.img(S.sphere((-40, 10, 10), 8)), p)
    r = lqtpy.analyze_single(p, out_dir=str(tmp_path / "o"), plots=True, viewer=False)
    for f in r["qc"]:
        assert {"id", "severity", "message"} <= set(f)
    assert r.get("thumb_uri", "").startswith("data:image/png")
    from lqtpy import datasets
    assert datasets.discon_index_target_path().endswith(".npz")
    assert callable(lqtpy.io.load_image)


# ---------------------------------------------------------------- review tool
def _tool(name):
    import importlib
    import sys
    tools = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    return importlib.import_module(name)


def test_review_selection_and_scoring(tmp_path, capsys):
    rv = _tool("qc_review")
    rows = []
    for i in range(300):
        fid = f"{i:010x}"
        f = [] if i % 3 == 0 else [{"id": ("holes", "specks", "flat")[i % 3 - 1 + (i % 2)],
                                     "sev": "warn", "grade": ("minor", "moderate")[i % 2]}]
        rows.append({"id": fid, "verdict": "usable", "findings": f})
    dropped = {r["id"] for r in rows[:40]}
    pick = rv.select(rows, 60, dropped=dropped)
    ids = [r["id"] for r in pick]
    assert len(ids) == len(set(ids)) == 60
    assert sum(1 for i in ids if i in dropped) >= 6
    assert any(not rv._problems(r) for r in pick)
    assert rv.select(rows, 60, dropped=dropped) == pick           # reproducible
    assert {rv.half(i) for i in ids} == {"tune", "validate"}
    labels = {r["id"]: {"ok": not rv._problems(r),
                        "problems": {f["id"]: f["grade"] for f in rv._problems(r)}} for r in pick}
    (tmp_path / "l.json").write_text(json.dumps({"labels": labels}))
    (tmp_path / "s.json").write_text(json.dumps(rows))
    rv.main(["score", str(tmp_path / "l.json"), "--survey", str(tmp_path / "s.json")])
    out = capsys.readouterr().out
    assert "grade agreement: exact 100%" in out and " 0.00" not in out
    assert "any problem:" in out and "(precision 1.00, recall 1.00)" in out
    # "Not fine" without details counts for masks, not for any type
    fid = next(r["id"] for r in pick if rv._problems(r))
    labels[fid] = {"ok": False, "bad": True, "problems": {}}
    (tmp_path / "l.json").write_text(json.dumps({"labels": labels}))
    rv.main(["score", str(tmp_path / "l.json"), "--survey", str(tmp_path / "s.json")])
    out = capsys.readouterr().out
    assert "(precision 1.00, recall 1.00)" in out and "60 labelled masks" in out


def test_review_pack_single_file(tmp_path):
    rv = _tool("qc_review")
    items = [{"id": "00000000aa", "half": "tune", "verdict": "usable", "vol_cm3": 1.0, "findings": []}]
    (tmp_path / "views").mkdir()
    (tmp_path / "index.html").write_text(rv._page(items), encoding="utf-8")
    for suffix in ("g", "s"):
        (tmp_path / "views" / f"00000000aa{suffix}.svg").write_text(
            '<svg xmlns="http://www.w3.org/2000/svg"><text>x</text></svg>', encoding="utf-8")
    rv.main(["pack", str(tmp_path)])
    page = (tmp_path / "review_phone.html").read_text(encoding="utf-8")
    assert "views/" not in page and "00000000aa" in page       # self-contained
    assert "<\/text>" in page and page.count("</script>") == 2   # svg can't close the data block
    assert 'KEY="lqtpy-qc-review-v1"' in page                    # same labels as the folder page


# ---------------------------------------------------------------- rules tuned on real masks
def _nonnote(m, **kw):
    return {f.id: f for f in qc.check(S.img(m), **kw) if f.severity != "info"}


def test_lesions_in_both_hemispheres_are_not_a_gap():
    """Two separate lesions taper toward the space between them; the slabs
    of a gap drawing end in flat faces. Only the second is a gap."""
    both = S.ellipsoid((-38, -10, 20), (22, 30, 24)) | S.ellipsoid((38, -10, 20), (22, 30, 24))
    assert "sparse_slices" not in _nonnote(both)
    slabs = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    zs = np.flatnonzero(slabs.any((0, 1)))
    slabs[:, :, [z for i, z in enumerate(zs) if (i // 3) % 2]] = False      # 3 on, 3 off
    assert "sparse_slices" in _nonnote(slabs)


def test_a_neck_between_two_blobs_is_not_sparse():
    w = S.mni_world()
    neck = (np.abs(w[..., 0] + 40) < 3) & (np.abs(w[..., 1] - 5) < 3) & (np.abs(w[..., 2] - 15) < 12)
    m = S.sphere((-40, 5, 0), 10) | S.sphere((-40, 5, 30), 10) | neck
    assert "sparse_slices" not in _nonnote(m)


def test_wide_plate_with_flat_faces_is_flat():
    """One thick slice drawn once: 8 mm thick, 60 x 34 mm across, cut flat
    on both faces. A lens of the same size that tapers is left alone."""
    w = S.mni_world()
    lens = S.ellipsoid((-35, 0, 20), (17, 30, 12))
    plate = lens & (np.abs(w[..., 2] - 20) < 4)
    f = _nonnote(plate).get("flat")
    assert f is not None and f.grade == "major"
    assert "flat" not in _nonnote(S.ellipsoid((-35, 0, 20), (17, 30, 5)))


def test_small_second_lesion_on_one_thick_slice_is_not_flat():
    """The plate rule applies only when plates hold most of the lesion (the
    outline-consistency rule is switched off here to test it alone)."""
    w = S.mni_world()
    small = S.ellipsoid((30, 20, 10), (16, 16, 6)) & (np.abs(w[..., 2] - 10) < 4.5)
    only_plate = dict(flat_min_consistency=2.0)
    assert "flat" not in _nonnote(S.sphere((-40, 5, 15), 20) | small, **only_plate)
    assert "flat" in _nonnote(small, **only_plate)      # alone, the same plate is the whole lesion


def test_stroke_through_a_lesion_is_found_where_it_sticks_out():
    """A one-voxel sagittal stroke crossing a lesion shows only as narrow
    strips beyond the outline; they are too small for the sheet rule. The
    drawing plane comes from the drawing tool here: on a smooth lesion with
    nothing else to go on, the stroke's own plane would be taken for it."""
    w = S.mni_world()
    base = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    stroke = (np.abs(w[..., 0] - qc_zoo.C_BASE[0]) < 0.5) & (np.abs(w[..., 1] - qc_zoo.C_BASE[1]) < 2) \
        & (np.abs(w[..., 2] - qc_zoo.C_BASE[2]) < qc_zoo.R_BASE[2] + 20)
    ctx = {"drawing_axis": 2}
    found = {f.id: f for f in qc.check(S.img(base | stroke), context=ctx)}
    assert "strokes" in found and any(s["kind"] == "line" for s in found["strokes"].evidence["strokes"])
    assert "strokes" not in {f.id for f in qc.check(S.img(base), context=ctx)}


def test_strokes_do_not_outvote_a_slice_by_slice_drawing():
    """Sagittal strokes through a jagged axial drawing: the axial plane
    still counts as the drawing plane, so the strokes are found."""
    w = S.mni_world()
    rng = np.random.default_rng(3)
    m = np.zeros(w.shape[:3], bool)
    for k in np.flatnonzero(S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE).any((0, 1))):
        dx, dy = rng.integers(-2, 3, 2)                      # each slice drawn a little differently
        m[:, :, k] = np.roll(S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)[:, :, k], (dx, dy), (0, 1))
    assert "strokes" not in {f.id for f in qc.check(S.img(m))}
    for x in (-6, 0, 6):
        m |= (np.abs(w[..., 0] - (qc_zoo.C_BASE[0] + x)) < 0.5) & (np.abs(w[..., 1] - qc_zoo.C_BASE[1]) < 2)             & (np.abs(w[..., 2] - qc_zoo.C_BASE[2]) < qc_zoo.R_BASE[2] + 20)
    assert "strokes" in {f.id for f in qc.check(S.img(m))}


def test_bilateral_lesions_are_not_called_a_mirror_copy():
    left = S.ellipsoid((-30, -20, 10), (10, 18, 12))
    similar = S.ellipsoid((31, -18, 11), (10, 17, 12))            # bilateral, alike, not identical
    msgs = [f.message for f in qc.check(S.img(left | similar)) if f.id == "plausibility"]
    assert not any("mirror" in m for m in msgs)
    copy = left | left[::-1]                                       # an exact flipped copy
    assert any("mirror" in f.message for f in qc.check(S.img(copy)) if f.id == "plausibility")


# ---------------------------------------------------------------- conveniences for drawing tools
def test_needs_attention_ignores_notes():
    clean = S.sphere((-40, 10, 10), 8)
    r = qc.assess(S.img(clean | S.sphere((20, 30, 20), 1)))           # one stray speck: a Note
    assert r.findings and not r.needs_attention and r.to_dict()["needs_attention"] is False
    gappy = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    gappy[:, :, ::2] = False
    assert qc.assess(S.img(gappy)).needs_attention


def test_finding_dict_has_ready_made_wording():
    gappy = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    gappy[:, :, ::2] = False
    d = next(f for f in qc.check(S.img(gappy)) if f.id == "sparse_slices").to_dict()
    assert d["label"] == "Problem, major" and d["where"].startswith("axial slice ")
    note = qc.check(S.img(S.sphere((-40, 10, 10), 8) | S.sphere((20, 30, 20), 1)))[0].to_dict()
    assert note["label"] == "Note"
    assert Finding("values", "info", "x").to_dict()["where"] == ""


def test_pen_colours_are_not_a_finding_when_the_tool_says_so():
    m = S.ellipsoid(qc_zoo.C_BASE, qc_zoo.R_BASE)
    lab = m.astype(np.uint8) * 3
    lab[m & (S.mni_world()[..., 1] > qc_zoo.C_BASE[1])] = 5             # two pens
    img = nib.Nifti1Image(lab, S.MNI_AFFINE)
    assert any(f.id == "values" and f.severity != "info" for f in qc.check(img))
    r = qc.assess(img, context={"labels_are_colours": True})
    assert not [f for f in r.findings if f.id == "values"] and not r.needs_attention
    assert r.context == {"labels_are_colours": True}
    # anything but True is ignored, like other bad context values
    assert any(f.id == "values" for f in qc.check(img, context={"labels_are_colours": "yes"}))
