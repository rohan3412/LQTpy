"""Group features: metadata, flip check, duplicates, outliers, comparison,
anonymised output, incremental and parallel runs, PDF bundle, rerun."""
import csv
import json
import os

import numpy as np
import pytest

import lqtpy
import synthetic as syn
from lqtpy import cohort
from lqtpy.metadata import expected_sides, load_metadata


@pytest.fixture(scope="module")
def folder(tmp_path_factory):
    import nibabel as nib
    d = tmp_path_factory.mktemp("cohort")
    les = {
        "Doe_John_01": syn.sphere((-40, 10, 10), 10),
        "copy_of_01": syn.sphere((-40, 10, 10), 10),
        "right_02": syn.sphere((40, 5, 15), 9),
        "left_03": syn.sphere((-30, -20, 20), 7),
        "left_04": syn.sphere((-42, 0, 5), 8),
    }
    for k, m in les.items():
        nib.save(syn.img(m), str(d / f"{k}.nii.gz"))
    with open(d / "meta.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["subject", "aphasia", "age", "lesion_side"])
        w.writerows([["Doe_John_01", "yes", 60, "left"], ["copy_of_01", "yes", 61, "right"],
                     ["right_02", "no", 70, "right"], ["left_03", "no", 55, "left"],
                     ["left_04", "yes", 50, "left"], ["nobody", "no", 1, ""]])
    return d


def test_metadata_matching_and_types(folder):
    names = ["Doe_John_01", "copy_of_01", "right_02"]
    m = load_metadata(str(folder / "meta.csv"), names)
    assert m["subject_column"] == "subject"
    assert m["by_subject"]["right_02"]["age"] == 70          # numeric column
    assert "nobody" in m["unmatched_rows"]
    sides, col, symptom = expected_sides(m)
    assert col == "lesion_side" and not symptom and sides["copy_of_01"] == "right"


def test_symptom_side_means_opposite_lesion():
    m = load_metadata([{"subject": "a", "hemiparesis_side": "R"}], ["a"])
    sides, col, symptom = expected_sides(m)
    assert symptom and sides["a"] == "left"


def test_duplicates_and_signatures():
    shape, aff = syn.MNI_SHAPE, syn.MNI_AFFINE
    a = np.flatnonzero(syn.sphere((-40, 10, 10), 12).ravel())
    b = a[::50].copy()
    b = np.setdiff1d(a, b)          # a re-export that lost 2% of its voxels
    c = np.flatnonzero(syn.sphere((30, 10, 10), 8).ravel())
    sigs = [cohort.signature(n, i, shape, aff) for n, i in (("a", a), ("a2", a), ("b", b), ("c", c))]
    d = cohort.find_duplicates(sigs)
    kinds = {(x["a"], x["b"]): x["kind"] for x in d}
    assert kinds[("a", "a2")] == "identical"
    assert any(set(k) == {"a", "b"} and v == "near-identical" for k, v in kinds.items())
    assert not any("c" in k for k in kinds)


def test_outliers():
    subs = [{"name": f"s{i}", "morphometry": {"volume_mm3": 1000 + 10 * i,
                                              "center_of_mass_mm": [-40 + i * 0.5, 10, 10]}}
            for i in range(12)]
    subs.append({"name": "far", "morphometry": {"volume_mm3": 1050, "center_of_mass_mm": [60, -80, 40]}})
    subs.append({"name": "huge", "morphometry": {"volume_mm3": 400000, "center_of_mass_mm": [-40, 10, 10]}})
    out = cohort.find_outliers(subs)
    assert set(out) == {"far", "huge"}
    assert out["huge"][0]["feature"] == "volume"


def test_compare_and_fdr():
    per = [[{"R": "x", "v": 50.0}], [{"R": "x", "v": 40.0}], [{"R": "x", "v": 60.0}],
           [], [], [{"R": "y", "v": 5.0}]]
    rows = cohort.compare(per, ["a", "a", "a", "b", "b", "b"], "a", "b", "R", "v")
    x = next(r for r in rows if r["Name"] == "x")
    assert x["NAffectedA"] == 3 and x["NAffectedB"] == 0 and x["DiffMean"] == 50.0
    assert x["PMannWhitney"] < 0.2 and 0 <= x["QMannWhitney"] <= 1
    assert cohort.fdr([0.01, 0.04, 0.03]) == [0.03, 0.04, 0.04]


def test_group_run_end_to_end(folder, tmp_path):
    out = tmp_path / "grp"
    res = lqtpy.analyze_group(str(folder), out_dir=str(out), plots=True,
                              metadata=str(folder / "meta.csv"), compare="aphasia",
                              group_by="aphasia", anonymise=True, incremental=True, viewer=False)
    names = [s["name"] for s in res["subjects"]]
    assert all(n.startswith("sub-") for n in names)
    # anonymised: no original name anywhere in the outputs
    for root, _, files in os.walk(out):
        for f in files:
            if f == "anonymisation_key.csv" or f.endswith(".json") and ".lqtpy_cache" in root:
                continue
            assert "Doe_John" not in f
            if f.endswith((".html", ".csv", ".txt", ".json")):
                with open(os.path.join(root, f), encoding="utf-8", errors="ignore") as fh:
                    assert "Doe_John" not in fh.read(), f
    with open(res["anonymisation_key"], encoding="utf-8") as fh:
        key = {r["Name"]: r["ID"] for r in csv.DictReader(fh)}
    assert "Doe_John_01" in key
    by = {s["name"]: s for s in res["subjects"]}
    dup = [s for s in res["subjects"] if any(f["id"] == "duplicate" for f in s["group_flags"])]
    assert len(dup) == 2
    assert any(f["id"] == "side_mismatch" for f in by[key["copy_of_01"]]["qc"])
    assert res["comparison"]["n_a"] + res["comparison"]["n_b"] == 5
    assert os.path.exists(out / "group_comparison_arterial_networks.csv")
    assert os.path.exists(out / "data_dictionary.csv")
    page = open(res["report_path"], encoding="utf-8").read()
    assert 'id="subj-search"' in page and 'data-rollup="arterial"' in page
    assert 'id="lqtpy-run"' in page and "Cohort checks" in page
    # incremental: a second run reuses every subject
    res2 = lqtpy.analyze_group(str(folder), out_dir=str(out), plots=True, viewer=False,
                               metadata=str(folder / "meta.csv"), anonymise=True, incremental=True)
    inc = next(m for m in res2["manifest"]["modules"] if m["id"] == "incremental")
    assert inc["reason"] == "5 of 5 subjects reused"
    assert [s["name"] for s in res2["subjects"]] == names   # same IDs from the key file


def test_parallel_matches_serial(folder):
    files = sorted(str(p) for p in folder.glob("*.nii.gz"))[:3]
    a = lqtpy.analyze_group(files, write=False)
    b = lqtpy.analyze_group(files, write=False, n_jobs=2)
    assert [s["morphometry"] for s in a["subjects"]] == [s["morphometry"] for s in b["subjects"]]
    assert a["atlases"][0]["df"] == b["atlases"][0]["df"]


def test_rerun_from_report(tmp_path):
    import nibabel as nib
    from lqtpy.repro import read_run_record, rerun
    p = tmp_path / "les.nii.gz"
    nib.save(syn.img(syn.sphere((-40, 10, 10), 8)), str(p))
    res = lqtpy.analyze_single(str(p), out_dir=str(tmp_path / "a"), plots=True, profile="research",
                               viewer=False)
    rec = read_run_record(res["report_path"])
    assert rec["settings"]["profile"] == "research" and len(rec["inputs"][0]["sha256"]) == 64
    again = rerun(res["report_path"], out_dir=str(tmp_path / "b"))
    assert again["morphometry"] == res["morphometry"]
    assert again["description"]["text"] == res["description"]["text"]
    # a moved file is found by checksum
    moved = tmp_path / "moved"
    moved.mkdir()
    os.replace(p, moved / "x.nii.gz")
    again = rerun(res["report_path"], out_dir=str(tmp_path / "c"), search=[str(moved)])
    assert again["morphometry"]["n_voxels"] == res["morphometry"]["n_voxels"]


def test_legacy_positional_flags_warn(tmp_path):
    les = syn.img(syn.sphere((-40, 10, 10), 6))
    with pytest.warns(DeprecationWarning):
        r = lqtpy.analyze_single(les, None, None, 0.5, True, False)
    assert r["morphometry"]["n_voxels"] > 0
    r = lqtpy.analyze_single(lesion_path=les, write=False)
    assert r["name"] == "lesion"
