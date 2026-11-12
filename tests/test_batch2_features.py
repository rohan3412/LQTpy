"""0.8.0 batch 2 features. All opt-in: defaults give the same output as before."""
from __future__ import annotations

import concurrent.futures as cf
import os

import nibabel as nib
import numpy as np
import pytest

import synthetic as S


def _sq(x):
    return x * x


# --------------------------------------------------------------------------- #
# -j auto
# --------------------------------------------------------------------------- #
def test_resolve_jobs():
    from lqtpy._jobs import AUTO_MAX, resolve_jobs
    assert resolve_jobs(1, 50) == 1
    assert resolve_jobs("auto", 2) == 1                    # a few files: one process
    k = resolve_jobs("auto", 100)
    assert 1 <= k <= AUTO_MAX and k <= max(1, (os.cpu_count() or 2) - 1)
    assert resolve_jobs(4, 2) == 2
    assert resolve_jobs(-1, 3) == min(os.cpu_count() or 2, 3)
    assert resolve_jobs("AUTO", 1) == 1


def test_parse_jobs_rejects_words():
    from lqtpy._jobs import parse_jobs
    assert parse_jobs("3") == 3 and parse_jobs(" auto ") == "auto"
    with pytest.raises(ValueError):
        parse_jobs("many")


def test_falls_back_to_one_process_when_pool_cannot_start(monkeypatch, capsys):
    from lqtpy import _jobs

    class Broken:
        def __init__(self, *a, **k):
            raise OSError("no processes here")

    monkeypatch.setattr(cf, "ProcessPoolExecutor", Broken)
    seen = []
    out = _jobs.run_tasks(_sq, range(6), 3, lambda i, r: seen.append(i))
    assert out == [0, 1, 4, 9, 16, 25] and sorted(seen) == list(range(6))
    assert "one process" in capsys.readouterr().err


def test_pool_results_keep_input_order():
    from lqtpy._jobs import run_tasks
    assert run_tasks(_sq, range(8), 2) == [i * i for i in range(8)]


def test_config_accepts_auto(tmp_path):
    from lqtpy.config import load_config
    f = tmp_path / "lqtpy.toml"
    f.write_text('[defaults]\njobs = "auto"\n')
    assert load_config(str(f))[0]["jobs"] == "auto"
    f.write_text('[defaults]\njobs = "lots"\n')
    with pytest.raises(ValueError):
        load_config(str(f))


def test_cli_accepts_auto():
    from lqtpy.cli import build_parser
    a = build_parser().parse_args(["group", "x", "-j", "auto"])
    assert a.jobs == "auto"
    with pytest.raises(SystemExit):
        build_parser().parse_args(["group", "x", "-j", "lots"])


# --------------------------------------------------------------------------- #
# lqtpy.summary()
# --------------------------------------------------------------------------- #
def test_summary_shape_is_fixed_and_json_safe():
    import json
    import lqtpy
    from lqtpy.summary import SCHEMA, SUMMARY_VERSION
    m = S.sphere([-35, 5, 15], 10) | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)])
    s = lqtpy.summary(S.img(m), atlases=["dk"], top=3)
    assert s["schema"] == SCHEMA and s["summary_version"] == SUMMARY_VERSION
    assert set(s) == {"schema", "summary_version", "lqtpy_version", "qc_version", "created",
                      "input", "settings", "volume", "location", "quality", "description",
                      "tissue", "regions", "coverage_pct", "tracts", "disconnection",
                      "key_structures", "border_zones"}
    assert s["location"]["side"] == "left" and len(s["regions"]["dk"]) <= 3
    assert any(f["id"] == "specks" for f in s["quality"]["findings"])
    json.loads(json.dumps(s, allow_nan=False))        # plain JSON types only
    assert "path" not in json.dumps(s["input"])


def test_summary_reuses_a_result_and_marks_skipped_parts_null():
    import lqtpy
    r = lqtpy.analyze_single(S.img(S.sphere([-35, 5, 15], 10)), atlases=["dk"], write=False,
                             tracts=False, qc=False, describe=False, disconnection=False)
    s = lqtpy.summary(r)
    assert s["tracts"] is None and s["quality"] is None and s["description"] is None
    assert s["disconnection"] is None


# --------------------------------------------------------------------------- #
# Show flagged voxels (flagged_voxels=True / --flagged-voxels)
# --------------------------------------------------------------------------- #
def _flawed_img():
    return S.img(S.sphere([-35, 5, 15], 12) | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)]))


def test_flagged_viewer_only_when_asked():
    pytest.importorskip("nilearn")
    from lqtpy.report import render_single
    import lqtpy
    kw = dict(atlases=["dk"], write=False, plots=True, tracts=False, disconnection=False)
    plain = lqtpy.analyze_single(_flawed_img(), **kw)
    assert "flagged" not in plain["viewer"]
    assert "viewer-flag-btn" not in render_single(plain)
    r = lqtpy.analyze_single(_flawed_img(), flagged_voxels=True, **kw)
    assert r["viewer"]["flagged_kinds"] == ["remove"] and r["viewer"]["flagged"] != r["viewer"]["overlay"]
    html = render_single(r)
    assert 'id="viewer-flag-btn" aria-pressed="false"' in html and "viewer-ov2" in html


def test_flagged_viewer_without_findings_adds_no_switch():
    pytest.importorskip("nilearn")
    import lqtpy
    r = lqtpy.analyze_single(S.img(S.sphere([-35, 5, 15], 12)), atlases=["dk"], write=False,
                             plots=True, tracts=False, disconnection=False, flagged_voxels=True)
    assert "flagged" not in r["viewer"]


# --------------------------------------------------------------------------- #
# --per-label
# --------------------------------------------------------------------------- #
def _multi(tmp_path, dtype=np.int16):
    m = S.sphere([-35, 5, 15], 10).astype(dtype) * 3 + S.sphere([35, -20, 10], 8).astype(dtype) * 7
    p = tmp_path / "multi.nii.gz"
    nib.save(nib.Nifti1Image(m, S.MNI_AFFINE), str(p))
    return p


def test_split_labels(tmp_path):
    from lqtpy.labels import label_values, split_labels
    p = _multi(tmp_path)
    assert label_values(str(p)) == [3, 7]
    parts = split_labels(str(p))
    assert set(np.unique(np.asarray(parts[7].dataobj))) == {0, 1}


def test_fractional_mask_is_refused(tmp_path):
    from lqtpy.errors import InvalidLesionError
    from lqtpy.labels import label_values
    p = tmp_path / "prob.nii.gz"
    nib.save(nib.Nifti1Image(S.sphere([-35, 5, 15], 8) * 0.7, S.MNI_AFFINE), str(p))
    with pytest.raises(InvalidLesionError):
        label_values(str(p))


def test_per_label_cli_writes_one_report_per_label(tmp_path):
    from lqtpy import cli
    p = _multi(tmp_path)
    out = tmp_path / "o"
    assert cli.main(["single", str(p), "-o", str(out), "--per-label", "-q", "-a", "dk",
                     "--no-tracts", "--no-disconnection", "--no-viewer"]) == 0
    rows = (out / "multi_labels.csv").read_text().splitlines()
    assert rows[0] == "Label,Name,Voxels,VolumeMM3,QC,Report" and len(rows) == 3
    assert (out / "multi_label3" / "multi_label3_report.html").exists()
    assert "/" in rows[1].split(",")[-1]


def test_too_many_labels_are_refused(tmp_path):
    from lqtpy import cli
    from lqtpy.errors import InvalidLesionError
    from lqtpy.labels import analyze_labels
    m = np.zeros(S.MNI_SHAPE, np.int16)
    for k in range(25):                                   # a parcellation-like mask
        m[20 + 5 * k:23 + 5 * k, 100:103, 80:83] = k + 1
    p = tmp_path / "parc.nii.gz"
    nib.save(nib.Nifti1Image(m, S.MNI_AFFINE), str(p))
    with pytest.raises(InvalidLesionError, match="25 label values"):
        analyze_labels(str(p), str(tmp_path / "o"), write=False)
    assert cli.main(["single", str(p), "--per-label", "-q"]) == 2
    assert not (tmp_path / "o").exists()


def test_without_per_label_labels_are_merged(tmp_path):
    import lqtpy
    p = _multi(tmp_path)
    n3, n7 = (int(S.sphere(c, r).sum()) for c, r in (([-35, 5, 15], 10), ([35, -20, 10], 8)))
    assert lqtpy.prepare(str(p)).n_voxels == n3 + n7


# --------------------------------------------------------------------------- #
# --bids
# --------------------------------------------------------------------------- #
def _bids(root):
    for sub, ses, c in (("01", "1", [-35, 5, 15]), ("01", "2", [-30, 0, 15]), ("02", None, [35, -20, 10])):
        d = root / f"sub-{sub}" / (f"ses-{ses}" if ses else "") / "anat"
        d.mkdir(parents=True, exist_ok=True)
        pre = f"sub-{sub}" + (f"_ses-{ses}" if ses else "")
        nib.save(S.img(S.sphere(c, 9)), str(d / f"{pre}_space-MNI152NLin6Asym_label-lesion_roi.nii.gz"))
        nib.save(S.img(S.sphere(c, 30)), str(d / f"{pre}_T1w.nii.gz"))          # not a mask
    (root / "participants.tsv").write_text("participant_id\tage\nsub-01\t61\nsub-02\tn/a\n")


def test_find_masks_names_and_metadata(tmp_path):
    from lqtpy import bids
    _bids(tmp_path)
    masks = bids.find_masks(str(tmp_path))
    assert [n for n, _, _ in masks] == ["sub-01_ses-1", "sub-01_ses-2", "sub-02"]
    meta = bids.participants(str(tmp_path), masks)
    assert meta["sub-01_ses-2"]["age"] == "61" and meta["sub-02"]["age"] == ""


def test_bids_derivative_layout(tmp_path):
    import json
    from lqtpy import cli
    _bids(tmp_path)
    assert cli.main(["group", str(tmp_path), "--bids", "-q", "-a", "dk", "--no-tracts",
                     "--no-disconnection", "--no-viewer"]) == 0
    out = tmp_path / "derivatives" / "lqtpy"
    d = json.loads((out / "dataset_description.json").read_text())
    assert d["DatasetType"] == "derivative" and d["GeneratedBy"][0]["Name"] == "lqtpy"
    f = out / "sub-01" / "ses-2" / "anat" / "sub-01_ses-2_space-MNI152NLin6Asym_desc-dkdamage_stats.tsv"
    assert f.exists() and "\t" in f.read_text().splitlines()[0]
    assert (out / "sub-02" / "anat" / "sub-02_space-MNI152NLin6Asym_desc-morphometry_stats.json").exists()
    assert "subjects/" in (out / ".bidsignore").read_text()
    # a second run doesn't pick up its own output as input
    from lqtpy import bids
    assert len(bids.find_masks(str(tmp_path), exclude=str(out))) == 3


def test_bids_refuses_anonymise(tmp_path):
    from lqtpy import cli
    _bids(tmp_path)
    assert cli.main(["group", str(tmp_path), "--bids", "--anonymise", "-q"]) == 2


# --------------------------------------------------------------------------- #
# QC example pictures (ReportStyle(qc_examples=True) / --qc-examples)
# --------------------------------------------------------------------------- #
def test_every_example_draws_something():
    from lqtpy.qc_examples import CAPTIONS, example_svg
    for fid in CAPTIONS:
        svg = example_svg(fid)
        assert svg.startswith("<svg") and svg.count("<rect") > 20, fid
    assert example_svg("not_mni") is None


def test_examples_only_when_asked():
    from lqtpy.report import render_single
    from lqtpy.report.style import ReportStyle
    import lqtpy
    r = lqtpy.analyze_single(_flawed_img(), atlases=["dk"], write=False, plots=True, tracts=False,
                             disconnection=False, viewer=False)
    assert "What does this look like" not in render_single(r)
    html = render_single(r, ReportStyle(qc_examples=True))
    assert html.count("What does this look like") == 1 and 'aria-label="Example: specks"' in html
