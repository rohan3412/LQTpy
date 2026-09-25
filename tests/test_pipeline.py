"""analyze_single/analyze_group end to end: manifest, extras, disconnection,
per-subject reports, and the CLI."""
from __future__ import annotations

import os

import numpy as np
import nibabel as nib
import pytest

import lqtpy
from lqtpy import _discon_index as di
from lqtpy.cli import main

import synthetic as S


def _status(res):
    return {m["id"]: m["status"] for m in res["manifest"]["modules"]}


@pytest.fixture
def mca():
    return S.img(S.sphere([-40, 10, 5], 12))


def test_core_only_manifest(mca, monkeypatch, tmp_path):
    monkeypatch.setenv("LQTPY_DATA_HOME", str(tmp_path))
    lqtpy.cache_clear()
    res = lqtpy.analyze_single(mca, write=False)
    st = _status(res)
    assert res["manifest"]["scope"] == "core"
    assert st["morphometry"] == st["qc"] == st["description"] == "ran"
    assert st["disconnection"] == "unavailable"
    assert st["measure:shape"] == "skipped"
    assert res["description"]["text"].startswith("A single")
    assert all(a["coverage"]["LesionVoxelsOnGrid"] > 0 for a in res["atlases"])


def test_extras_manifest(mca):
    res = lqtpy.analyze_single(mca, write=False, extras="all")
    assert res["manifest"]["scope"] == "core + extras"
    assert _status(res)["measure:shape"] == "ran"
    assert res["measures"]["shape"]["descriptors"]
    assert "Shape" in dict(res["description"]["sections"])


def test_unknown_extra_rejected(mca):
    with pytest.raises(ValueError):
        lqtpy.analyze_single(mca, write=False, extras=["nope"])


def test_disconnection_runs_automatically_when_index_installed(mca, monkeypatch, tmp_path):
    lines = [np.linspace([-60, 10, 5], [-20, 10, 5], 40) + np.random.default_rng(i).normal(0, 1, 3)
             for i in range(50)]
    idx = di.build_index([("AF_L", lines)], di.TRACT_SHAPE, di.TRACT_AFFINE)
    di.save_index(str(tmp_path / "discon_index_v1.npz"), idx)
    monkeypatch.setenv("LQTPY_DATA_HOME", str(tmp_path))
    lqtpy.cache_clear()
    try:
        res = lqtpy.analyze_single(mca, write=False)
        assert _status(res)["disconnection"] == "ran"
        assert res["disconnection"][0]["percent_disconnected"] > 50
        assert _status(lqtpy.analyze_single(mca, write=False, disconnection=False))["disconnection"] == "skipped"
    finally:
        lqtpy.cache_clear()


def test_single_writes_report_with_scope_and_description(mca, tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    pytest.importorskip("pandas")
    res = lqtpy.analyze_single(mca, out_dir=str(tmp_path), plots=True, extras="all")
    page = open(res["report_path"], encoding="utf-8").read()
    for text in ("Core + extras", "Summary", "Extras", "Streamline disconnection"):
        assert text in page
    assert os.path.exists(tmp_path / "lesion_description.txt")


def test_group_subject_reports_and_links(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    pytest.importorskip("pandas")
    pairs = [("a", S.img(S.sphere([-40, 10, 5], 8))), ("b", S.img(S.sphere([25, -85, 5], 8)))]
    res = lqtpy.analyze_group(pairs, atlases=["aal"], out_dir=str(tmp_path), plots=True,
                              tracts=False)
    group = open(res["report_path"], encoding="utf-8").read()
    for s in res["subjects"]:
        rel = s["report_path"].replace("\\", "/")
        assert f'href="{rel}"' in group
        sub = open(tmp_path / s["report_path"], encoding="utf-8").read()
        assert "group_report.html" in sub and "Group report</a>" in sub
    assert os.path.exists(tmp_path / "group_aal_lobes.csv")
    assert _status(res)["subject_reports"] == "ran"


def test_group_duplicate_names_get_suffixes():
    pairs = [("x", S.img(S.sphere([-40, 10, 5], 5))), ("x", S.img(S.sphere([40, 10, 5], 5)))]
    res = lqtpy.analyze_group(pairs, atlases=["aal"], write=False, tracts=False, describe=False)
    assert [s["name"] for s in res["subjects"]] == ["x", "x_2"]


# ------------------------------------------------------------------- CLI
@pytest.fixture
def gapped_file(tmp_path):
    path = tmp_path / "gapped.nii.gz"
    nib.save(S.img(S.native_drawing([-35, 5, 15], (14, 18, 16), draw_every=2)), str(path))
    return str(path)


def test_cli_version(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0 and "lqtpy" in capsys.readouterr().out


def test_cli_qc_and_repair(gapped_file, tmp_path, capsys):
    assert main(["qc", gapped_file]) == 0
    assert "some axial slices only" in capsys.readouterr().out
    out = str(tmp_path / "fixed.nii.gz")
    assert main(["repair", gapped_file, "-o", out]) == 0
    assert os.path.exists(out)
    assert main(["qc", out]) == 0
    assert "No quality problems" in capsys.readouterr().out.splitlines()[-1]


def test_cli_describe(gapped_file, capsys):
    assert main(["describe", gapped_file, "--no-shape"]) == 0
    assert capsys.readouterr().out.startswith("A single")


def test_cli_errors_are_one_line(tmp_path, capsys):
    assert main(["describe", str(tmp_path / "missing.nii.gz")]) == 2
    err = capsys.readouterr().err
    assert err.startswith("lqtpy:") and "Traceback" not in err
