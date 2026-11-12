"""0.8.0 batch 1: cache --prune, doctor, demo, --export-problems, --xlsx,
notebook cards, install-shortcut. All opt-in; none changes an existing output."""
from __future__ import annotations

import os
import sys

import nibabel as nib
import numpy as np
import pytest

import synthetic as S
from lqtpy import cli


def _run(*argv):
    return cli.main(list(argv))


# --------------------------------------------------------------------------- #
# cache --prune
# --------------------------------------------------------------------------- #
def test_prune_removes_only_other_version_folders(tmp_path, monkeypatch, capsys):
    from lqtpy import _diskcache
    from lqtpy._version import __version__
    monkeypatch.setenv("LQTPY_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("LQTPY_NO_DISK_CACHE", raising=False)
    for name in ("0.1.0", "0.6.1", __version__):
        (tmp_path / name).mkdir()
        (tmp_path / name / "a.npy").write_bytes(b"x" * 2048)
    (tmp_path / "notes").mkdir()                       # not a version: left alone
    assert {os.path.basename(p) for p, _ in _diskcache.old_versions()} == {"0.1.0", "0.6.1"}
    assert _run("cache") == 0
    assert "lqtpy cache --prune" in capsys.readouterr().out
    assert _run("cache", "--prune") == 0
    assert "Freed" in capsys.readouterr().out
    left = {p.name for p in tmp_path.iterdir()}
    assert left == {__version__, "notes"}
    assert _run("cache", "--prune") == 0
    assert "No caches of older" in capsys.readouterr().out


def test_prune_does_nothing_when_cache_is_off(monkeypatch, capsys):
    monkeypatch.setenv("LQTPY_NO_DISK_CACHE", "1")
    assert _run("cache", "--prune") == 0
    assert "off" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #
def test_doctor_reports_and_flags_bad_config(tmp_path, monkeypatch, capsys):
    from lqtpy.doctor import run_checks
    monkeypatch.chdir(tmp_path)
    checks = run_checks()
    assert all(s in ("ok", "note", "problem") for s, _, _ in checks)
    assert any("numpy" in w for _, w, _ in checks)
    (tmp_path / "lqtpy.toml").write_text("[defaults]\nnot_a_setting = 1\n")
    bad = [c for c in run_checks() if c[0] == "problem"]
    assert bad and "not_a_setting" in bad[0][1] and bad[0][2]
    capsys.readouterr()
    code = cli.main(["--config", str(tmp_path / "lqtpy.toml"), "doctor"])
    assert code == 1 and "problem" in capsys.readouterr().out


def test_every_note_says_how_to_fix_it(monkeypatch, tmp_path):
    from lqtpy.doctor import run_checks
    monkeypatch.chdir(tmp_path)
    for status, what, fix in run_checks():
        if status != "ok":
            assert fix, what


# --------------------------------------------------------------------------- #
# demo
# --------------------------------------------------------------------------- #
def test_demo_lesion_is_a_clean_mni_mask():
    from lqtpy import qc
    from lqtpy.demo import demo_lesion
    img = demo_lesion()
    n = int(np.asarray(img.dataobj).sum())
    assert 20_000 < n < 80_000
    assert qc.assess(img).verdict == "usable"


def test_demo_writes_a_report(tmp_path, capsys):
    assert _run("demo", "--no-open", "-o", str(tmp_path / "d")) == 0
    out = capsys.readouterr().out
    assert os.path.exists(tmp_path / "d" / "demo_lesion_report.html") and "made-up" in out


# --------------------------------------------------------------------------- #
# qc --export-problems
# --------------------------------------------------------------------------- #
def _flawed(tmp_path):
    m = S.sphere([-35, 5, 15], 12) | S.blobs([([30, -40, 30], 1), ([10, 40, 0], 1)])
    p = tmp_path / "flawed.nii.gz"
    nib.save(S.img(m), str(p))
    return p


def test_export_problems_writes_one_overlay_per_finding(tmp_path, capsys):
    p = _flawed(tmp_path)
    out = tmp_path / "ov"
    assert _run("qc", str(p), "--export-problems", str(out)) == 0
    files = sorted(os.listdir(out))
    assert files == ["flawed_specks.nii.gz"]
    ov = nib.load(str(out / files[0]))
    src = nib.load(str(p))
    assert ov.shape == src.shape and np.allclose(ov.affine, src.affine)
    data = np.asarray(ov.dataobj)
    assert set(np.unique(data)) == {0, 1} and data.sum() >= 2
    assert (np.asarray(src.dataobj)[data == 1] > 0).all()      # marks lesion voxels only


def test_export_problems_never_overwrites(tmp_path):
    from lqtpy.qc_tools import export_problems
    p = _flawed(tmp_path)
    export_problems(str(p), str(tmp_path / "ov"))
    with pytest.raises(FileExistsError):
        export_problems(str(p), str(tmp_path / "ov"))


def test_export_problems_clean_mask_says_so(tmp_path, capsys):
    p = tmp_path / "clean.nii.gz"
    nib.save(S.img(S.sphere([-35, 5, 15], 12)), str(p))
    assert _run("qc", str(p), "--export-problems", str(tmp_path / "ov")) == 0
    assert "nothing to export" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# --xlsx
# --------------------------------------------------------------------------- #
def test_workbook_has_a_sheet_per_table_and_the_dictionary(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    from lqtpy.xlsx import write_workbook
    (tmp_path / "s_dk_damage.csv").write_text("RegionName,LesionVoxels,PercentDamage\nInsula,120,4.5\n")
    (tmp_path / "s_tracts.csv").write_text("Tract,PercentOverlap\nAF_L,12.25\n")
    (tmp_path / "data_dictionary.csv").write_text("File,Column,Unit,Meaning\ns_tracts.csv,Tract,,x\n")
    (tmp_path / "anonymisation_key.csv").write_text("id,name\nA1,secret\n")
    path = write_workbook(str(tmp_path))
    wb = openpyxl.load_workbook(path)
    assert wb.sheetnames == ["dk_damage", "tracts", "data_dictionary"]      # key never included
    ws = wb["dk_damage"]
    assert [c.value for c in ws[1]] == ["RegionName", "LesionVoxels", "PercentDamage"]
    assert [c.value for c in ws[2]] == ["Insula", 120, 4.5]                  # numbers stay numbers


def test_sheet_names_are_legal_and_unique():
    from lqtpy.xlsx import _sheet_names
    names = _sheet_names(["a_" + "x" * 50 + "1.csv", "a_" + "x" * 50 + "2.csv", "a_b:c.csv"], "a_")
    assert len(set(n.lower() for n in names.values())) == 3
    assert all(len(n) <= 31 and not set(n) & set("[]:*?/\\") for n in names.values())


def test_xlsx_flag_writes_the_workbook(tmp_path):
    pytest.importorskip("openpyxl")
    p = tmp_path / "l.nii.gz"
    nib.save(S.img(S.sphere([-35, 5, 15], 10)), str(p))
    out = tmp_path / "o"
    assert _run("single", str(p), "-o", str(out), "-q", "--no-viewer", "--no-tracts",
                "--no-disconnection", "-a", "dk", "--xlsx") == 0
    assert (out / "lqtpy_tables.xlsx").exists()


# --------------------------------------------------------------------------- #
# notebook cards
# --------------------------------------------------------------------------- #
def test_cards_render_escaped_html(tmp_path):
    import lqtpy
    from lqtpy import notebook, qc
    p = lqtpy.prepare(S.img(S.sphere([-35, 5, 15], 12) | S.blobs([([30, -40, 30], 1)])), name="a<b>")
    h = notebook.card_html(p)
    assert "a&lt;b&gt;" in h and "cc" in h and "<b>" not in h.replace("<b>", "", 0).split("a&lt;")[0]
    r = notebook.card_html(qc.assess(p))
    assert "Mask quality" in r and "score" in r
    d = notebook.card_html(lqtpy.describe(p))
    assert d.startswith("<div") and "Description" in d
    assert notebook.card_html(42) is None


def test_enable_is_off_without_ipython_session():
    from lqtpy import notebook
    if "IPython" in sys.modules and __import__("IPython").get_ipython() is not None:
        pytest.skip("inside IPython")
    assert notebook.enable() is False
    notebook.disable()


# --------------------------------------------------------------------------- #
# install-shortcut
# --------------------------------------------------------------------------- #
def test_shortcut_installs_and_removes_in_sendto(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path))
    from lqtpy import shortcut
    assert _run("install-shortcut") == 0
    path = shortcut.sendto_path()
    with open(path, encoding="mbcs", errors="replace", newline="") as fh:
        text = fh.read()
    assert sys.executable in text and "-m lqtpy single" in text and "lqtpy_reports" in text
    assert "\r\n" in text
    assert _run("install-shortcut", "--remove") == 0
    assert not os.path.exists(path)
    assert _run("install-shortcut", "--remove") == 0
    assert "No Send to entry" in capsys.readouterr().out


def test_shortcut_refuses_off_windows(monkeypatch, capsys):
    monkeypatch.setattr(sys, "platform", "linux")
    assert _run("install-shortcut") == 2
    assert "Windows" in capsys.readouterr().err
