"""Formats, native space, config, folder QC, web front end, watch mode,
plugin template, the SUIT download, figures and the CLI."""
import csv
import json
import os
import threading
import urllib.request

import numpy as np
import nibabel as nib
import pytest

import lqtpy
import synthetic as syn


# ------------------------------------------------------------------ formats
def test_nrrd_and_mgz_round_trip(tmp_path):
    from lqtpy.formats import read_nrrd, write_nrrd
    img = syn.img(syn.sphere((-40, 10, 10), 8))
    ref = lqtpy.lesion_stats(img)
    for enc in ("gzip", "raw"):
        p = write_nrrd(str(tmp_path / f"a_{enc}.nrrd"), img, encoding=enc)
        assert lqtpy.lesion_stats(p)["center_of_mass_mm"] == ref["center_of_mass_mm"]
    np.testing.assert_allclose(read_nrrd(p).affine, img.affine)
    mgz = tmp_path / "a.mgz"
    nib.save(nib.MGHImage(np.asarray(img.dataobj), img.affine), str(mgz))
    assert lqtpy.lesion_stats(str(mgz))["volume_mm3"] == ref["volume_mm3"]
    from lqtpy.analysis import _normalise_inputs
    assert {n for n, _ in _normalise_inputs(str(tmp_path))} == {"a_gzip", "a_raw", "a"}


def test_dicom_seg_needs_highdicom(tmp_path):
    pytest.importorskip("pydicom", reason="only checks the error when highdicom is missing")


def test_dicom_seg_error_message(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name == "highdicom":
            raise ImportError
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    from lqtpy.formats import read_dicom_seg
    with pytest.raises(lqtpy.MissingDependencyError, match="highdicom"):
        read_dicom_seg(str(tmp_path / "x.dcm"))


# ------------------------------------------------------------------ native space
def _native_drawing():
    aff = np.diag([1.0, 1.0, 5.0, 1.0])
    aff[:3, 3] = [-100, -120, -80]
    d = np.zeros((200, 240, 40), np.uint8)
    for k in range(15, 22, 2):   # drawn on every other thick slice
        d[50:70, 120:140, k] = 1
    return d, aff


def test_native_warp_and_drawing_checks(tmp_path):
    from lqtpy.native import native_qc, warp_to_mni
    d, aff = _native_drawing()
    T = np.eye(4)
    T[0, 3] = 5
    mni = warp_to_mni((d, aff), T)
    assert lqtpy.lesion_stats(mni)["center_of_mass_mm"][0] == pytest.approx(-35.5, abs=1)
    ids = [f.id for f in native_qc((d, aff))]
    assert ids == ["sparse_slices"]
    np.savetxt(tmp_path / "T.txt", T)
    res = lqtpy.analyze_single((d, aff), write=False, transform=str(tmp_path / "T.txt"))
    msgs = [f["message"] for f in res["qc"]]
    assert any(m.startswith("In the native-space drawing") for m in msgs)


def test_deformation_field(tmp_path):
    from lqtpy.native import warp_to_mni
    d, aff = _native_drawing()
    shape = (91, 109, 91)
    raff = np.diag([2.0, 2.0, 2.0, 1.0])
    raff[:3, 3] = [-90, -126, -72]
    ijk = np.indices(shape).reshape(3, -1).T
    world = ijk @ raff[:3, :3].T + raff[:3, 3]
    field = (world - [5, 0, 0]).reshape(*shape, 3).astype(np.float32)  # MNI -> native = x - 5
    out = warp_to_mni((d, aff), nib.Nifti1Image(field, raff))
    assert lqtpy.lesion_stats(out)["center_of_mass_mm"][0] == pytest.approx(-35.5, abs=1.5)


# ------------------------------------------------------------------ config
def test_config_file(tmp_path, monkeypatch):
    from lqtpy.config import load_config
    p = tmp_path / "lqtpy.toml"
    p.write_text('[defaults]\natlases = ["aal", "dk"]  # two\nprofile = "research"\njobs = 2\npdf = false\n')
    cfg, used = load_config(str(p))
    assert cfg == {"atlases": ["aal", "dk"], "profile": "research", "jobs": 2, "pdf": False}
    p.write_text("[defaults]\nnope = 1\n")
    with pytest.raises(ValueError, match="unknown setting"):
        load_config(str(p))
    from lqtpy.config import _mini_toml
    assert _mini_toml('[defaults]\nx = "a#b" # c\n') == {"defaults": {"x": "a#b"}}


def test_cli_uses_config(tmp_path, monkeypatch, capsys):
    from lqtpy.cli import main
    les = tmp_path / "les.nii.gz"
    nib.save(syn.img(syn.sphere((-40, 10, 10), 6)), str(les))
    (tmp_path / "lqtpy.toml").write_text(f'[defaults]\nprofile = "research"\nout_dir = "{(tmp_path / "o").as_posix()}"\n')
    monkeypatch.chdir(tmp_path)
    assert main(["single", str(les), "-q", "--atlas", "aal", "--no-viewer"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 1 and out[0].endswith("les_report.html")     # path printed once
    assert os.sep in out[0] and "/" not in out[0].replace(os.sep, "") or os.sep == "/"
    rec = json.loads(open(out[0], encoding="utf-8").read().split('id="lqtpy-run">')[1].split("</script>")[0])
    assert rec["settings"]["profile"] == "research"
    assert main(["config"]) == 0
    assert "research" in capsys.readouterr().out


def test_cli_atlas_choices_hide_tissue():
    from lqtpy.cli import build_parser
    sp = build_parser()._subparsers._group_actions[0].choices["single"]
    act = next(a for a in sp._actions if a.dest == "atlas")
    assert "tissue" not in act.choices and "aal" in act.choices
    ext = next(a for a in sp._actions if a.dest == "extras")
    assert ext.default == "none"


# ------------------------------------------------------------------ folder QC
def test_folder_qc_and_clean(tmp_path):
    from lqtpy.qc import check_folder, clean_folder
    from lqtpy.report.qcpage import render_qc_folder
    m = syn.sphere((-40, 10, 10), 8)
    m[0, 100, 90] = True   # a fragment on the image edge
    nib.save(syn.img(m), str(tmp_path / "edge.nii.gz"))
    nib.save(syn.img(syn.sphere((30, 0, 10), 7)), str(tmp_path / "fine.nii.gz"))
    (tmp_path / "broken.nii.gz").write_bytes(b"not a nifti")
    rows = {r["name"]: r for r in check_folder(str(tmp_path))}
    assert rows["broken"]["error"] and rows["fine"]["worst"] in ("ok", "info")
    assert any(f["id"] == "edge_artifact" for f in rows["edge"]["flags"])
    with pytest.raises(ValueError):
        clean_folder(str(tmp_path), str(tmp_path))
    done = {os.path.basename(c["file"]): c for c in clean_folder(str(tmp_path), str(tmp_path / "clean"))}
    assert done["edge.nii.gz"]["status"] == "written"
    assert done["fine.nii.gz"]["status"] == "nothing to remove"
    assert clean_folder(str(tmp_path), str(tmp_path / "clean"))[1]["status"] == "exists"
    page = render_qc_folder(list(rows.values()), str(tmp_path))
    assert "Mask quality" in page and "edge" in page


# ------------------------------------------------------------------ web
def test_web_upload_returns_report(tmp_path):
    from lqtpy.web import make_server
    srv = make_server("127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        port = srv.server_address[1]
        assert b"Upload a lesion mask" in urllib.request.urlopen(f"http://127.0.0.1:{port}/").read()
        p = tmp_path / "Smith_Jane.nii.gz"
        nib.save(syn.img(syn.sphere((-40, 10, 10), 6)), str(p))
        boundary = "----lqtpytest"
        body = (f"--{boundary}\r\nContent-Disposition: form-data; name=\"profile\"\r\n\r\nstroke\r\n"
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"lesion\"; filename=\"{p.name}\"\r\n"
                "Content-Type: application/octet-stream\r\n\r\n").encode() + p.read_bytes() + \
               f"\r\n--{boundary}--\r\n".encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/analyse", data=body, method="POST",
                                     headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
        page = urllib.request.urlopen(req, timeout=120).read().decode("utf-8")
        assert "Lesion quantification report" in page
        assert "Smith_Jane" not in page          # uploads are anonymised
        bad = urllib.request.Request(f"http://127.0.0.1:{port}/analyse", data=b"x", method="POST",
                                     headers={"Content-Type": "text/plain"})
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(bad)
    finally:
        srv.shutdown()
        srv.server_close()


# ------------------------------------------------------------------ watch
def test_watch_rebuilds_on_new_mask(tmp_path):
    from lqtpy.watch import watch
    src = tmp_path / "in"
    src.mkdir()
    nib.save(syn.img(syn.sphere((-40, 10, 10), 6)), str(src / "a.nii.gz"))
    runs = []
    watch(str(src), str(tmp_path / "out"), interval=0.01, max_cycles=3, plots=False,
          on_run=lambda r: runs.append(r["n_subjects"]))
    assert runs == [1]


# ------------------------------------------------------------------ plugin template
def test_plugin_template(tmp_path):
    import importlib
    import sys
    from lqtpy.registry import get_measure, register_measure
    from lqtpy.scaffold import new_plugin
    root = new_plugin("roughness", str(tmp_path))
    assert os.path.exists(os.path.join(root, "pyproject.toml"))
    with pytest.raises(FileExistsError):
        new_plugin("roughness", str(tmp_path))
    sys.path.insert(0, root)
    try:
        mod = importlib.import_module("lqtpy_roughness")
        register_measure(mod.MEASURES[0], overwrite=True)
        r = lqtpy.analyze_single(syn.img(syn.sphere((-40, 10, 10), 8)), write=False,
                                 extras=["roughness"])
        assert r["measures"]["roughness"]["descriptors"]
    finally:
        sys.path.remove(root)
        from lqtpy import registry
        registry._MEASURES.pop("roughness", None)


# ------------------------------------------------------------------ SUIT (local "download")
def test_suit_fetch_from_local_source(tmp_path, monkeypatch):
    from lqtpy import datasets
    from lqtpy.atlases import cache_clear, list_atlases, load_atlas
    src = tmp_path / "src"
    src.mkdir()
    m = np.zeros(syn.MNI_SHAPE, np.uint8)
    m[syn.sphere((-20, -60, -30), 10)] = 1
    m[syn.sphere((20, -60, -30), 10)] = 2
    nib.save(nib.Nifti1Image(m, syn.MNI_AFFINE), str(src / "atl-Anatom_space-MNI_dseg.nii"))
    (src / "atl-Anatom.tsv").write_text("index\tname\tcolor\n1\tLeft_CrusI\t#fff\n2\tRight_CrusI\t#fff\n")
    monkeypatch.setenv("LQTPY_DATA_HOME", str(tmp_path / "home"))
    from lqtpy import atlases
    atlases._CUSTOM.pop("suit", None)
    assert "suit" not in list_atlases()
    with pytest.raises(lqtpy.DataUnavailableError, match="fetch suit"):
        load_atlas("suit")
    datasets.fetch_suit(src.as_uri() + "/")
    try:
        assert "suit" in list_atlases()
        rows = lqtpy.atlas_overlap(syn.img(syn.sphere((-20, -60, -30), 5)), "suit")
        assert rows[0]["RegionName"] == "Left_CrusI"
        assert os.path.exists(tmp_path / "home" / "suit" / "SHA256SUMS")
    finally:
        atlases._CUSTOM.pop("suit", None)
        cache_clear()


# ------------------------------------------------------------------ figures
def test_figures_are_fast_and_themeable():
    import time
    from lqtpy import figures
    les = syn.img(syn.sphere((-40, 10, 10), 8))
    figures.glass_brain_svg(les)
    t = time.perf_counter()
    g = figures.glass_brain_svg(les)
    s = figures.slices_svg(les)
    assert time.perf_counter() - t < 2.0
    assert g.startswith("<svg") and 'role="img"' in g and "data:image/png;base64," in s
    png = figures.png_bytes(np.zeros((4, 5, 4), np.uint8))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    fm = figures.flat_map_svg("dk", {"precentral_left": 40.0},
                              labels={"precentral_left": ("left precentral", "dk|left_precentral")})
    assert 'data-hl="dk|left_precentral"' in fm and "--fl:" in fm and "--fd:" in fm
    v = figures.viewer_data(les)
    assert v["params"]["nbSlice"]["X"] > 0 and v["bg"].startswith("data:image/png")


def test_report_features_present(tmp_path):
    res = lqtpy.analyze_single(syn.img(syn.sphere((-45, 15, 10), 9)), out_dir=str(tmp_path),
                               plots=True)
    page = open(res["report_path"], encoding="utf-8").read()
    assert 'data-copy="desc-plain"' in page                  # copy description
    assert 'class="gl"' in page                               # glossary popovers
    assert 'data-dl="dl' in page                              # downloadable tables
    assert "lqtpy rerun" in page and 'id="lqtpy-run"' in page  # reproducibility
    assert 'data-hl="dk|' in page                             # table <-> map linking
    assert "lqtpy.sort." in page                              # remembered sort
    if res.get("viewer"):
        assert 'id="viewer-canvas"' in page                   # slice viewer
    assert os.path.exists(tmp_path / "data_dictionary.csv")
    with open(tmp_path / "data_dictionary.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert all(r["Meaning"] for r in rows), [r["Column"] for r in rows if not r["Meaning"]]
