"""``lqtpy.api``: the numeric web-handler surface.

Parity tests assert every ``api.*`` function returns exactly what the
existing orchestrator-level functions (``compute_morphometry``,
``region_overlap``, ``network_rollup``, ``tract_overlap``, and
``analyze_single`` itself) already compute for the same lesion, for every
synthetic fixture. A subprocess test asserts the normal call path never
imports matplotlib/nilearn/pandas. Atlas-registration tests build a tiny
synthetic labelled atlas in code and exercise it through the same api
functions.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import nibabel as nib
import pytest

import lqtpy
from lqtpy import api
from lqtpy.atlases import (RegionMeta, atlas_from_arrays, cache_clear,
                           list_atlases, load_atlas, register_atlas)
from lqtpy import atlases as _atlases_mod
from lqtpy.errors import UnknownAtlasError
from lqtpy.io import load_lesion
from lqtpy.morphometry import compute_morphometry
from lqtpy.overlap import lobe_rollup, network_rollup, region_overlap
from lqtpy.tracts import tract_overlap, tracts_available

_FIXTURES = ["sphere_lesion_img", "offgrid_lesion_img", "oblique_lesion_img",
             "two_blob_lesion_img", "probabilistic_lesion_img"]


# --------------------------------------------------------------------------- #
# Parity: api.* vs. the lower-level functions analyze_single/group wrap.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_lesion_stats_matches_compute_morphometry(fixture_name, request):
    img = request.getfixturevalue(fixture_name)
    expected_img, _ = load_lesion(img, threshold=0.5)
    expected = compute_morphometry(expected_img)
    assert api.lesion_stats(img) == expected


@pytest.mark.parametrize("atlas_key", ["aal", "schaefer", "dk"])
@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_atlas_overlap_matches_region_overlap(fixture_name, atlas_key, request):
    img = request.getfixturevalue(fixture_name)
    lesion_img, _ = load_lesion(img, threshold=0.5)
    atlas = load_atlas(atlas_key)
    expected = region_overlap(lesion_img, atlas)
    assert api.atlas_overlap(img, atlas_key) == expected


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_network_rollup_for_matches_network_rollup(fixture_name, request):
    img = request.getfixturevalue(fixture_name)
    lesion_img, _ = load_lesion(img, threshold=0.5)
    atlas = load_atlas("schaefer")
    expected = network_rollup(region_overlap(lesion_img, atlas), atlas)
    assert api.network_rollup_for(img, "schaefer") == expected


@pytest.mark.parametrize("atlas_key", ["aal", "dk"])
@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_lobe_overlap_matches_lobe_rollup(fixture_name, atlas_key, request):
    img = request.getfixturevalue(fixture_name)
    lesion_img, _ = load_lesion(img, threshold=0.5)
    atlas = load_atlas(atlas_key)
    expected = lobe_rollup(region_overlap(lesion_img, atlas), atlas)
    assert api.lobe_overlap(img, atlas_key) == expected


def test_lobe_overlap_unmapped_atlas_raises_typed_error(sphere_lesion_img):
    from lqtpy.errors import NoLobeMappingError
    with pytest.raises(NoLobeMappingError):
        api.lobe_overlap(sphere_lesion_img, "schaefer")
    # also backward compatible with a plain `except ValueError`.
    with pytest.raises(ValueError):
        api.lobe_overlap(sphere_lesion_img, "schaefer")


@pytest.mark.parametrize("fixture_name", _FIXTURES)
def test_tract_proxy_overlap_matches_tract_overlap(fixture_name, request):
    if not tracts_available():
        pytest.skip("tract index not bundled in this checkout")
    img = request.getfixturevalue(fixture_name)
    lesion_img, _ = load_lesion(img, threshold=0.5)
    expected = tract_overlap(lesion_img)
    assert api.tract_proxy_overlap(img) == expected


def test_atlas_overlap_matches_analyze_single(write_nifti, sphere_lesion_img):
    path = write_nifti(sphere_lesion_img)
    result = lqtpy.analyze_single(path, atlases=["aal"], write=False,
                                  plots=False, verbose=False)
    expected = result["atlases"][0]["df"]
    assert api.atlas_overlap(path, "aal") == expected


def test_lesion_stats_matches_analyze_single(write_nifti, sphere_lesion_img):
    path = write_nifti(sphere_lesion_img)
    result = lqtpy.analyze_single(path, atlases=["aal"], write=False,
                                  plots=False, verbose=False)
    assert api.lesion_stats(path) == result["morphometry"]


def test_tract_disconnection_raises_data_unavailable_without_index(
        sphere_lesion_img, monkeypatch, tmp_path):
    # Real per-tract disconnection (see tests/test_disconnection.py for
    # full coverage on a synthetic index) requires a locally-built index;
    # with none built/pointed to, it must fail with the typed,
    # actionable DataUnavailableError, not the old NotImplementedError.
    from lqtpy import datasets
    from lqtpy.errors import DataUnavailableError
    monkeypatch.setenv("LQTPY_DATA_HOME", str(tmp_path / "empty_data_home"))
    datasets.discon_index_cache_clear()
    with pytest.raises(DataUnavailableError, match="build_discon_index.py"):
        api.tract_disconnection(sphere_lesion_img)
    datasets.discon_index_cache_clear()


def test_atlas_overlap_unknown_atlas_raises_typed_error(sphere_lesion_img):
    with pytest.raises(UnknownAtlasError):
        api.atlas_overlap(sphere_lesion_img, "not-a-real-atlas")
    # also backward compatible with a plain `except KeyError`.
    with pytest.raises(KeyError):
        api.atlas_overlap(sphere_lesion_img, "not-a-real-atlas")


# --------------------------------------------------------------------------- #
# as_lesion_image acceptance: path / nibabel image / (array, affine) tuple.
# --------------------------------------------------------------------------- #
def test_api_accepts_array_affine_tuple(sphere_lesion_img):
    data = np.asarray(sphere_lesion_img.get_fdata())
    affine = sphere_lesion_img.affine
    assert api.lesion_stats((data, affine)) == api.lesion_stats(sphere_lesion_img)


def test_api_accepts_path(write_nifti, sphere_lesion_img):
    path = write_nifti(sphere_lesion_img)
    assert api.lesion_stats(path) == api.lesion_stats(sphere_lesion_img)


# --------------------------------------------------------------------------- #
# as_lesion_image validation: wrong shape/affine/dtype -> typed errors.
# --------------------------------------------------------------------------- #
def test_as_lesion_image_rejects_2d():
    from lqtpy.errors import InvalidLesionError
    from lqtpy.io import as_lesion_image
    with pytest.raises(InvalidLesionError):
        as_lesion_image((np.zeros((4, 4)), np.eye(4)))


def test_as_lesion_image_rejects_4d_non_singleton():
    from lqtpy.errors import InvalidLesionError
    from lqtpy.io import as_lesion_image
    with pytest.raises(InvalidLesionError):
        as_lesion_image((np.zeros((4, 4, 4, 3), dtype=np.uint8), np.eye(4)))


def test_as_lesion_image_squeezes_4d_singleton():
    from lqtpy.io import as_lesion_image
    img = as_lesion_image((np.zeros((4, 4, 4, 1), dtype=np.uint8), np.eye(4)))
    assert img.shape == (4, 4, 4)


def test_as_lesion_image_rejects_singular_affine():
    from lqtpy.errors import GeometryError
    from lqtpy.io import as_lesion_image
    affine = np.eye(4)
    affine[0, 0] = 0.0  # singular: zero row in the linear part
    with pytest.raises(GeometryError):
        as_lesion_image((np.zeros((4, 4, 4), dtype=np.uint8), affine))


def test_as_lesion_image_rejects_non_finite_affine():
    from lqtpy.errors import GeometryError
    from lqtpy.io import as_lesion_image
    affine = np.eye(4)
    affine[0, 3] = float("nan")
    with pytest.raises(GeometryError):
        as_lesion_image((np.zeros((4, 4, 4), dtype=np.uint8), affine))


def test_as_lesion_image_rejects_non_numeric_dtype():
    from lqtpy.errors import InvalidLesionError
    from lqtpy.io import as_lesion_image
    with pytest.raises(InvalidLesionError):
        as_lesion_image((np.zeros((4, 4, 4), dtype=object), np.eye(4)))


def test_as_lesion_image_rejects_missing_file(tmp_path):
    from lqtpy.io import as_lesion_image
    with pytest.raises(FileNotFoundError):
        as_lesion_image(str(tmp_path / "does_not_exist.nii.gz"))


# --------------------------------------------------------------------------- #
# Atlas registration: register_atlas / atlas_from_arrays.
# --------------------------------------------------------------------------- #
@pytest.fixture
def tiny_atlas_img():
    shape = (64, 64, 64)
    affine = np.diag([1.0, 1.0, 1.0, 1.0])
    affine[:3, 3] = -32.0
    data = np.zeros(shape, dtype=np.int16)
    data[10:20, 10:20, 10:20] = 1
    data[40:50, 40:50, 40:50] = 2
    return nib.Nifti1Image(data, affine)


@pytest.fixture
def _unregister():
    """Remove any custom atlas keys this test registers, on teardown."""
    keys = []
    yield keys
    for k in keys:
        _atlases_mod._CUSTOM.pop(k, None)
    cache_clear()


def test_register_atlas_and_use_through_load_atlas(tiny_atlas_img, sphere_lesion_img,
                                                    _unregister):
    register_atlas("tiny", "Tiny Test Atlas", tiny_atlas_img,
                   {1: "RegionA", 2: RegionMeta(RegionName="RegionB", Group="G")})
    _unregister.append("tiny")

    assert "tiny" in list_atlases()
    atlas = load_atlas("tiny")
    assert atlas.key == "tiny"
    assert atlas.labels[1].RegionName == "RegionA"
    assert atlas.labels[2] == RegionMeta(RegionName="RegionB", Group="G")

    # Works through the numeric API too, exactly like a bundled atlas.
    records = api.atlas_overlap(sphere_lesion_img, "tiny")
    assert isinstance(records, list)
    expected = region_overlap(sphere_lesion_img, atlas)
    assert records == expected


def test_register_atlas_overwrite_protection(tiny_atlas_img, _unregister):
    register_atlas("tiny2", "Tiny 2", tiny_atlas_img, {1: "A"})
    _unregister.append("tiny2")

    with pytest.raises(ValueError):
        register_atlas("tiny2", "Tiny 2 again", tiny_atlas_img, {1: "A"})

    register_atlas("tiny2", "Tiny 2 v2", tiny_atlas_img, {1: "A2"}, overwrite=True)
    assert load_atlas("tiny2").labels[1].RegionName == "A2"


def test_atlas_from_arrays(tiny_atlas_img, _unregister):
    data = np.asarray(tiny_atlas_img.dataobj)
    affine = tiny_atlas_img.affine
    atlas_from_arrays("tiny3", "Tiny 3", data, affine, {1: "A", 2: "B"})
    _unregister.append("tiny3")

    atlas = load_atlas("tiny3")
    assert atlas.labels[1].RegionName == "A"
    assert atlas.labels[2].RegionName == "B"


def test_register_atlas_rejects_bad_label_value(tiny_atlas_img, _unregister):
    _unregister.append("tiny4")
    with pytest.raises(TypeError):
        register_atlas("tiny4", "Tiny 4", tiny_atlas_img, {1: 12345})


# --------------------------------------------------------------------------- #
# import lqtpy.api + calling every implemented function must never import
# matplotlib/nilearn/pandas.
# --------------------------------------------------------------------------- #
def test_api_module_never_imports_heavy_deps():
    code = (
        "import sys\n"
        "import numpy as np\n"
        "import nibabel as nib\n"
        "import lqtpy.api as api\n"
        "shape = (64, 64, 64)\n"
        "affine = np.eye(4)\n"
        "affine[:3, 3] = -32.0\n"
        "zz, yy, xx = np.ogrid[:64, :64, :64]\n"
        "mask = (zz - 32) ** 2 + (yy - 32) ** 2 + (xx - 32) ** 2 <= 10 ** 2\n"
        "img = nib.Nifti1Image(mask.astype(np.uint8), affine)\n"
        "api.lesion_stats(img)\n"
        "api.atlas_overlap(img, 'aal')\n"
        "api.network_rollup_for(img, 'schaefer')\n"
        "api.lobe_overlap(img, 'aal')\n"
        "api.tract_proxy_overlap(img)\n"
        "import os\n"
        "from lqtpy.errors import DataUnavailableError\n"
        "os.environ['LQTPY_DATA_HOME'] = os.path.join(os.getcwd(), 'no-such-dir')\n"
        "try:\n"
        "    api.tract_disconnection(img)\n"
        "except DataUnavailableError:\n"
        "    pass\n"
        "bad = sorted(m for m in ('matplotlib', 'nilearn', 'pandas') if m in sys.modules)\n"
        "assert bad == [], bad\n"
        "print('OK')\n"
    )
    proc = subprocess.run([sys.executable, "-c", code],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
