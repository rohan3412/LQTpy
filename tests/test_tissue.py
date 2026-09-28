"""Tissue composition and location classes (lqtpy.tissue) and their use in
descriptions, results and reports."""
from __future__ import annotations

import numpy as np
import pytest

import lqtpy
from lqtpy import tissue
from lqtpy.atlases import load_atlas

import synthetic as S


# Synthetic lesions placed on template anatomy (MNI mm).
PLACES = {
    "periventricular": ((-22, -10, 26), 4),   # beside the body of the left lateral ventricle
    "deep white matter": ((-26, -10, 38), 3),  # centrum semiovale
    "cortical": ((-45, -10, 50), 4),           # precentral gyrus
    "deep grey matter": ((-12, -18, 6), 4),    # thalamus (borders the ventricle)
    "infratentorial": ((0, -28, -32), 4),      # pons
}


@pytest.mark.parametrize("expected", list(PLACES))
def test_part_classes(expected):
    centre, r = PLACES[expected]
    res = tissue.locate(S.img(S.sphere(centre, r)))
    assert len(res["parts"]) == 1
    assert res["parts"][0]["class"] == expected
    assert res["counts"][expected] == 1


def test_thalamus_is_not_called_periventricular():
    part = tissue.locate(S.img(S.sphere((-12, -18, 6), 4)))["parts"][0]
    assert part["dist_ventricle_mm"] > 0
    assert part["class"] == "deep grey matter"


def test_composition_sums_to_100_and_summary_class():
    comp = tissue.composition(S.img(S.sphere((-40, 5, 10), 22)))
    assert set(comp) == {"outside", *tissue.TISSUES}
    assert sum(comp.values()) == pytest.approx(100, abs=0.5)
    res = tissue.locate(S.img(S.sphere((-40, 5, 10), 22)))
    assert res["class"] == "cortico-subcortical"
    assert tissue.locate(S.img(S.sphere((0, -28, -32), 4)))["class"] == "infratentorial"
    assert tissue.locate(S.img(S.sphere((-24, 0, 12), 5)))["class"] == "subcortical"


def test_outside_the_head_is_outside():
    m = np.zeros(S.MNI_SHAPE, bool)
    m[2:6, 2:6, 2:6] = True  # image corner
    res = tissue.locate(S.img(m))
    assert res["composition"]["outside"] == 100.0
    assert res["class"] is None
    assert res["parts"][0]["class"] == "outside the brain"
    assert res["counts"]["outside the brain"] == 1


def test_empty_lesion_is_an_input_error():
    with pytest.raises(lqtpy.InvalidLesionError):
        tissue.locate(S.img(np.zeros(S.MNI_SHAPE, bool)))


def test_stroke_parts_use_tissue_class_small_vessel_uses_location():
    m = S.img(S.blobs([((-40, 5, 10), 12), ((-22, -10, 26), 4)]))
    parts = dict(lqtpy.describe(m).sections)["Parts"]
    assert "it is cortico-subcortical" in parts or "it is cortical" in parts
    assert "periventricular" not in parts
    res = dict(lqtpy.describe(m, profile="research").sections)["Parts"]
    assert "it is periventricular" in res


def test_parts_follow_gap_aware_clusters():
    """Part ids match the description's (qc.clusters) numbering."""
    m = S.blobs([((-40, 5, 10), 12), ((-22, -10, 26), 4)])
    res = tissue.locate(S.img(m))
    assert [p["id"] for p in res["parts"]] == [1, 2]
    assert res["parts"][0]["voxels"] > res["parts"][1]["voxels"]
    assert res["parts"][1]["class"] == "periventricular"


def test_distance_maps_match_the_atlas():
    to_ctx, to_vent, aff = tissue.distance_maps()
    lab = load_atlas("tissue").label_array()
    assert to_ctx.shape == lab.shape and not to_ctx.flags.writeable
    assert np.all(to_ctx[np.isin(lab, (1, 2))] == 0)
    assert np.all(to_vent[np.isin(lab, (7, 8))] == 0)


def test_tissue_atlas_schema():
    at = load_atlas("tissue")
    assert {m.Tissue for m in at.labels.values()} >= {"Cortex", "White matter", "Deep grey matter",
                                                     "CSF", "Cerebellum", "Brainstem"}
    assert at.labels[1].Hemisphere == "L" and at.labels[2].Hemisphere == "R"
    counts = at.region_voxels()
    # Plausible adult volumes on the template (cm³).
    wm = (counts[3] + counts[4]) / 1000
    assert 400 < wm < 600
    assert "tissue" not in lqtpy.atlases.DEFAULT_ATLASES


def test_descriptions_use_tissue_words():
    d = lqtpy.describe(S.img(S.sphere((-40, 5, 10), 22)))
    assert "By tissue it is cortico-subcortical" in d.text
    assert dict(d.sections)["Tissue"]
    lac = lqtpy.describe(S.img(S.sphere((-24, 0, 12), 5)))
    assert "small subcortical lesion" in lac.text and "lacunar-sized" in lac.text
    none = lqtpy.describe(S.img(S.sphere((-40, 5, 10), 22)), tissue=False)
    assert "Tissue" not in dict(none.sections)


def test_small_vessel_counts_by_location():
    m = S.blobs([((-22, -10, 26), 4), ((22, -8, 26), 3), ((-26, -10, 38), 3),
                 ((-45, -10, 50), 3), ((0, -28, -32), 3)])
    d = lqtpy.describe(S.img(m), profile="small_vessel")
    assert "By location:" in d.text
    assert "2 periventricular" in d.text and "1 infratentorial" in d.text
    assert "**2 periventricular**" in dict(d.marked)["Summary"]


def test_unlabelled_share_is_broken_down():
    d = lqtpy.describe(S.img(S.sphere((-26, -10, 38), 4)))
    loc = dict(d.sections)["Location"]
    assert "unnamed white matter" in loc
    assert "no bundled atlas names" not in loc


def test_analyze_single_carries_tissue(tmp_path):
    res = lqtpy.analyze_single(S.img(S.sphere((-40, 5, 10), 12)), out_dir=str(tmp_path),
                               plots=False, verbose=False)
    assert res["tissue"]["class"] == "cortico-subcortical"
    assert any(m["id"] == "tissue" and m["status"] == "ran" for m in res["manifest"]["modules"])


def test_group_tissue_columns(tmp_path):
    import csv
    subs = {"a": S.img(S.sphere((-40, 5, 10), 12)), "b": S.img(S.sphere((0, -28, -32), 4))}
    d = tmp_path / "in"
    d.mkdir()
    import nibabel as nib
    for k, im in subs.items():
        nib.save(im, str(d / f"{k}.nii.gz"))
    res = lqtpy.analyze_group(str(d), out_dir=str(tmp_path / "out"), plots=False,
                              verbose=False, subject_reports=False)
    assert res["tissue_classes"] == {"cortico-subcortical": 1, "infratentorial": 1}
    with open(tmp_path / "out" / "group_subject_summary.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["TissueClass"] for r in rows} == {"cortico-subcortical", "infratentorial"}
    assert "NPeriventricular" in rows[0]


def test_report_shows_tissue():
    from lqtpy.report import render_single
    res = lqtpy.analyze_single(S.img(S.blobs([((-40, 5, 10), 12), ((-22, -10, 26), 4)])),
                               write=False, plots=False, verbose=False)
    html = render_single(res)
    assert ">Tissue<" in html
    assert "Separate parts by location" in html and "periventricular" in html
