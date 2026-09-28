"""Natural-language descriptions: facts are right, text says what they say."""
from __future__ import annotations

import pytest

from lqtpy.describe import PROFILES, describe, facts

import synthetic as S


@pytest.fixture(scope="module")
def mca():
    return S.img(S.sphere([-40, 10, 5], 15))


def test_hierarchy_facts_add_up(mca):
    f = facts(mca)
    total = sum(h["pct_of_lesion"] for h in f["hemispheres"])
    assert total == pytest.approx(100, abs=1.0)
    left = f["hemispheres"][0]
    assert left["hemisphere"] == "left"
    lobe_sum = (sum(lb["pct_of_lesion"] for lb in left["lobes"]) + left["wm_pct"]
                + left["unlabelled_pct"])
    assert lobe_sum == pytest.approx(left["pct_of_lesion"], abs=1.0)


def test_stroke_text(mca):
    d = describe(mca, shape=True)
    t = d.text
    assert t.startswith("A single 14.1 cm³ lesion in the left hemisphere.")
    assert "left frontal lobe" in t and "insula" in t
    assert "Broca's area" in t
    assert "not predictions of symptoms" in t
    assert "compact and roughly round" in t
    assert [h for h, _ in d.sections][:2] == ["Summary", "Location"]


def test_two_parts_described_separately():
    d = describe(S.img(S.sphere([-40, 10, 5], 12) | S.sphere([25, -85, 5], 9)))
    assert "2 separate parts" in d.text
    parts = dict(d.sections)["Parts"]
    assert "Part 1" in parts and "Part 2" in parts and "right occipital lobe" in parts


def test_small_vessel_summarises():
    m = S.blobs([([(-1) ** i * (20 + (i % 3) * 4), -20 + i * 4, 20 + (i % 4) * 3], 2 + (i % 3))
                 for i in range(12)])
    d = describe(S.img(m), profile="small_vessel")
    assert d.text.startswith("12 separate lesions (6 left, 6 right)")
    assert "Part 4" not in d.text


def test_gap_drawing_is_one_lesion_and_caveated():
    m = S.native_drawing([-35, 5, 15], (14, 18, 16), draw_every=2)
    d = describe(S.img(m), shape=True)
    assert d.text.startswith("A single")
    assert "Shape isn't described" in d.text
    assert "some axial slices only" in d.text


def test_tumour_midline():
    d = describe(S.img(S.ellipsoid([12, 30, 20], (22, 20, 18))), profile="tumour")
    assert "crossing the midline" in d.text


@pytest.mark.parametrize("profile", sorted(PROFILES))
def test_every_profile_is_deterministic(mca, profile):
    assert describe(mca, profile=profile).text == describe(mca, profile=profile).text


def test_unknown_profile():
    with pytest.raises(ValueError):
        describe(S.img(S.sphere([0, 0, 0], 3)), profile="poetry")


def test_white_matter_and_territory_for_capsular_lacune():
    d = describe(S.img(S.sphere([-24, -12, 12], 6)))
    loc = dict(d.sections)["Location"]
    assert "posterior limb of the internal capsule" in loc
    terr = dict(d.sections)["Vascular territory"]
    assert terr.startswith("It lies in the left middle cerebral artery (MCA) territory (86%): "
                           "almost entirely its lateral lenticulostriate (deep perforator) "
                           "territory (86%).")
    marked = dict(d.marked)["Vascular territory"]
    assert "**the left middle cerebral artery (MCA) territory (86%)**" in marked
    assert "**" not in d.text
    assert "<strong>" in d.html()


def test_territory_for_occipital_lesion():
    d = describe(S.img(S.sphere([20, -85, 5], 12)))
    assert "right posterior cerebral artery (PCA) territory (97%): almost entirely its occipital part" in dict(d.sections)["Vascular territory"]
