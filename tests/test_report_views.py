"""The standalone report: dashboard + clinical views, theme toggle, PDF."""
from __future__ import annotations

import os

import pytest

import lqtpy

import synthetic as S


@pytest.fixture(scope="module")
def report(tmp_path_factory):
    pytest.importorskip("matplotlib")
    pytest.importorskip("nilearn")
    pytest.importorskip("pandas")
    out = tmp_path_factory.mktemp("rep")
    res = lqtpy.analyze_single(S.img(S.sphere([-40, 10, 5], 12)), out_dir=str(out), plots=True,
                               extras="all", tracts=False)
    return res, open(res["report_path"], encoding="utf-8").read()


def test_both_views_and_controls(report):
    _, page = report
    assert 'data-theme="dark"' in page                      # dark by default
    assert 'class="view view-dash"' in page and 'class="view view-clin"' in page
    for control in ('data-set-view="dashboard"', 'data-set-view="clinical"', 'id="theme-btn"',
                    'id="pdf-btn"'):
        assert control in page
    assert "@media print" in page                           # PDF prints the clinical view


def test_key_findings_emphasised_not_raw(report):
    _, page = report
    assert "<strong>" in page
    assert "**" not in page.split("<script>")[0]


def test_core_and_extras_marked(report):
    _, page = report
    assert "Core + extras" in page
    assert 'class="card extras"' in page
    assert "Extra measures" in page                         # clinical section


def test_images_embedded_once(report):
    res, page = report
    assert page.count(res["glass_uri"]) == 1                 # clinical view borrows by id
    assert page.count(res["glass_uri_light"]) == 1
    assert 'data-src-from="img-glass"' in page
    # The clinical view / PDF borrow the white-background version.
    assert f'id="img-glass" src="{res["glass_uri_light"][:60]}' in page


def test_sortable_tables_and_light_plates(report):
    _, page = report
    assert 'th.sortable' in page and "aria-sort" in page
    assert 'class="plate light"' in page


def test_to_pdf(report, tmp_path):
    from lqtpy.report.pdf import find_browser, to_pdf
    if find_browser() is None:
        pytest.skip("no Chromium-family browser available")
    res, _ = report
    pdf = to_pdf(res["report_path"], tmp_path / "r.pdf")
    with open(pdf, "rb") as fh:
        assert fh.read(5) == b"%PDF-"
    assert os.path.getsize(pdf) > 10_000
