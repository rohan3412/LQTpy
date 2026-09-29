"""Visual regression of the reports (opt-in: LQTPY_VISUAL=1).

Screenshots sample reports in three views and compares them with
tests/visual/baseline/. See tools/visual_check.py; accept an intended
design change with ``python tools/visual_check.py --update``.
"""
import os
import sys

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("LQTPY_VISUAL") != "1",
                                reason="visual regression runs with LQTPY_VISUAL=1")


def test_reports_look_the_same():
    pytest.importorskip("PIL")
    from lqtpy.report.pdf import find_browser
    if find_browser() is None:
        pytest.skip("no Chromium-family browser")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
    import visual_check
    assert visual_check.run(update=False) == 0
