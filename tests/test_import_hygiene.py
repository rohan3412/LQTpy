"""``import lqtpy`` must not drag in heavy optional dependencies or mutate
global interpreter state (warnings filters, matplotlib backend).

Every assertion here runs in a **fresh subprocess interpreter**
(``subprocess.run([sys.executable, "-c", ...])``) rather than in-process,
because pytest itself (or another test file collected earlier) may already
have imported matplotlib or otherwise changed warnings state -- that would
make an in-process check pass or fail for the wrong reason.
"""
from __future__ import annotations

import subprocess
import sys

import pytest


def _run(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True, text=True, timeout=60,
    )


def test_import_lqtpy_does_not_import_matplotlib():
    proc = _run(
        "import sys\n"
        "import lqtpy\n"
        "assert 'matplotlib' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_does_not_import_nilearn():
    proc = _run(
        "import sys\n"
        "import lqtpy\n"
        "assert 'nilearn' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_does_not_change_warnings_filters():
    """lqtpy's own code must not touch ``warnings.filters`` (it used to, via
    a blanket ``warnings.filterwarnings("ignore")`` in ``plotting.py``).

    The baseline is captured *after* pre-importing lqtpy's declared core
    dependencies (numpy, nibabel, scipy.ndimage) rather than before anything
    is imported: numpy and scipy mutate ``warnings.filters`` themselves as a
    documented side effect of their own import machinery (e.g. registering an
    ``ignore`` filter for a "matrix subclass" warning, and an ``always``
    filter for ``scipy.special.SpecialFunctionWarning``), independent of and
    before lqtpy's own code ever runs. lqtpy legitimately needs to import
    those packages, so a before/after around the bare `import lqtpy`
    statement would fail here for reasons that have nothing to do with
    lqtpy's own behaviour. Pre-importing the same dependency closure lqtpy
    itself pulls in isolates what lqtpy's *own* code does to warnings state.
    """
    proc = _run(
        "import warnings, copy\n"
        "import numpy, nibabel\n"
        "from scipy import ndimage\n"
        "before = copy.deepcopy(warnings.filters)\n"
        "import lqtpy\n"
        "after = warnings.filters\n"
        "assert before == after, (before, after)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_does_not_force_matplotlib_backend():
    """If matplotlib is already imported *before* lqtpy, importing lqtpy must
    not call ``matplotlib.use(...)`` and change the active backend."""
    pytest.importorskip("matplotlib")
    proc = _run(
        "import matplotlib\n"
        "before = matplotlib.get_backend()\n"
        "import lqtpy\n"
        "after = matplotlib.get_backend()\n"
        "assert before == after, (before, after)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_does_not_import_pandas():
    proc = _run(
        "import sys\n"
        "import lqtpy\n"
        "assert 'pandas' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------- #
# lqtpy.report.fragments: pure HTML-fragment builders meant for a host
# application to embed directly. Must stay free of matplotlib/nilearn/pandas
# too -- importing it (which also runs lqtpy/report/__init__.py, since a
# submodule import always executes its parent package first) must not pull
# in report-page assembly's own heavier, lazily-imported dependencies.
# --------------------------------------------------------------------------- #
def test_import_lqtpy_report_fragments_does_not_import_matplotlib():
    proc = _run(
        "import sys\n"
        "import lqtpy.report.fragments\n"
        "assert 'matplotlib' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_report_fragments_does_not_import_nilearn():
    proc = _run(
        "import sys\n"
        "import lqtpy.report.fragments\n"
        "assert 'nilearn' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout


def test_import_lqtpy_report_fragments_does_not_import_pandas():
    proc = _run(
        "import sys\n"
        "import lqtpy.report.fragments\n"
        "assert 'pandas' not in sys.modules, sorted(sys.modules)\n"
        "print('OK')\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "OK" in proc.stdout
