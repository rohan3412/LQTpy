"""Save an LQTpy HTML report as PDF (the clinical view, A4).

The reports print the clinical document through their own print styles, so
any Chromium-family browser can turn one into a PDF headlessly. No Python
dependency is needed: :func:`to_pdf` finds Microsoft Edge, Google Chrome or
Chromium (or the executable in the ``LQTPY_BROWSER`` environment variable)
and runs it with ``--print-to-pdf``. If none is found and WeasyPrint is
installed, it is used instead; otherwise a
:class:`~lqtpy.errors.MissingDependencyError` explains the options.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from ..errors import MissingDependencyError

__all__ = ["to_pdf", "find_browser"]

_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
]
_NAMES = ["msedge", "microsoft-edge", "google-chrome", "google-chrome-stable", "chromium",
          "chromium-browser", "chrome"]


def find_browser() -> "str | None":
    """Path to a Chromium-family browser, or None."""
    env = os.environ.get("LQTPY_BROWSER")
    if env and Path(env).exists():
        return env
    for c in _CANDIDATES:
        if Path(c).exists():
            return c
    for n in _NAMES:
        p = shutil.which(n)
        if p:
            return p
    return None


def to_pdf(source, pdf_path=None, *, browser: "str | None" = None, timeout: float = 120.0) -> str:
    """Save a report as PDF and return the PDF's path.

    ``source`` is the path of an HTML report written by ``analyze_single``/
    ``analyze_group`` (``result["report_path"]``), an HTML string, or a
    result dict with plots (it is rendered first). ``pdf_path`` defaults to
    the HTML path with a ``.pdf`` suffix (or a temp file for a string/dict).
    The PDF is always the clinical report, on white, A4.
    """
    tmp_dir = None
    if isinstance(source, dict):
        from . import render_group, render_single
        source = render_group(source) if "n_subjects" in source else render_single(source)
    if isinstance(source, (str, os.PathLike)) and Path(source).suffix.lower() in (".html", ".htm") \
            and Path(source).exists():
        html_path = Path(source).resolve()
    else:
        tmp_dir = tempfile.mkdtemp(prefix="lqtpy-pdf-")
        html_path = Path(tmp_dir) / "report.html"
        html_path.write_text(str(source), encoding="utf-8")
    pdf = Path(pdf_path) if pdf_path else (
        html_path.with_suffix(".pdf") if tmp_dir is None else Path(tempfile.gettempdir()) / "lqtpy_report.pdf")
    pdf = pdf.resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)

    exe = browser or find_browser()
    try:
        if exe:
            with tempfile.TemporaryDirectory(prefix="lqtpy-browser-") as profile:
                cmd = [exe, "--headless=new", "--disable-gpu", "--no-first-run",
                       "--no-default-browser-check", f"--user-data-dir={profile}",
                       "--no-pdf-header-footer", "--print-to-pdf-no-header",
                       "--virtual-time-budget=10000", f"--print-to-pdf={pdf}",
                       html_path.as_uri()]
                if pdf.exists():
                    pdf.unlink()
                subprocess.run(cmd, check=False, timeout=timeout,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # Some launchers (Edge on Windows) return before the browser
                # process has written the file: wait for it, and for its size
                # to settle, before giving up.
                deadline = time.monotonic() + timeout
                last = -1
                while time.monotonic() < deadline:
                    size = pdf.stat().st_size if pdf.exists() else -1
                    if size > 0 and size == last:
                        break
                    last = size
                    time.sleep(0.5)
            if pdf.exists() and pdf.stat().st_size > 0:
                return str(pdf)
        try:
            import weasyprint  # type: ignore
        except ImportError:
            weasyprint = None
        if weasyprint is not None:
            weasyprint.HTML(filename=str(html_path)).write_pdf(str(pdf))
            return str(pdf)
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    raise MissingDependencyError(
        "pdf",
        "Saving a PDF needs Microsoft Edge, Google Chrome or Chromium (set LQTPY_BROWSER to its "
        "path if it isn't found), or the weasyprint package. You can also open the HTML report "
        "and use its Export PDF button.")
