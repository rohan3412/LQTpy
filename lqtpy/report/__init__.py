"""Self-contained HTML report assembly.

``lqtpy.report`` is a package: :mod:`lqtpy.report.style` holds the
:class:`~lqtpy.report.style.ReportStyle` knobs, :mod:`lqtpy.report.fragments`
holds pure, dependency-light HTML-fragment builders meant for a host
application to embed directly, and this module's own
:func:`render_single`/:func:`render_group` assemble LQTpy's standalone
report (used by ``analyze_single``/``analyze_group`` when ``plots=True``)
out of those same fragments plus the plot images/tables that predate them.

Nothing at this package's import time touches matplotlib/nilearn/pandas --
``lqtpy.analysis`` only imports ``render_single``/``render_group`` lazily,
inside the ``plots=True`` branch, exactly as it imported the old
``lqtpy/report.py`` module before this package split.
"""
from __future__ import annotations

from . import fragments, style
from ._standalone import render_group, render_single
from .pdf import to_pdf
from .style import DARK_STYLE, DEFAULT_STYLE, ReportStyle

__all__ = [
    "render_single", "render_group", "to_pdf",
    "ReportStyle", "DEFAULT_STYLE", "DARK_STYLE",
    "fragments", "style",
]
