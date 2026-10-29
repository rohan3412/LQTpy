"""Matplotlib/nilearn plots, kept for callers that use them directly.

Reports no longer use this module: their figures are drawn with numpy in
:mod:`lqtpy.figures`. :func:`glass_brain_png` returns a base64 ``data:`` URI
(or None on failure)."""
from __future__ import annotations

import base64
import io
import warnings


def use_agg():
    """Force matplotlib's headless ``Agg`` backend.

    Opt-in only: a library must never force a backend on its caller's behalf,
    but a headless program (the CLI) legitimately wants this before any
    rendering happens. Call it once, early, e.g. from ``cli.py``.
    """
    import matplotlib
    matplotlib.use("Agg")


def _fig_to_uri(fig, dpi: int = 120) -> str:
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _safe(fn):
    """Run a plot builder, returning None (and a warning) on any failure."""
    import matplotlib.pyplot as plt

    try:
        return fn()
    except Exception as exc:  # plotting must never break the report
        warnings.warn(f"plot failed: {exc}")
        plt.close("all")
        return None


#: The report's lesion colour (the same as :data:`lqtpy.figures.LESION_RGB`).
LESION_HEX = "#f5793a"


def _lesion_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([LESION_HEX, LESION_HEX])


def glass_brain_png(lesion_img, title=None, theme: str = "dark") -> str | None:
    from nilearn import plotting

    def build():
        disp = plotting.plot_glass_brain(
            lesion_img, title=title, cmap=_lesion_cmap(), colorbar=False,
            display_mode="lyrz", alpha=0.95, threshold=0.5, black_bg=theme == "dark",
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)
