"""Plot generation. Every function returns a base64 ``data:`` URI (or None on
failure) so reports stay self-contained single files."""
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


#: The report's lesion hue (see ggseg_flat.THEMES): one colour for "lesion"
#: in every brain figure, on a black plate that works in light and dark themes.
LESION_HEX = "#f5793a"


def _lesion_cmap():
    from matplotlib.colors import ListedColormap
    return ListedColormap([LESION_HEX, LESION_HEX])


def glass_brain_png(lesion_img, title=None) -> str | None:
    from nilearn import plotting

    def build():
        disp = plotting.plot_glass_brain(
            lesion_img, title=title, cmap=_lesion_cmap(), colorbar=False,
            display_mode="lyrz", alpha=0.95, threshold=0.5, black_bg=True,
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)


def slices_png(lesion_img, cut_coords, title=None) -> str | None:
    from nilearn import plotting

    def build():
        disp = plotting.plot_roi(
            lesion_img, title=title, cut_coords=cut_coords,
            display_mode="ortho", cmap=_lesion_cmap(), alpha=0.75, colorbar=False,
            black_bg=True,
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)


def _hbar_png(labels, values, xlabel, title, color, annot=None, xmax=None):
    """Shared horizontal-bar builder (labels/values already top-N, any order)."""
    if labels is None or len(labels) == 0:
        return None

    def build():
        import matplotlib.pyplot as plt

        lab = list(labels)[::-1]
        val = list(values)[::-1]
        ann = list(annot)[::-1] if annot is not None else None
        fig, ax = plt.subplots(figsize=(7, max(2.5, 0.36 * len(lab))))
        ax.barh(lab, val, color=color)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        top = xmax if xmax is not None else max(1.0, max(val)) * 1.12
        ax.set_xlim(0, top)
        for y, v in enumerate(val):
            txt = ann[y] if ann is not None else f"{v:.0f}%"
            ax.text(v + top * 0.01, y, txt, va="center", fontsize=8)
        fig.tight_layout()
        return _fig_to_uri(fig)
    return _safe(build)


def damage_bar_png(df, top=15, title="Top damaged regions") -> str | None:
    if df is None or df.empty:
        return None
    d = df.head(top)
    xmax = max(100.0, float(d["PercentDamage"].max()) * 1.05)
    return _hbar_png(d["RegionName"], d["PercentDamage"],
                     "Percent of region damaged (%)", title, "#c1443c", xmax=xmax)


def network_bar_png(df, title="Network involvement (Yeo-7)") -> str | None:
    if df is None or df.empty:
        return None
    return _hbar_png(df["Network"], df["PercentDamage"],
                     "Percent of network damaged (%)", title, "#2f6f8a")


def tract_bar_png(df, top=15, title="White-matter tract overlap") -> str | None:
    if df is None or df.empty:
        return None
    d = df.head(top)
    return _hbar_png(d["Tract"], d["PercentOverlap"],
                     "Percent of tract territory lesioned (%)", title, "#5b8c3e")


# --------------------------------------------------------------------------- #
# ggseg-style flat-brain rendering for the Python-ggseg `aseg` atlas (used for
# AAL's subcortical/cerebellar surrogate map; DK and Schaefer use the 4-view
# renderer in ggseg_flat.py instead).
# --------------------------------------------------------------------------- #
_GGSEG_OUTLINES = {
    "aseg": ["Coronal", "Sagittal"],
}


def _ggseg_dir(atlas_key: str) -> str:
    import os
    import ggseg
    return os.path.join(os.path.dirname(ggseg.__file__), "data", atlas_key)


def render_ggseg(data: dict, atlas_key: str, cmap=None, vmax=None,
                 title="", ylabel="Percent damaged (%)", lw=1.6,
                 edgecolor=None, na_color=None, figsize=(9, 6),
                 fontsize=13, theme: str = "light") -> str | None:
    """Draw a ggseg flat-brain map with the classic thick-black-border style.

    ``data`` maps ggseg region keys -> values. Labelled regions are coloured by
    ``cmap``; every other atlas region is filled ``na_color``. Works for any
    ggseg atlas directory (``dk``, ``aseg``, ...).
    """
    if not data:
        return None
    from .ggseg_flat import THEMES, lesion_cmap
    t = THEMES[theme]
    cmap = cmap or lesion_cmap(theme)
    edgecolor = edgecolor or t["edge"]
    na_color = na_color or t["na"]

    def build():
        import os
        from glob import glob
        import ggseg
        import matplotlib.pyplot as plt
        from matplotlib.path import Path
        import matplotlib.patches as patches

        wd = _ggseg_dir(atlas_key)
        outlines = _GGSEG_OUTLINES.get(atlas_key, [])
        plt.close("all")

        outline_txt = [open(os.path.join(wd, e)).read() for e in outlines]
        ax = ggseg._create_figure_(outline_txt, figsize, t["bg"], title,
                                   fontsize, t["fg"])

        def patch(text, facecolor):
            codes, verts = ggseg._svg_parse_(text)
            ax.add_patch(patches.PathPatch(
                Path(verts, codes), facecolor=facecolor,
                edgecolor=edgecolor, lw=lw))

        all_files = glob(os.path.join(wd, "*"))
        all_regions = [os.path.splitext(os.path.basename(e))[0] for e in all_files]
        # Fill non-data regions (excluding the outline shells) with na_color.
        na = set(all_regions).difference(data).difference(outlines)
        for r in na:
            patch(open(os.path.join(wd, r)).read(), na_color)

        vmx = vmax if vmax is not None else max(1.0, max(data.values()))
        import matplotlib.colors as mcolors
        cm = cmap if not isinstance(cmap, str) else plt.get_cmap(cmap)
        norm = mcolors.Normalize(vmin=0, vmax=vmx)
        for key, val in data.items():
            fp = os.path.join(wd, key)
            if os.path.isfile(fp):
                patch(open(fp).read(), cm(norm(val)))

        ggseg._add_colorbar_(ax, cm, norm, t["fg"], fontsize * 0.7, ylabel)
        fig = ax.figure
        fig.set_facecolor(t["bg"])
        return _fig_to_uri(fig)
    return _safe(build)


def ggseg_aseg_png(damage_by_aseg: dict, title="", theme: str = "light") -> str | None:
    """Render an aseg subcortical/cerebellar map coloured by percent damaged."""
    return render_ggseg(damage_by_aseg, "aseg", title=title, theme=theme)


def group_overlap_png(freq_img, n_subjects, title=None) -> str | None:
    from nilearn import plotting

    def build():
        from .ggseg_flat import lesion_cmap
        disp = plotting.plot_stat_map(
            freq_img, title=title, cmap=lesion_cmap("dark"), colorbar=True, threshold=0.5,
            vmax=max(1, int(n_subjects)), display_mode="z", cut_coords=6,
            black_bg=True,
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)


def group_damage_bar_png(df, top=15, title="Most frequently damaged regions") -> str | None:
    """Group bar chart: mean percent damage with subject-count annotation."""
    if df is None or df.empty:
        return None
    d = df.head(top)
    annot = [f"n={n}/{t}" for n, t in zip(d["NSubjects"], d["NSubjectsTotal"])]
    return _hbar_png(d["RegionName"], d["CohortMeanPercentDamage"],
                     "Mean percent damaged, all subjects (%)",
                     title, "#8a2f8f", annot=annot)


def group_network_bar_png(df, title="Network involvement (group)") -> str | None:
    if df is None or df.empty:
        return None
    annot = [f"n={n}/{t}" for n, t in zip(df["NSubjects"], df["NSubjectsTotal"])]
    return _hbar_png(df["Network"], df["CohortMeanPercentDamage"],
                     "Mean percent of network damaged, all subjects (%)", title, "#2f6f8a",
                     annot=annot)


def group_tract_bar_png(df, top=15, title="Tract overlap (group)") -> str | None:
    if df is None or df.empty:
        return None
    d = df.head(top)
    annot = [f"n={n}/{t}" for n, t in zip(d["NSubjects"], d["NSubjectsTotal"])]
    return _hbar_png(d["Tract"], d["CohortMeanPercentOverlap"],
                     "Mean percent of tract territory lesioned, all subjects (%)",
                     title, "#5b8c3e", annot=annot)
