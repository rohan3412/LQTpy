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


def _fig_to_uri(fig, dpi: int = 110) -> str:
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


def glass_brain_png(lesion_img, title="Lesion (glass brain)") -> str | None:
    from nilearn import plotting

    def build():
        disp = plotting.plot_glass_brain(
            lesion_img, title=title, cmap="autumn", colorbar=False,
            display_mode="lyrz", alpha=0.9, threshold=0.5,
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)


def slices_png(lesion_img, cut_coords, title="Lesion overlay") -> str | None:
    from nilearn import plotting

    def build():
        disp = plotting.plot_roi(
            lesion_img, title=title, cut_coords=cut_coords,
            display_mode="ortho", cmap="autumn", alpha=0.7, colorbar=False,
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


def render_ggseg(data: dict, atlas_key: str, cmap="Reds", vmax=None,
                 title="", ylabel="Percent damage (%)", lw=1.6,
                 edgecolor="black", na_color="#b8bcc2", figsize=(9, 6),
                 fontsize=13) -> str | None:
    """Draw a ggseg flat-brain map with the classic thick-black-border style.

    ``data`` maps ggseg region keys -> values. Labelled regions are coloured by
    ``cmap``; every other atlas region is filled ``na_color``. Works for any
    ggseg atlas directory (``dk``, ``aseg``, ...).
    """
    if not data:
        return None

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
        ax = ggseg._create_figure_(outline_txt, figsize, "white", title,
                                   fontsize, "black")

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
        cm, norm = ggseg._get_cmap_(cmap, data.values(), vminmax=[0, vmx])
        for key, val in data.items():
            fp = os.path.join(wd, key)
            if os.path.isfile(fp):
                patch(open(fp).read(), cm(norm(val)))

        ggseg._add_colorbar_(ax, cm, norm, "black", fontsize * 0.7, ylabel)
        fig = ax.figure
        fig.set_facecolor("white")
        return _fig_to_uri(fig)
    return _safe(build)


def ggseg_aseg_png(damage_by_aseg: dict,
                   title="Subcortical damage (aseg)") -> str | None:
    """Render an aseg subcortical/cerebellar map coloured by percent damage."""
    return render_ggseg(damage_by_aseg, "aseg", cmap="Reds", title=title)


def group_overlap_png(freq_img, n_subjects, title=None) -> str | None:
    from nilearn import plotting

    def build():
        ttl = title or f"Lesion overlap (n={n_subjects})"
        disp = plotting.plot_stat_map(
            freq_img, title=ttl, cmap="hot", colorbar=True, threshold=0.5,
            vmax=max(1, int(n_subjects)), display_mode="z", cut_coords=6,
            black_bg=False,
        )
        fig = disp.frame_axes.figure
        return _fig_to_uri(fig)
    return _safe(build)


def group_damage_bar_png(df, top=15, title="Most frequently damaged regions") -> str | None:
    """Group bar chart: mean percent damage with subject-count annotation."""
    if df is None or df.empty:
        return None
    d = df.head(top)
    annot = [f"n={n}" for n in d["NSubjects"]]
    return _hbar_png(d["RegionName"], d["MeanPercentDamage"],
                     "Mean percent damage across affected subjects (%)",
                     title, "#8a2f8f", annot=annot)


def group_network_bar_png(df, title="Network involvement (group)") -> str | None:
    if df is None or df.empty:
        return None
    annot = [f"n={n}" for n in df["NSubjects"]]
    return _hbar_png(df["Network"], df["MeanPercentDamage"],
                     "Mean percent of network damaged (%)", title, "#2f6f8a", annot=annot)


def group_tract_bar_png(df, top=15, title="Tract overlap (group)") -> str | None:
    if df is None or df.empty:
        return None
    d = df.head(top)
    annot = [f"n={n}" for n in d["NSubjects"]]
    return _hbar_png(d["Tract"], d["MeanPercentOverlap"],
                     "Mean percent of tract territory lesioned (%)",
                     title, "#5b8c3e", annot=annot)
