"""Generic ggseg-style flat-brain renderer (DK, Schaefer-100).

Geometry was exported once from R (``tools/export_ggseg_geometry.R``) into
``data/ggseg_<atlas>.csv.gz``: per-region polygon vertices for four views
(lateral, medial, inferior, superior) x two hemispheres. This module lays that
geometry into a labeled 2 (hemisphere) x 4 (view) grid -- title on top, view
names along the bottom, hemisphere names down the left, axis titles "side" /
"hemisphere" -- matching the classic ggseg atlas-plot look, then fills each
region by its damage value with thin black borders.
"""
from __future__ import annotations

import base64
import io
import os
import threading
import warnings

import numpy as np
import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

_VIEWS = ("lateral", "medial", "inferior", "superior")
_ROWS = ("right", "left")  # top row, bottom row -- matches the reference plot

# How to build the damage-dict lookup key from a geometry row's
# (region, hemi) columns, per atlas. Schaefer region names already encode
# hemisphere (e.g. "7Networks_LH_Vis_1"); DK region names don't, so hemi is
# appended to match lqtpy.atlases' ggseg_map values (e.g. "fusiform_left").
_KEY_FN = {
    "dk": lambda region, hemi: f"{region}_{hemi}",
    "schaefer100": lambda region, hemi: region,
}

_GEOM_CACHE: dict[str, list] = {}
#: Guards the double-checked lazy build of _GEOM_CACHE entries so concurrent
#: first callers build exactly once and all see the same object.
_geom_lock = threading.Lock()


def _load_geometry(atlas_key: str):
    """Parse ``ggseg_<atlas_key>.csv.gz`` into laid-out (key, polygon) tuples."""
    layout = _GEOM_CACHE.get(atlas_key)
    if layout is not None:
        return layout

    with _geom_lock:
        layout = _GEOM_CACHE.get(atlas_key)
        if layout is not None:
            return layout
        return _build_geometry(atlas_key)


def _build_geometry(atlas_key: str):
    path = os.path.join(_DATA_DIR, f"ggseg_{atlas_key}.csv.gz")
    df = pd.read_csv(path)
    df = df[df["view"].isin(_VIEWS)].copy()
    key_fn = _KEY_FN[atlas_key]

    # Bounding box per (hemi, view) panel.
    bbox = {}
    for (hemi, view), s in df.groupby(["hemi", "view"]):
        bbox[(hemi, view)] = (s["x"].min(), s["y"].min(),
                             s["x"].max() - s["x"].min(),
                             s["y"].max() - s["y"].min())

    gap_frac = 0.12
    # Column x-offsets: width per view = max width across hemispheres.
    col_w = {v: max(bbox.get((h, v), (0, 0, 0, 0))[2] for h in _ROWS) for v in _VIEWS}
    col_x0 = {}
    x = 0.0
    for v in _VIEWS:
        col_x0[v] = x
        x += col_w[v] * (1 + gap_frac)

    # Row height: uniform, tallest panel overall, so hemispheres line up.
    row_h = max(b[3] for b in bbox.values())
    row_y0 = {"right": row_h * (1 + gap_frac), "left": 0.0}

    polys = []
    key = ["hemi", "view", "region", "group", "subgroup"]
    for (hemi, view, region, _g, _sg), s in df.groupby(key, sort=False):
        x0, y0, w, h = bbox[(hemi, view)]
        # Centre each panel within its column/row cell.
        xoff = col_x0[view] + (col_w[view] - w) / 2 - x0
        yoff = row_y0[hemi] + (row_h - h) / 2 - y0
        xs = s["x"].to_numpy() + xoff
        ys = s["y"].to_numpy() + yoff
        xy = np.column_stack([xs, ys])
        xy.flags.writeable = False
        polys.append((key_fn(region, hemi), xy))

    layout = {
        "polys": polys,
        "col_centers": {v: col_x0[v] + col_w[v] / 2 for v in _VIEWS},
        "row_centers": {h: row_y0[h] + row_h / 2 for h in _ROWS},
        "xlim": (0, x - col_w[_VIEWS[-1]] * gap_frac),
        "ylim": (0, row_y0["right"] + row_h),
    }
    _GEOM_CACHE[atlas_key] = layout
    return layout


#: Report themes: figure background, text, region outlines, unaffected
#: fill, and the lesion-hue magnitude ramp (low -> high damage). In dark
#: mode the ramp runs dark -> bright so more damage reads as brighter; it is
#: not a simple inversion of the light ramp.
THEMES = {
    "light": {"bg": "#ffffff", "fg": "#14181f", "edge": "#2b2f36", "na": "#d9dce1",
              "ramp": ["#fde5d6", "#f9b890", "#f08a55", "#d9612b", "#a8431a", "#72290d"]},
    "dark": {"bg": "#161a21", "fg": "#e8ebf0", "edge": "#0b0d11", "na": "#3a414c",
             "ramp": ["#5a2c17", "#8a3d1c", "#c25224", "#eb6d35", "#f89a65", "#ffc9a3"]},
}


def lesion_cmap(theme: str = "light"):
    """The report's lesion-hue magnitude colormap for ``theme``."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(f"lqtpy_lesion_{theme}", THEMES[theme]["ramp"])


def _fig_to_uri(fig, dpi=115, facecolor="white"):
    import matplotlib.pyplot as plt

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight",
                facecolor=facecolor)
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def flat_map_png(atlas_key: str, damage_by_region: dict, title="",
                 cmap=None, na_color=None, edgecolor=None,
                 lw=1.2, vmax=None, ylabel="Percent damaged (%)",
                 theme: str = "light") -> str | None:
    """Render a labeled 2x4 (hemisphere x view) ggseg flat-brain damage map.

    ``damage_by_region`` maps region keys (see ``_KEY_FN`` per atlas) -> value.
    ``theme`` ("light"/"dark") sets background, text, outlines and the
    lesion-hue ramp; explicit ``cmap``/``na_color``/``edgecolor`` override it.
    Returns a base64 ``data:`` PNG URI, or None if geometry is unavailable or
    rendering fails.
    """
    t = THEMES[theme]
    cmap = cmap or lesion_cmap(theme)
    na_color = na_color or t["na"]
    edgecolor = edgecolor or t["edge"]
    path = os.path.join(_DATA_DIR, f"ggseg_{atlas_key}.csv.gz")
    if not os.path.exists(path):
        return None
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection
    import matplotlib.colors as mcolors
    import matplotlib.cm as cm

    try:
        layout = _load_geometry(atlas_key)
        vmx = vmax if vmax is not None else max(1.0, max(damage_by_region.values(), default=1.0))
        norm = mcolors.Normalize(vmin=0, vmax=vmx)
        sm = cm.ScalarMappable(norm=norm, cmap=cmap)

        fig, ax = plt.subplots(figsize=(11, 5.5))
        fig.set_facecolor(t["bg"])
        ax.set_facecolor(t["bg"])
        patches, colors = [], []
        for key, xy in layout["polys"]:
            patches.append(Polygon(xy, closed=True))
            val = damage_by_region.get(key)
            colors.append(sm.to_rgba(val) if val is not None else na_color)
        ax.add_collection(PatchCollection(
            patches, facecolors=colors, edgecolors=edgecolor, linewidths=lw))

        x0, x1 = layout["xlim"]
        y0, y1 = layout["ylim"]
        pad = 0.06 * (x1 - x0)
        ax.set_xlim(x0 - pad, x1 + pad)
        ax.set_ylim(y0 - pad, y1 + pad * 1.4)
        ax.set_aspect("equal")
        ax.set_xticks([layout["col_centers"][v] for v in _VIEWS])
        ax.set_xticklabels(_VIEWS, fontsize=10, color=t["fg"])
        ax.set_yticks([layout["row_centers"][h] for h in _ROWS])
        ax.set_yticklabels(_ROWS, fontsize=10, color=t["fg"])
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xlabel("side", fontsize=11, color=t["fg"])
        ax.set_ylabel("hemisphere", fontsize=11, color=t["fg"])
        if title:
            ax.set_title(title, fontsize=13, color=t["fg"], pad=10)

        cb = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.03)
        cb.set_label(ylabel, fontsize=9, color=t["fg"])
        cb.ax.tick_params(labelsize=8, colors=t["fg"])
        cb.outline.set_edgecolor(t["fg"])
        return _fig_to_uri(fig, facecolor=t["bg"])
    except Exception as exc:
        warnings.warn(f"{atlas_key} flat plot failed: {exc}")
        plt.close("all")
        return None


def dk_flat_png(damage_by_region: dict, title="",
                **kw) -> str | None:
    return flat_map_png("dk", damage_by_region, title=title, **kw)


def schaefer_flat_png(damage_by_region: dict, title="", **kw) -> str | None:
    return flat_map_png("schaefer100", damage_by_region, title=title, **kw)
