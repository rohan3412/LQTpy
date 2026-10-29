"""Brain figures for reports, drawn with numpy (no matplotlib, no nilearn).

Every figure is an inline SVG that works on both report themes: raster
panels are PNGs with a transparent background (the plate behind them is
black in the dark dashboard and white in light mode, the clinical view and
the PDF), and labels are SVG text coloured by the page's CSS. One render
serves both themes, which is what brought a single report with figures back
from ~8 s to well under a second.

The anatomy is the bundled tissue atlas (cortex, white matter, deep grey
matter, ventricles, cerebellum, brainstem, CSF) shaded as a stylised T1
image, so no template download is needed.

- :func:`glass_brain_svg`  maximum-intensity projections (left, front, top)
  with brain, ventricle and deep-grey outlines.
- :func:`slices_svg`       three orthogonal slices through a point.
- :func:`overlap_svg`      a group frequency map on axial slices (theme
  specific colour ramps, so it returns ``{"light": svg, "dark": svg}``).
- :func:`thumbnail_uri`    a small axial projection for group tables.
- :func:`flat_map_svg`, :func:`aseg_svg`  vector ggseg-style region maps
  whose regions carry ``data-hl`` keys, so a report can highlight a region
  on the map when its table row is hovered (and the reverse).
- :func:`viewer_data`      sprites and parameters for the scroll-through
  slice viewer (brainsprite, embedded in the report).
"""
from __future__ import annotations

import base64
import html
import struct
import threading
import zlib

import numpy as np
from scipy import ndimage

from ._diskcache import once

__all__ = ["png_bytes", "png_uri", "glass_brain_svg", "slices_svg", "overlap_svg",
           "thumbnail_uri", "flat_map_svg", "aseg_svg", "viewer_data", "LESION_RGB"]

#: The report's lesion colour (same hue as the dashboard bars).
LESION_RGB = np.array([245, 121, 58], float) / 255.0
_LINE = 0.56  # grey of outlines: readable on black and on white


# --------------------------------------------------------------------------- #
# PNG
# --------------------------------------------------------------------------- #
def png_bytes(img) -> bytes:
    """Encode a uint8 image (H x W gray, H x W x 3 RGB or H x W x 4 RGBA)."""
    img = np.ascontiguousarray(img, dtype=np.uint8)
    if img.ndim == 2:
        ch, ctype = 1, 0
    else:
        ch = img.shape[2]
        ctype = {1: 0, 2: 4, 3: 2, 4: 6}[ch]
    h, w = img.shape[:2]
    raw = np.zeros((h, w * ch + 1), np.uint8)
    raw[:, 1:] = img.reshape(h, -1)

    def chunk(tag, data):
        out = struct.pack(">I", len(data)) + tag + data
        return out + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw.tobytes(), 6)) + chunk(b"IEND", b""))


def png_uri(img) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes(img)).decode("ascii")


def _rgba_u8(rgb, alpha) -> np.ndarray:
    out = np.concatenate([np.clip(rgb, 0, 1), np.clip(alpha, 0, 1)[..., None]], axis=-1)
    return np.round(out * 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Template (tissue atlas, reoriented to RAS)
# --------------------------------------------------------------------------- #
# Stylised T1 intensities per tissue label (0 = background).
_INTENSITY = np.zeros(256, np.float32)
_INTENSITY[[1, 2]] = 0.50      # cortex
_INTENSITY[[3, 4]] = 0.84      # white matter
_INTENSITY[[5, 6]] = 0.63      # deep grey
_INTENSITY[[7, 8, 9]] = 0.14   # ventricles
_INTENSITY[[10, 11]] = 0.58    # cerebellum
_INTENSITY[12] = 0.74          # brainstem
_INTENSITY[[13, 14]] = 0.22    # CSF


@once
def _template():
    """``(labels_ras, affine_ras, ornt)``: the tissue atlas in RAS order and
    the orientation transform from the atlas grid (for lesion masks)."""
    import nibabel as nib
    from .atlases import load_atlas
    at = load_atlas("tissue")
    aff = np.asarray(at.img.affine, float)
    ornt = nib.orientations.io_orientation(aff)
    lab = nib.orientations.apply_orientation(at.label_array(), ornt)
    lab = np.ascontiguousarray(lab.astype(np.uint8))
    lab.flags.writeable = False
    aff_ras = aff @ nib.orientations.inv_ornt_aff(ornt, at.label_array().shape)
    return lab, aff_ras, ornt, at


def _lesion_ras(lesion, threshold=0.5) -> np.ndarray:
    """(N, 3) voxel indices of the lesion on the template grid, in RAS order
    (where ``nibabel.orientations.apply_orientation`` puts them), cached on
    the prepared lesion."""
    from .prepared import prepare
    _, _, ornt, at = _template()
    p = prepare(lesion, threshold)

    def build():
        gm = p.on_grid(at.img)
        ijk = gm.ijk()
        out = np.empty_like(ijk)
        for k in range(3):
            v = ijk[:, k]
            out[:, int(ornt[k, 0])] = gm.shape[k] - 1 - v if ornt[k, 1] < 0 else v
        out.flags.writeable = False
        return out
    return p.cached(("figures_ras",), build)


def _plane(ijk, shape, axis: int, index: "int | None" = None) -> np.ndarray:
    """The lesion seen along ``axis``: voxels per pixel summed through the
    volume (``index=None``, a projection) or the slice at ``index`` (bool)."""
    keep = [a for a in range(3) if a != axis]
    pts = ijk if index is None else ijk[ijk[:, axis] == index]
    n = shape[keep[0]] * shape[keep[1]]
    cnt = np.bincount(pts[:, keep[0]] * shape[keep[1]] + pts[:, keep[1]], minlength=n)
    cnt = cnt.reshape(shape[keep[0]], shape[keep[1]])
    return cnt if index is None else cnt > 0


def _pooled(ijk, step: int, shape) -> np.ndarray:
    """Distinct voxel indices after max-pooling by ``step`` (a downsampled
    voxel is set when any voxel it covers is), inside ``shape``."""
    d = np.unique(ijk // step, axis=0) if step > 1 else ijk
    return d[np.all(d < np.asarray(shape), axis=1)]


def _to_display(plane: np.ndarray, flip_cols: bool = False) -> np.ndarray:
    """A plane indexed (horizontal axis, vertical axis) in RAS order ->
    image rows top-down."""
    img = plane.T[::-1]
    return img[:, ::-1] if flip_cols else img


def _smooth_shade(lab2d: np.ndarray, sigma: float = 0.6):
    a = (lab2d > 0).astype(np.float32)
    i = _INTENSITY[lab2d]
    ga = ndimage.gaussian_filter(a, sigma)
    gi = ndimage.gaussian_filter(i * a, sigma)
    return np.where(ga > 1e-3, gi / np.maximum(ga, 1e-3), 0.0), ga


def _zoom(a, s):
    return ndimage.zoom(np.asarray(a, np.float32), s, order=1, prefilter=False) if s != 1 else a


def _edge(m: np.ndarray) -> np.ndarray:
    return m & ~ndimage.binary_erosion(m)


# --------------------------------------------------------------------------- #
# SVG assembly
# --------------------------------------------------------------------------- #
def _svg_panels(panels, *, label: str, gap: int = 14, extra: str = "", pad_right: int = 0,
                css_class: str = "bfig") -> str:
    """``panels``: [(uri, w, h, [(x, y, text, anchor), ...])] laid out left
    to right, bottoms aligned."""
    H = max(p[2] for p in panels) + 8
    W = sum(p[1] for p in panels) + gap * (len(panels) - 1) + pad_right
    out = [f'<svg class="{css_class}" viewBox="0 0 {W} {H}" role="img" '
           f'aria-label="{html.escape(label)}" preserveAspectRatio="xMidYMid meet">']
    x = 0
    for uri, w, h, texts in panels:
        y = H - h - 4
        out.append(f'<image href="{uri}" x="{x}" y="{y}" width="{w}" height="{h}"/>')
        for tx, ty, t, anchor in texts:
            out.append(f'<text x="{x + tx:.0f}" y="{y + ty:.0f}" text-anchor="{anchor}">'
                       f'{html.escape(t)}</text>')
        x += w + gap
    out.append(extra)
    out.append("</svg>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Glass brain
# --------------------------------------------------------------------------- #
_GLASS: dict = {}


def _glass_base(lab, axis, scale, flip_cols):
    """The template part of a glass-brain panel (brain, ventricle, deep-grey
    and hindbrain outlines): the same for every lesion, drawn once."""
    key = (lab.shape, axis, scale, flip_cols)
    if key not in _GLASS:
        def proj(m):
            return _to_display(m.any(axis=axis), flip_cols)

        brain = proj(lab > 0)
        parts = [(proj(np.isin(lab, (7, 8, 9))), 0.75), (proj(np.isin(lab, (5, 6))), 0.5),
                 (proj(np.isin(lab, (10, 11, 12))), 0.55)]
        bz = _zoom(brain, scale) > 0.5
        rgb = np.full(bz.shape + (3,), _LINE, np.float32)
        alpha = np.where(bz, 0.10, 0.0).astype(np.float32)
        for m, a in parts:
            e = _edge(_zoom(m, scale) > 0.5)
            alpha = np.where(e, np.maximum(alpha, a), alpha)
        e = _edge(bz)
        alpha = np.where(e, 1.0, alpha)
        rgb.flags.writeable = alpha.flags.writeable = False
        _GLASS[key] = (rgb, alpha)
    return _GLASS[key]


def _glass_panel(lab, cnt, axis, scale, flip_cols=False):
    """``cnt``: the lesion's voxels per pixel, projected along ``axis``."""
    rgb, alpha = _glass_base(lab, axis, scale, flip_cols)
    cnt = _to_display(cnt.astype(np.float32), flip_cols)
    if cnt.max() > 0:
        c = _zoom(cnt, scale)
        dens = c / max(float(c.max()), 1e-6)
        la = np.clip((c - 0.25) / 0.5, 0, 1) * (0.62 + 0.38 * np.sqrt(dens))
        rgb = rgb * (1 - la[..., None]) + LESION_RGB * la[..., None]
        alpha = np.maximum(alpha, la)
    return _rgba_u8(rgb, alpha)


def glass_brain_svg(lesion, *, threshold: float = 0.5, scale: float = 2.0) -> str:
    """Projections of the lesion through the brain: from the left, from the
    front and from above (neurological convention: left is on the left)."""
    lab, _, _, _ = _template()
    les = _lesion_ras(lesion, threshold)
    views = [(0, True, [("A", "P"), None]), (1, False, [("L", "R"), None]),
             (2, False, [("L", "R"), None])]
    panels = []
    for axis, flip, (lr, _) in views:
        img = _glass_panel(lab, _plane(les, lab.shape, axis), axis, scale, flip_cols=flip)
        h, w = img.shape[:2]
        panels.append((png_uri(img), w, h, [(10, 18, lr[0], "start"), (w - 10, 18, lr[1], "end")]))
    return _svg_panels(panels, label="Glass-brain projections of the lesion: from the left, "
                                     "from the front and from above")


# --------------------------------------------------------------------------- #
# Slices
# --------------------------------------------------------------------------- #
def _slice_panel(lab2d, les2d, scale):
    shade, a = _smooth_shade(lab2d)
    shade, a = _zoom(shade, scale), _zoom(a, scale)
    rgb = np.repeat(shade[..., None], 3, axis=-1)
    alpha = np.clip(a * 1.15, 0, 1)
    if les2d.any():
        lz = _zoom(les2d.astype(np.float32), scale)
        la = np.clip((lz - 0.3) / 0.4, 0, 1)
        edge = _edge(lz > 0.5)
        mix = la * 0.72
        mix = np.where(edge, 1.0, mix)
        rgb = rgb * (1 - mix[..., None]) + LESION_RGB * mix[..., None]
        alpha = np.maximum(alpha, la)
    return _rgba_u8(rgb, alpha)


def slices_svg(lesion, cut_mm=None, *, threshold: float = 0.5, scale: float = 2.0) -> str:
    """Sagittal, coronal and axial slices through ``cut_mm`` (default: the
    lesion's centre of mass), neurological convention."""
    lab, aff, _, _ = _template()
    les = _lesion_ras(lesion, threshold)
    if cut_mm is None:
        c = les.mean(0) if len(les) else np.array(lab.shape) / 2
        cut_mm = aff[:3, :3] @ c + aff[:3, 3]
    vox = np.round(np.linalg.inv(aff) @ np.append(np.asarray(cut_mm, float), 1))[:3].astype(int)
    vox = np.clip(vox, 0, np.array(lab.shape) - 1)
    x, y, z = (float(v) for v in cut_mm)
    specs = [("x", x, lab[vox[0]], _plane(les, lab.shape, 0, vox[0]), ("P", "A")),
             ("y", y, lab[:, vox[1]], _plane(les, lab.shape, 1, vox[1]), ("L", "R")),
             ("z", z, lab[:, :, vox[2]], _plane(les, lab.shape, 2, vox[2]), ("L", "R"))]
    panels = []
    for name, val, l2, m2, lr in specs:
        img = _slice_panel(_to_display(l2), _to_display(m2), scale)
        h, w = img.shape[:2]
        panels.append((png_uri(img), w, h, [(8, 18, f"{name} = {val:.0f}", "start"),
                                             (8, h - 8, lr[0], "start"),
                                             (w - 8, h - 8, lr[1], "end")]))
    return _svg_panels(panels, label=f"Slices through {x:.0f}, {y:.0f}, {z:.0f} mm")


# --------------------------------------------------------------------------- #
# Group overlap
# --------------------------------------------------------------------------- #
def _ramp(stops: "list[str]", t: np.ndarray) -> np.ndarray:
    cols = np.array([[int(s[i:i + 2], 16) / 255.0 for i in (1, 3, 5)] for s in stops])
    t = np.clip(np.asarray(t, float), 0, 1) * (len(stops) - 1)
    lo = np.floor(t).astype(int).clip(0, len(stops) - 2)
    f = (t - lo)[..., None]
    return cols[lo] * (1 - f) + cols[lo + 1] * f


def ramp_hex(stops: "list[str]", t: float) -> str:
    r = _ramp(stops, np.array([t]))[0]
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in r)


def _overlap_ramp(theme: str) -> "list[str]":
    # On white the palest stops vanish against the anatomy: start saturated.
    return THEMES["light"]["ramp"][2:] if theme == "light" else THEMES["dark"]["ramp"][1:]


def overlap_svg(freq_img, n_subjects: int, *, n_cuts: int = 6, scale: float = 1.4) -> dict:
    """Axial slices of a subject-count map; ``{"light": svg, "dark": svg}``."""
    import nibabel as nib
    lab, aff, ornt, at = _template()
    data = np.asarray(freq_img.dataobj)
    if data.shape != at.label_array().shape or not np.allclose(freq_img.affine, at.img.affine,
                                                                 atol=1e-3):
        from ._resample import resample_to_img
        data = np.asarray(resample_to_img(freq_img, at.img, interpolation="nearest").dataobj)
    f = np.asarray(nib.orientations.apply_orientation(data, ornt), float)
    zs = np.flatnonzero(f.any(axis=(0, 1)))
    if not len(zs):
        zs = np.flatnonzero(lab.any(axis=(0, 1)))
    cuts = np.unique(np.linspace(zs[0], zs[-1], n_cuts + 2)[1:-1].round().astype(int))
    vmax = max(1, int(n_subjects))
    out = {}
    for theme in ("light", "dark"):
        stops = _overlap_ramp(theme)
        panels = []
        for k in cuts:
            shade, a = _smooth_shade(_to_display(lab[:, :, k]))
            shade, a = _zoom(shade, scale), _zoom(a, scale)
            rgb = np.repeat(shade[..., None], 3, axis=-1)
            alpha = np.clip(a * 1.15, 0, 1)
            fz = ndimage.zoom(_to_display(f[:, :, k]), scale, order=0)
            on = fz > 0
            col = _ramp(stops, (fz - 1) / max(vmax - 1, 1))
            rgb = np.where(on[..., None], col * 0.92 + rgb * 0.08, rgb)
            alpha = np.where(on, 1.0, alpha)
            img = _rgba_u8(rgb, alpha)
            h, w = img.shape[:2]
            zmm = (aff @ np.array([0, 0, k, 1]))[2]
            panels.append((png_uri(img), w, h, [(6, 16, f"z = {zmm:.0f}", "start")]))
        gid = f"ovr-{theme}"
        stops_svg = "".join(f'<stop offset="{i / (len(stops) - 1):.3f}" stop-color="{c}"/>'
                            for i, c in enumerate(stops))
        W = sum(p[1] for p in panels) + 14 * (len(panels) - 1)
        H = max(p[2] for p in panels) + 8
        bar = (f'<defs><linearGradient id="{gid}" x1="0" x2="0" y1="1" y2="0">{stops_svg}'
               f'</linearGradient></defs><rect x="{W + 14}" y="12" width="12" height="{H - 30}" '
               f'fill="url(#{gid})"/><text x="{W + 30}" y="22">{vmax}</text>'
               f'<text x="{W + 30}" y="{H - 18}">1</text>'
               f'<text x="{W + 20}" y="{H - 2}" text-anchor="middle">subjects</text>')
        out[theme] = _svg_panels(panels, label="Number of subjects with a lesion at each voxel",
                                 extra=bar, pad_right=70)
    return out


# --------------------------------------------------------------------------- #
# Thumbnail
# --------------------------------------------------------------------------- #
def thumbnail_uri(lesion, *, threshold: float = 0.5) -> str:
    """A small axial projection (about 91 x 109 px) for group tables."""
    lab, _, _, _ = _template()
    lab2 = _downsample(lab, 2)
    les = _pooled(_lesion_ras(lesion, threshold), 2, lab2.shape)
    return png_uri(_glass_panel(lab2, _plane(les, lab2.shape, 2), 2, 1.0))


# --------------------------------------------------------------------------- #
# Vector region maps (ggseg geometry)
# --------------------------------------------------------------------------- #
_VIEWS = ("lateral", "medial", "inferior", "superior")
_ROWS = ("right", "left")
_GEOM: dict = {}
_geom_lock = threading.Lock()
#: Theme colour ramps for region maps (light and dark report themes).
THEMES = {
    "light": {"ramp": ["#fde5d6", "#f9b890", "#f08a55", "#d9612b", "#a8431a", "#72290d"]},
    "dark": {"ramp": ["#5a2c17", "#8a3d1c", "#c25224", "#eb6d35", "#f89a65", "#ffc9a3"]},
}


def _geometry(atlas_key: str) -> dict:
    """ggseg flat-map polygons laid out in a 2 x 4 grid (hemispheres x
    views), read with the csv module (no pandas)."""
    g = _GEOM.get(atlas_key)
    if g is not None:
        return g
    with _geom_lock:
        if atlas_key in _GEOM:
            return _GEOM[atlas_key]
        import csv
        import gzip
        import os
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                            f"ggseg_{atlas_key}.csv.gz")
        groups: "dict[tuple, list]" = {}
        with gzip.open(path, "rt", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["view"] not in _VIEWS:
                    continue
                k = (r["hemi"], r["view"], r["region"], r["group"], r["subgroup"])
                groups.setdefault(k, []).append((float(r["x"]), float(r["y"])))
        pts = {k: np.array(v) for k, v in groups.items()}
        bbox = {}
        for (hemi, view, *_), xy in pts.items():
            b = bbox.get((hemi, view))
            lo, hi = xy.min(0), xy.max(0)
            bbox[(hemi, view)] = (np.minimum(b[0], lo), np.maximum(b[1], hi)) if b else (lo, hi)
        gap = 0.12
        col_w = {v: max((bbox[(h, v)][1][0] - bbox[(h, v)][0][0]) for h in _ROWS if (h, v) in bbox)
                 for v in _VIEWS}
        col_x0, x = {}, 0.0
        for v in _VIEWS:
            col_x0[v] = x
            x += col_w[v] * (1 + gap)
        row_h = max(b[1][1] - b[0][1] for b in bbox.values())
        row_y0 = {"right": row_h * (1 + gap), "left": 0.0}
        key_fn = (lambda region, hemi: f"{region}_{hemi}") if atlas_key == "dk" else (
            lambda region, hemi: region)
        polys = []
        for (hemi, view, region, _g, _sg), xy in pts.items():
            (x0, y0), (x1, y1) = bbox[(hemi, view)]
            w, h = x1 - x0, y1 - y0
            off = np.array([col_x0[view] + (col_w[view] - w) / 2 - x0,
                            row_y0[hemi] + (row_h - h) / 2 - y0])
            polys.append((key_fn(region, hemi), xy + off))
        g = _GEOM[atlas_key] = {
            "polys": polys,
            "col_centers": {v: col_x0[v] + col_w[v] / 2 for v in _VIEWS},
            "row_centers": {h: row_y0[h] + row_h / 2 for h in _ROWS},
            "xlim": (0, x - col_w[_VIEWS[-1]] * gap), "ylim": (0, row_y0["right"] + row_h)}
    return g


def _theme_colours(value, vmax):
    t = 0.0 if not vmax else float(value) / float(vmax)
    return ramp_hex(THEMES["light"]["ramp"], t), ramp_hex(THEMES["dark"]["ramp"], t)


def _colourbar(x, y, h, vmax, label, uid, k: float = 1.0) -> str:
    parts = []
    for theme in ("light", "dark"):
        st = THEMES[theme]["ramp"]
        stops = "".join(f'<stop offset="{i / (len(st) - 1):.3f}" stop-color="{c}"/>'
                        for i, c in enumerate(st))
        parts.append(f'<linearGradient id="{uid}-{theme}" x1="0" x2="0" y1="1" y2="0">{stops}'
                     "</linearGradient>")
    return (f'<defs>{"".join(parts)}</defs>'
            f'<rect class="cb-light" x="{x:.0f}" y="{y:.0f}" width="{14 * k:.0f}" height="{h:.0f}" '
            f'fill="url(#{uid}-light)"/>'
            f'<rect class="cb-dark" x="{x:.0f}" y="{y:.0f}" width="{14 * k:.0f}" height="{h:.0f}" '
            f'fill="url(#{uid}-dark)"/>'
            f'<text x="{x + 20 * k:.0f}" y="{y + 10 * k:.0f}">{vmax:.0f}%</text>'
            f'<text x="{x + 20 * k:.0f}" y="{y + h:.0f}">0</text>'
            f'<text x="{x + 7 * k:.0f}" y="{y + h + 22 * k:.0f}" text-anchor="middle">'
            f'{html.escape(label)}</text>')


def _path_el(d, key, value, vmax, title, hl) -> str:
    hl_attr = f' data-hl="{html.escape(hl)}"' if hl else ""
    tip = f"<title>{html.escape(title)}</title>" if title else ""
    if value is None:
        return f'<path class="na" d="{d}"{hl_attr}>{tip}</path>'
    fl, fd = _theme_colours(value, vmax)
    return f'<path d="{d}" style="--fl:{fl};--fd:{fd}"{hl_attr}>{tip}</path>'


def _poly_d(p: np.ndarray, tol: float = 2.0) -> str:
    """SVG subpath for a polygon, dropping vertices closer than ``tol``
    units to the last one kept (the geometry is far denser than the map is
    drawn; Schaefer-100 went from 1.2 MB to a fraction of that)."""
    keep = [0]
    last = p[0]
    for i in range(1, len(p)):
        if abs(p[i, 0] - last[0]) + abs(p[i, 1] - last[1]) >= tol:
            keep.append(i)
            last = p[i]
    q = np.round(p[keep], 1)
    if len(q) < 3:
        q = np.round(p, 1)
    return "M" + "L".join(f"{a:g},{b:g}" for a, b in q) + "Z"


_PATHS: dict = {}


def _map_paths(atlas_key, layout, s, padl, padt) -> "dict[str, str]":
    """Region key -> SVG path data of a flat map: fixed geometry, built once."""
    key = (atlas_key, s, padl, padt)
    if key not in _PATHS:
        x0, y1 = layout["xlim"][0], layout["ylim"][1]
        by_key: "dict[str, list]" = {}
        for k, xy in layout["polys"]:
            by_key.setdefault(k, []).append(
                np.column_stack([(xy[:, 0] - x0) * s + padl, (y1 - xy[:, 1]) * s + padt]))
        _PATHS[key] = {k: "".join(_poly_d(p) for p in polys) for k, polys in by_key.items()}
    return _PATHS[key]


def flat_map_svg(atlas_key: str, values: dict, *, labels: "dict | None" = None,
                 vmax: "float | None" = None, label: str = "Percent damaged",
                 uid: str = "fm") -> "str | None":
    """A 2 (hemisphere) x 4 (view) region map as inline SVG.

    ``atlas_key``: ggseg geometry key (``dk`` or ``schaefer100``).
    ``values``: ggseg region key -> value; ``labels``: key -> ``(name,
    highlight_key)`` for tooltips and table linking."""
    try:
        layout = _geometry(atlas_key)
    except Exception:
        return None
    labels = labels or {}
    vmx = vmax if vmax is not None else max(1.0, max(values.values(), default=1.0))
    x0, x1 = layout["xlim"]
    y0, y1 = layout["ylim"]
    s = 1000.0 / (x1 - x0)
    padl, padt = 70, 10
    W, Hmap = 1000 + padl, (y1 - y0) * s
    H = Hmap + padt + 34

    paths = []
    for key, d in _map_paths(atlas_key, layout, s, padl, padt).items():
        name, hl = labels.get(key, (key.replace("_", " "), None))
        v = values.get(key)
        title = f"{name}: {v:.1f}%" if v is not None else name
        paths.append(_path_el(d, key, v, vmx, title, hl))
    texts = [f'<text x="{(layout["col_centers"][v] - x0) * s + padl:.0f}" y="{H - 12:.0f}" '
             f'text-anchor="middle">{v}</text>' for v in _VIEWS]
    texts += [f'<text x="{padl - 12}" y="{(y1 - layout["row_centers"][h]) * s + padt + 4:.0f}" '
              f'text-anchor="end">{h}</text>' for h in ("left", "right")]
    bar = _colourbar(W + 20, padt + 10, Hmap - 60, vmx, label, uid)
    return (f'<svg class="fm" viewBox="0 0 {W + 110:.0f} {H:.0f}" role="img" '
            f'aria-label="{html.escape(label)} map">{"".join(paths)}{"".join(texts)}{bar}</svg>')


def aseg_svg(values: dict, *, labels: "dict | None" = None, vmax: "float | None" = None,
             label: str = "Percent damaged", uid: str = "as") -> "str | None":
    """Subcortical/cerebellar region map on ggseg's aseg geometry (needs the
    ``ggseg`` package for the geometry files)."""
    import glob
    import os
    import re
    try:
        import ggseg
    except ImportError:
        return None
    d = os.path.join(os.path.dirname(ggseg.__file__), "data", "aseg")
    files = sorted(glob.glob(os.path.join(d, "*")))
    if not files or not values:
        return None
    labels = labels or {}
    vmx = vmax if vmax is not None else max(1.0, max(values.values(), default=1.0))
    shells, paths, nums = [], [], []
    for fp in files:
        key = os.path.basename(fp)
        text = open(fp, encoding="utf-8").read().strip()
        nums += [float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", text)]
        if key in ("Coronal", "Sagittal"):
            shells.append(f'<path class="shell" d="{text}"/>')
            continue
        name, hl = labels.get(key, (key.replace("-", " "), None))
        v = values.get(key)
        title = f"{name}: {v:.1f}%" if v is not None else name
        paths.append(_path_el(text, key, v, vmx, title, hl))
    xs, ys = np.array(nums[0::2]), np.array(nums[1::2])
    bx0, bx1, by0, by1 = xs.min(), xs.max(), ys.min(), ys.max()
    bw, bh = bx1 - bx0, by1 - by0
    k = bh / 300.0
    bar = _colourbar(bx1 + 40 * k, by0 + 20 * k, bh - 80 * k, vmx, label, uid, k)
    return (f'<svg class="fm aseg" viewBox="{bx0 - 10 * k:.0f} {by0 - 10 * k:.0f} '
            f'{bw + 170 * k:.0f} {bh + 40 * k:.0f}" role="img" aria-label="{html.escape(label)} map"'
            f' style="--fs:{12 * k:.1f}px;--sw:{0.8 * k:.2f}">'
            f'{"".join(shells)}{"".join(paths)}{bar}</svg>')


# --------------------------------------------------------------------------- #
# Slice viewer (brainsprite)
# --------------------------------------------------------------------------- #
_SPRITES: dict = {}


def _sprite(vol: np.ndarray) -> np.ndarray:
    """Sagittal slices pasted row by row (brainsprite's layout)."""
    nx, ny, nz = vol.shape[:3]
    nrows = int(np.ceil(np.sqrt(nx)))
    ncols = int(np.ceil(nx / nrows))
    out = np.zeros((nrows * nz, ncols * ny) + vol.shape[3:], vol.dtype)
    for xx in range(nx):
        r, c = divmod(xx, ncols)
        out[r * nz:(r + 1) * nz, c * ny:(c + 1) * ny] = np.swapaxes(vol[xx, :, ::-1], 0, 1)
    return out


def _downsample(vol, step):
    if step == 1:
        return vol
    return vol[::step, ::step, ::step]


def brainsprite_js() -> "str | None":
    """The brainsprite viewer script bundled with nilearn (MIT licence)."""
    import os
    try:
        import nilearn
    except ImportError:
        return None
    p = os.path.join(os.path.dirname(nilearn.__file__), "plotting", "data", "js",
                     "brainsprite.min.js")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return fh.read()


def viewer_data(lesion, *, threshold: float = 0.5, step: int = 2) -> "dict | None":
    """Everything the report's slice viewer needs: ``bg`` and ``overlay``
    sprite URIs and brainsprite ``params`` (without canvas/sprite ids)."""
    lab, aff, _, _ = _template()
    les = _lesion_ras(lesion, threshold)
    key = ("bg", step)
    if key not in _SPRITES:
        l2 = _downsample(lab, step)
        shade = _INTENSITY[l2]
        a = (l2 > 0).astype(np.float32)
        shade = ndimage.gaussian_filter(shade * a, 0.5) / np.maximum(ndimage.gaussian_filter(a, 0.5), 1e-3)
        rgba = _rgba_u8(np.repeat(shade[..., None], 3, axis=-1), np.clip(ndimage.gaussian_filter(a, 0.5) * 1.2, 0, 1))
        _SPRITES[key] = png_uri(_sprite(rgba))
    lab2 = _downsample(lab, step)
    ijk = _pooled(les, step, lab2.shape)
    shape = lab2.shape
    ov = np.zeros(shape + (4,), np.uint8)
    ov[tuple(ijk.T)] = [245, 121, 58, 255]
    aff2 = aff.copy()
    aff2[:3, :3] = aff[:3, :3] * step
    c = np.round(ijk.mean(0)).astype(int) if len(ijk) else np.array(shape) // 2
    params = {"nbSlice": {"X": shape[0], "Y": shape[1], "Z": shape[2]},
              "overlay": {"nbSlice": {"X": shape[0], "Y": shape[1], "Z": shape[2]}, "opacity": 0.8},
              "crosshair": True, "affine": aff2.tolist(), "flagCoordinates": True,
              "flagValue": False, "numSlice": {"X": int(c[0]), "Y": int(c[1]), "Z": int(c[2])},
              "radiological": False, "showLR": True, "colorCrosshair": "#7ab3ff", "smooth": True}
    return {"bg": _SPRITES[key], "overlay": png_uri(_sprite(ov)), "params": params}
