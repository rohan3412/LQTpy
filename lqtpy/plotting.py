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


# --------------------------------------------------------------------------- #
# Glass-brain thumbnail, drawn from the lesion's slab only
# --------------------------------------------------------------------------- #
# nilearn projects the whole volume in all four panels. Most of it is empty,
# so the lesion's slab along y and z (x is kept whole: the l/r panels split
# at x = 0) is handed to nilearn instead, and before the figure is saved
# each panel's overlay gets back the full-size array and extent the uncropped
# call would have drawn. The PNG is byte-identical; whenever that can't be
# guaranteed (oblique grid, colour range would change, an unexpected panel)
# the volume is drawn whole, as before.

_BOUNDS_CACHE: dict = {}
_PANEL_AXES = {"y": (0, 2), "x": (1, 2), "l": (1, 2), "r": (1, 2), "z": (0, 1)}


def _slab(img, margin: int = 2):
    """``(slab image, full shape, full affine)``, or None when cropping is
    unsafe or would save little."""
    import numpy as np
    import nibabel as nib
    if not hasattr(img, "dataobj") or len(img.shape) != 3:
        return None
    aff = np.asarray(img.affine, float)
    lin = aff[:3, :3]
    if np.any(lin != np.diag(np.diag(lin))):
        return None
    # Read once and keep it on the image, as nilearn does, so the second
    # theme of the same image reuses it.
    data = getattr(img, "_data_cache", None)
    if data is None:
        data = np.asanyarray(img.dataobj)
        img._data_cache = data
    keep = data != 0
    if data.dtype.kind == "f":
        keep &= np.isfinite(data)
    ys = np.flatnonzero(keep.any(axis=(0, 2)))
    if ys.size == 0:
        return None
    zs = np.flatnonzero(keep.any(axis=(0, 1)))
    n = data.shape
    y0, y1 = max(int(ys[0]) - margin, 0), min(int(ys[-1]) + 1 + margin, n[1])
    z0, z1 = max(int(zs[0]) - margin, 0), min(int(zs[-1]) + 1 + margin, n[2])
    if (y1 - y0) * (z1 - z0) > 0.8 * n[1] * n[2]:
        return None
    sub = np.ascontiguousarray(data[:, y0:y1, z0:z1])
    fin = sub if sub.dtype.kind != "f" else np.where(np.isfinite(sub), sub, 0)
    if not (fin.min() <= 0 <= fin.max()):          # nilearn's colour range must not change
        return None
    new = aff.copy()
    new[:3, 3] = lin @ np.array([0.0, y0, z0]) + aff[:3, 3]
    return nib.Nifti1Image(sub, new), n, aff


def _reordered_bounds(shape, affine):
    """nilearn's image bounds after its own reordering, as the full-size
    call computes them (cached per grid)."""
    import numpy as np
    import nibabel as nib
    from nilearn.image import reorder_img
    from nilearn.image.resampling import get_bounds
    key = (tuple(shape), np.asarray(affine, float).tobytes())
    if key not in _BOUNDS_CACHE:
        try:
            ref = reorder_img(nib.Nifti1Image(np.zeros(shape, np.uint8), affine),
                              resample="nearest", copy_header=True)
        except TypeError:                           # nilearn without copy_header
            ref = reorder_img(nib.Nifti1Image(np.zeros(shape, np.uint8), affine),
                              resample="nearest")
        _BOUNDS_CACHE[key] = (ref.shape, np.asarray(ref.affine, float),
                              get_bounds(ref.shape, ref.affine))
    return _BOUNDS_CACHE[key]


def _restore_full_frame(disp, slab_img, shape, affine) -> bool:
    """Give each panel's overlay the full-size array and extent of the
    uncropped call. False (nothing changed) when any panel isn't as expected."""
    import numpy as np
    fshape, faff, fb = _reordered_bounds(shape, affine)
    sshape, _saff, sb = _reordered_bounds(slab_img.shape, slab_img.affine)
    swaps = []
    for d, gax in disp.axes.items():
        if d not in _PANEL_AXES:
            return False
        h, v = _PANEL_AXES[d]
        want = (sb[h][0], sb[h][1], sb[v][0], sb[v][1])
        # The overlay is the one 2-D image with the slab's extent (the dark
        # theme also draws a 2 x 2 black background in every panel).
        ims = [im for im in gax.ax.images
               if im.get_array().ndim == 2 and tuple(im.get_extent()) == want]
        if len(ims) != 1:
            return False
        im = ims[0]
        A = im.get_array()
        if A.shape != (sshape[v], sshape[h]):
            return False
        c0 = int(round((sb[h][0] - fb[h][0]) / faff[h, h]))
        r0 = int(round((fb[v][1] - sb[v][1]) / faff[v, v]))
        if c0 < 0 or r0 < 0 or c0 + A.shape[1] > fshape[h] or r0 + A.shape[0] > fshape[v]:
            return False
        F = np.ma.masked_array(np.zeros((fshape[v], fshape[h]), A.dtype),
                               mask=np.ones((fshape[v], fshape[h]), bool))
        F.data[r0:r0 + A.shape[0], c0:c0 + A.shape[1]] = np.ma.getdata(A)
        F.mask[r0:r0 + A.shape[0], c0:c0 + A.shape[1]] = np.ma.getmaskarray(A)
        # The panel's limits: nilearn widens them to the lesion's bounding
        # box (get_mask_bounds), which is measured on the image it was given.
        # Re-measure it on the full grid with nilearn's own formula.
        ob = getattr(gax, "_object_bounds", None)
        if not isinstance(ob, list) or len(ob) != 2:
            return False
        box = []
        for k, (lo, hi) in ((h, ob[1][0:2]), (v, ob[1][2:4])):
            off = int(round((sb[k][0] - fb[k][0]) / faff[k, k]))
            span_s, span_f = sb[k][1] - sb[k][0], fb[k][1] - fb[k][0]
            for val in (lo, hi):
                s = (val - sb[k][0]) * sshape[k] / span_s
                if abs(s - round(s)) > 1e-6:
                    return False
                box.append(fb[k][0] + (int(round(s)) + off) * span_f / fshape[k])
        swaps.append((im, F, (fb[h][0], fb[h][1], fb[v][0], fb[v][1]), gax, d, tuple(box)))
    for im, F, ext, gax, d, box in swaps:
        im.set_data(F)
        im.set_extent(ext)
        gax.shape = F.T.shape
        gax._object_bounds[1] = box
        gax.ax.axis(gax.get_object_bounds())
        if d == "l" and not (gax.ax.get_xlim()[0] > gax.ax.get_xlim()[1]):
            gax.ax.invert_xaxis()
    return True


def glass_brain_png(lesion_img, title=None, theme: str = "dark") -> str | None:
    import gc
    from nilearn import plotting

    def draw(img):
        return plotting.plot_glass_brain(
            img, title=title, cmap=_lesion_cmap(), colorbar=False,
            display_mode="lyrz", alpha=0.95, threshold=0.5, black_bg=theme == "dark",
        )

    def build():
        try:
            cut = _slab(lesion_img)
        except Exception:
            cut = None
        if cut is not None:
            disp = None
            try:
                disp = draw(cut[0])
                if _restore_full_frame(disp, *cut):
                    return _fig_to_uri(disp.frame_axes.figure)
            except Exception:
                pass
            import matplotlib.pyplot as plt
            if disp is not None:
                plt.close(disp.frame_axes.figure)
        return _fig_to_uri(draw(lesion_img).frame_axes.figure)
    # nilearn runs a full garbage collection five times per figure. With
    # everything that already exists frozen, those runs only look at the
    # figure's own objects; one full collection afterwards frees the closed
    # figure. The same image, with one heap-wide collection instead of five.
    gc.freeze()
    try:
        return _safe(build)
    finally:
        gc.unfreeze()
        gc.collect()
