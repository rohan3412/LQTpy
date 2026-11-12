"""White-matter tract overlap (structural-overlap proxy).

Intersects a lesion with the 80 binary tract masks in the HCP842 population
tractography atlas (a vendored DSI Studio distribution is one place this data
is available, but DSI Studio is only a redistribution vector -- the atlas's
own origin is Yeh et al. 2018, *NeuroImage*, CC BY-SA 4.0; see
``lqtpy/data/atlases/SOURCE.md`` for full provenance, citation, and the
required Human Connectome Project acknowledgment) -- which includes the
optic radiations, IFOF, ILF, VOF and corpus callosum -- and reports, per
tract, the fraction of the tract's voxels that the lesion touches.

This is a *structural-overlap proxy*, NOT streamline-level disconnection: a
small lesion can fully disconnect a tract while overlapping few of its voxels.
The tracts also live in DSI Studio's ICBM-2009a space, so alignment to FSL
MNI152 lesions is world-coordinate accurate to ~1-2 mm.

The tract masks are subject-invariant, so their voxel indices are loaded lazily
once and cached for reuse across every lesion/subject.
"""
from __future__ import annotations

import os

import numpy as np
import nibabel as nib

from .atlases import DATA_DIR
from .errors import DataUnavailableError, MissingDependencyError
from ._diskcache import once

# Precomputed tract voxel indices (built by tools/build_tract_index.py). Reading
# the raw 4D nifti per-volume re-decompresses the whole gzip stream (~130 s), so
# we ship this compact index instead.
_TRACT_NPZ = os.path.join(DATA_DIR, "hcp842_tracts.npz")

def tracts_available() -> bool:
    return os.path.exists(_TRACT_NPZ)


def _pretty(name: str) -> tuple[str, str]:
    """'Optic_Radiation_L' -> ('Optic Radiation (L)', 'L')."""
    hemi = ""
    if name.endswith("_L"):
        hemi, name = "L", name[:-2]
    elif name.endswith("_R"):
        hemi, name = "R", name[:-2]
    pretty = name.replace("_", " ")
    return (f"{pretty} ({hemi})" if hemi else pretty), hemi


@once
def _load_tracts():
    """Load and cache the precomputed per-tract voxel indices + a 3D ref image."""
    if not tracts_available():
        raise DataUnavailableError(
            f"White-matter tract index not found at '{_TRACT_NPZ}'. This "
            "data file should ship inside the package; reinstall lqtpy.")
    from ._diskcache import cached_arrays, file_key

    def read():
        with np.load(_TRACT_NPZ, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    npz = cached_arrays(f"hcp842_tracts_{file_key(_TRACT_NPZ)}", read)
    shape3 = tuple(int(s) for s in npz["shape"])
    ref = nib.Nifti1Image(np.zeros(shape3, dtype=np.uint8), npz["affine"])
    tracts = []
    for i, name in enumerate(npz["names"]):
        key = f"idx_{i}"
        if key not in npz:
            continue
        idx = npz[key]
        pretty, hemi = _pretty(str(name))
        tracts.append({"name": pretty, "hemi": hemi, "idx": idx, "n_vox": int(idx.size)})
    return {"ref": ref, "shape": shape3, "tracts": tracts}


@once
def _lookup():
    """Every tract voxel, sorted by its index on the tract grid: ``vox``
    (that index), ``tract`` (position in ``_load_tracts()["tracts"]``) and
    ``pos`` (position in that tract's ``idx``). A lesion finds the tract
    voxels it covers in time proportional to its own size, not the tracts'."""
    from ._diskcache import cached_arrays, file_key
    tracts = _load_tracts()["tracts"]

    def build():
        sizes = [t["idx"].size for t in tracts]
        vox = np.concatenate([t["idx"] for t in tracts]).astype(np.int64)
        order = np.argsort(vox, kind="stable")
        return {"vox": vox[order],
                "tract": np.repeat(np.arange(len(tracts), dtype=np.int16), sizes)[order],
                "pos": np.concatenate([np.arange(n, dtype=np.int32) for n in sizes])[order]}
    return cached_arrays(f"hcp842_tract_lookup_{file_key(_TRACT_NPZ)}", build)


def covered(gm) -> "list[np.ndarray]":
    """For each tract (in ``_load_tracts()["tracts"]`` order), the positions in
    its ``idx`` of the voxels the lesion covers; ``gm`` is the lesion's
    :class:`~lqtpy.prepared.GridMask` on the tract grid. The same voxels as
    ``np.flatnonzero(gm.mask.reshape(-1)[t["idx"]])``."""
    lk = _lookup()
    lo = np.searchsorted(lk["vox"], gm.flat, "left")
    n = np.searchsorted(lk["vox"], gm.flat, "right") - lo
    lo, n = lo[n > 0], n[n > 0]
    sel = np.repeat(lo - (np.cumsum(n) - n), n) + np.arange(int(n.sum()))
    tract, pos = lk["tract"][sel], lk["pos"][sel]
    order = np.argsort(tract, kind="stable")
    bounds = np.concatenate([[0], np.cumsum(np.bincount(tract, minlength=len(_load_tracts()["tracts"])))])
    pos = pos[order]
    return [pos[a:b] for a, b in zip(bounds[:-1], bounds[1:])]


_COLUMNS = ["Tract", "Hemi", "TractVoxels", "LesionVoxels",
            "PercentOverlap", "PercentOfLesion"]


def tract_overlap(lesion_img, frame: str | None = None):
    """Per-tract structural overlap of ``lesion_img`` with the HCP842 tracts.

    Returns records (``list[dict]``) by default -- keys Tract, Hemi,
    TractVoxels, LesionVoxels, PercentOverlap, PercentOfLesion. Tracts with
    zero overlap are dropped, sorted by PercentOverlap desc. Pass
    ``frame="pandas"`` for a DataFrame instead (lazy import; raises a clear
    ``ImportError`` naming the ``tables`` extra if pandas isn't installed).
    """
    from .prepared import PreparedLesion, prepare

    cache = _load_tracts()
    p = lesion_img if isinstance(lesion_img, PreparedLesion) else prepare(lesion_img, threshold=0.0)
    gm = p.on_grid(cache["ref"])
    lesion_total = gm.n_voxels

    rows = []
    for t, hit in zip(cache["tracts"], covered(gm)):
        inter = int(hit.size)
        if inter == 0:
            continue
        rows.append({
            "Tract": t["name"],
            "Hemi": t["hemi"],
            "TractVoxels": t["n_vox"],
            "LesionVoxels": inter,
            "PercentOverlap": round(100.0 * inter / t["n_vox"], 2),
            "PercentOfLesion": round(100.0 * inter / lesion_total, 2) if lesion_total else 0.0,
        })
    rows.sort(key=lambda r: r["PercentOverlap"], reverse=True)

    if frame == "pandas":
        try:
            import pandas as pd
        except ImportError as exc:
            raise MissingDependencyError(
                "tables",
                "frame='pandas' requires pandas. Install the 'tables' extra: "
                "pip install 'lqtpy[tables]'"
            ) from exc
        return pd.DataFrame(rows, columns=_COLUMNS)
    return rows
