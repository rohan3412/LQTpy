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
import threading

import numpy as np
import nibabel as nib

from .atlases import DATA_DIR
from .errors import DataUnavailableError, MissingDependencyError

# Precomputed tract voxel indices (built by tools/build_tract_index.py). Reading
# the raw 4D nifti per-volume re-decompresses the whole gzip stream (~130 s), so
# we ship this compact index instead.
_TRACT_NPZ = os.path.join(DATA_DIR, "hcp842_tracts.npz")

_TRACTS_CACHE = None
#: Guards the double-checked lazy build of _TRACTS_CACHE so concurrent first
#: callers build exactly once and all see the same object.
_tracts_lock = threading.Lock()


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


def _load_tracts():
    """Load and cache the precomputed per-tract voxel indices + a 3D ref image."""
    global _TRACTS_CACHE
    if _TRACTS_CACHE is not None:
        return _TRACTS_CACHE

    with _tracts_lock:
        if _TRACTS_CACHE is not None:
            return _TRACTS_CACHE

        if not tracts_available():
            raise DataUnavailableError(
                f"White-matter tract index not found at '{_TRACT_NPZ}'. This "
                "data file should ship inside the package; reinstall lqtpy.")

        npz = np.load(_TRACT_NPZ, allow_pickle=False)
        shape3 = tuple(int(s) for s in npz["shape"])
        affine = npz["affine"]
        names = list(npz["names"])
        ref = nib.Nifti1Image(np.zeros(shape3, dtype=np.uint8), affine)

        tracts = []
        for i, name in enumerate(names):
            key = f"idx_{i}"
            if key not in npz.files:
                continue
            idx = npz[key]
            idx.flags.writeable = False
            pretty, hemi = _pretty(str(name))
            tracts.append({"name": pretty, "hemi": hemi,
                           "idx": idx, "n_vox": int(idx.size)})

        _TRACTS_CACHE = {"ref": ref, "shape": shape3, "tracts": tracts}
        return _TRACTS_CACHE


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
    mask_flat = p.on_grid(cache["ref"]).mask.ravel()
    lesion_total = int(mask_flat.sum())

    rows = []
    for t in cache["tracts"]:
        inter = int(mask_flat[t["idx"]].sum())
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
