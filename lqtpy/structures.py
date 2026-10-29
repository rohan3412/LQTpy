"""Key structures: distance from the lesion, and tract load at cross-sections.

Overlap alone says nothing about a lesion that stops 2 mm short of the
corticospinal tract, and voxel overlap under-rates damage to a tract that
narrows: cutting the whole internal-capsule part of the corticospinal tract
touches only a small share of its voxels. This module adds two measures:

:func:`distances`
    Shortest distance (mm) from the lesion to each key structure: the left
    and right corticospinal tracts, arcuate fasciculi and optic radiations
    (HCP842 tract masks), and the primary motor cortex, Broca's area and
    primary auditory cortex (DKT regions). 0 means the lesion reaches it.

:func:`cross_section_load`
    For each tract the lesion touches: the tract is cut into 1 mm slabs
    along its main axis; the load is the largest share of one slab's
    cross-section the lesion covers ("max cross-section load"), and the
    share covered at the tract's narrowest slab ("bottleneck load", e.g. the
    posterior limb of the internal capsule for the corticospinal tract).
    Slabs are straight cuts along the tract's principal axis, so for curved
    tracts (arcuate, cingulum) they are an approximation.

Both are core-tier and fast after a one-time build per structure (distance
maps are cached as quarter-millimetre uint8 arrays, like the tissue maps).
"""
from __future__ import annotations

import threading

import numpy as np
from scipy import ndimage

__all__ = ["distances", "cross_section_load", "KEY_STRUCTURES"]

#: name -> (source, identifiers, hemisphere)
KEY_STRUCTURES = {
    "left corticospinal tract": ("tract", ("Cortico Spinal Tract (L)",), "left"),
    "right corticospinal tract": ("tract", ("Cortico Spinal Tract (R)",), "right"),
    "left arcuate fasciculus": ("tract", ("Arcuate Fasciculus (L)",), "left"),
    "right arcuate fasciculus": ("tract", ("Arcuate Fasciculus (R)",), "right"),
    "left optic radiation": ("tract", ("Optic Radiation (L)",), "left"),
    "right optic radiation": ("tract", ("Optic Radiation (R)",), "right"),
    "left primary motor cortex": ("dk", ("left_precentral",), "left"),
    "right primary motor cortex": ("dk", ("right_precentral",), "right"),
    "left Broca's area": ("dk", ("left_parsopercularis", "left_parstriangularis"), "left"),
    "right Broca's area homologue": ("dk", ("right_parsopercularis", "right_parstriangularis"),
                                     "right"),
    "left primary auditory cortex": ("dk", ("left_transversetemporal",), "left"),
    "right primary auditory cortex": ("dk", ("right_transversetemporal",), "right"),
}

#: Distances are stored capped at this (mm); anything farther reads as ">= 60".
_DIST: dict = {}
_lock = threading.Lock()


def _structure_mask(name):
    """(mask, affine) of a key structure on its source grid."""
    src, ids, _ = KEY_STRUCTURES[name]
    if src == "tract":
        from .tracts import _load_tracts
        cache = _load_tracts()
        m = np.zeros(cache["shape"], bool).ravel()
        for t in cache["tracts"]:
            if t["name"] in ids:
                m[t["idx"]] = True
        return m.reshape(cache["shape"]), np.asarray(cache["ref"].affine, float)
    from .atlases import load_atlas
    at = load_atlas("dk")
    lab = at.label_array()
    want = [k for k, v in at.labels.items() if v.RegionName in ids]
    return np.isin(lab, want), np.asarray(at.img.affine, float)


def _distance_map(name):
    d = _DIST.get(name)
    if d is None:
        with _lock:
            d = _DIST.get(name)
            if d is None:
                from ._diskcache import cached_arrays
                m, aff = _structure_mask(name)
                zooms = np.linalg.norm(aff[:3, :3], axis=0)

                def build():
                    dist = ndimage.distance_transform_edt(~m, sampling=zooms)
                    return {"q": np.minimum(np.round(dist * 4), 255).astype(np.uint8)}
                key = "dist_" + "".join(c if c.isalnum() else "_" for c in name)
                q = cached_arrays(key, build)["q"]
                d = _DIST[name] = (q, aff, int(m.sum()))
    return d


def distances(lesion, *, threshold: float = 0.5, names=None) -> "list[dict]":
    """Shortest distance from the lesion to each key structure.

    Records: ``Structure, Hemisphere, DistanceMM`` (0 = the lesion reaches
    it; capped at 63.75), ``Reaches`` (bool). Sorted nearest first."""
    from .prepared import prepare
    p = prepare(lesion, threshold)
    names = list(names or KEY_STRUCTURES)
    return [dict(r) for r in p.cached(("key_distances", tuple(names)),
                                      lambda: _distances(p, names))]


def _distances(p, names):
    rows = []
    world = p.world_coords()
    for name in names:
        q, aff, n = _distance_map(name)
        if not n:
            continue
        inv = np.linalg.inv(aff)
        ijk = np.floor(world @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(np.int64)
        ok = np.all((ijk >= 0) & (ijk < np.array(q.shape)), axis=1)
        if not ok.any():
            continue
        dmm = float(q[tuple(ijk[ok].T)].min()) / 4.0
        rows.append({"Structure": name, "Hemisphere": KEY_STRUCTURES[name][2],
                     "DistanceMM": round(dmm, 1), "Reaches": dmm <= 0.0})
    rows.sort(key=lambda r: (r["DistanceMM"], r["Structure"]))
    return rows


# --------------------------------------------------------------------------- #
# Cross-section load
# --------------------------------------------------------------------------- #
_SLABS: dict = {}
_slab_lock = threading.Lock()
#: Slabs with fewer tract voxels than this are the tract's frayed ends,
#: not a cross-section.
MIN_SLAB_VOXELS = 15


def _tract_slabs(t, shape, affine):
    """Per tract: slab index of each tract voxel along its principal axis,
    the voxel count per slab, and the bottleneck slab."""
    key = t["name"]
    s = _SLABS.get(key)
    if s is None:
        with _slab_lock:
            s = _SLABS.get(key)
            if s is None:
                ijk = np.column_stack(np.unravel_index(t["idx"], shape)).astype(float)
                world = ijk @ affine[:3, :3].T + affine[:3, 3]
                c = world.mean(0)
                _, _, vt = np.linalg.svd(world - c, full_matrices=False)
                axis = vt[0]
                if axis[np.argmax(np.abs(axis))] < 0:
                    axis = -axis
                proj = (world - c) @ axis
                slab = np.floor(proj - proj.min()).astype(np.int32)
                counts = np.bincount(slab)
                valid = np.flatnonzero(counts >= MIN_SLAB_VOXELS)
                # The bottleneck: the narrowest slab in the middle 80% of the
                # tract's length (not its tapering ends).
                if len(valid):
                    lo, hi = np.percentile(valid, [10, 90])
                    mid = valid[(valid >= lo) & (valid <= hi)]
                    bott = int(mid[np.argmin(counts[mid])]) if len(mid) else int(valid[0])
                else:
                    bott = -1
                slab.flags.writeable = False
                s = _SLABS[key] = {"slab": slab, "counts": counts, "bottleneck": bott,
                                   "axis": axis, "centre": c, "min_proj": float(proj.min())}
    return s


def cross_section_load(lesion, *, threshold: float = 0.5, min_overlap_pct: float = 0.0) -> "list[dict]":
    """Tract damage at cross-sections, for every tract the lesion touches.

    Records: ``Tract, Hemi, PercentOverlap`` (voxel overlap, as
    :func:`lqtpy.tracts.tract_overlap`), ``MaxCrossSectionLoad`` (largest
    share of one 1 mm cross-section covered), ``BottleneckLoad`` (share of
    the narrowest cross-section covered), ``BottleneckAreaMM2``. Sorted by
    max cross-section load."""
    from .prepared import prepare
    from .tracts import _load_tracts
    p = prepare(lesion, threshold)
    cache = _load_tracts()
    return [dict(r) for r in p.cached(("cross_section", float(min_overlap_pct)),
                                      lambda: _cross(p, cache, min_overlap_pct))]


def _cross(p, cache, min_overlap_pct):
    shape, aff = cache["shape"], np.asarray(cache["ref"].affine, float)
    flat = p.on_grid(cache["ref"]).mask.reshape(-1)
    vox_mm2 = float(np.prod(np.linalg.norm(aff[:3, :3], axis=0)[:2]))
    rows = []
    for t in cache["tracts"]:
        hit = flat[t["idx"]]
        n_hit = int(np.count_nonzero(hit))
        if not n_hit:
            continue
        pct = 100.0 * n_hit / t["n_vox"]
        if pct < min_overlap_pct:
            continue
        s = _tract_slabs(t, shape, aff)
        hit_counts = np.bincount(s["slab"][hit], minlength=len(s["counts"]))
        valid = s["counts"] >= MIN_SLAB_VOXELS
        frac = np.where(valid, hit_counts / np.maximum(s["counts"], 1), 0.0)
        b = s["bottleneck"]
        rows.append({"Tract": t["name"], "Hemi": t["hemi"], "PercentOverlap": round(pct, 2),
                     "MaxCrossSectionLoad": round(100.0 * float(frac.max()), 1),
                     "BottleneckLoad": round(100.0 * float(frac[b]), 1) if b >= 0 else 0.0,
                     "BottleneckAreaMM2": round(float(s["counts"][b]) * vox_mm2, 0) if b >= 0 else 0.0})
    rows.sort(key=lambda r: -r["MaxCrossSectionLoad"])
    return rows
