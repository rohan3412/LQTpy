"""Border-zone ("watershed") locations between arterial territories.

Border zones lie where two arterial supplies meet, and infarcts there are
usually attributed to low flow rather than one blocked artery. Two kinds
are named in stroke reporting:

- **cortical (external) border zones**: between the cortical territories
  of the anterior and middle cerebral arteries (ACA-MCA, a frontal-parietal
  strip), the middle and posterior cerebral arteries (MCA-PCA, a
  parieto-occipital strip), and the ACA and PCA (ACA-PCA, parieto-occipital,
  near the midline).
- **internal border zone**: in the white matter of the corona radiata and
  centrum semiovale, between the deep lenticulostriate supply and the
  superficial (medullary) MCA branches. Internal border-zone infarcts often
  form a chain of small lesions along it.

The zones are derived from the bundled arterial territory atlas (Liu et al.
2023): a voxel of territory A lies in the A-B border zone when it is within
``half_width_mm`` (default 6 mm) of territory B in the same hemisphere; the
cortical zones keep cortex and the white matter within 10 mm of it, and
the internal zone the deeper white matter above the basal ganglia. Territory boundaries vary between people, so these zones
are approximate: the description says a location *fits* a border-zone
pattern, never that a lesion *is* a watershed infarct.

The zone map is built once (about 1 s) and cached.
"""
from __future__ import annotations


import numpy as np
from scipy import ndimage

from ._diskcache import once

__all__ = ["zone_map", "border_zones", "ZONES", "HALF_WIDTH_MM"]

HALF_WIDTH_MM = 6.0
#: The internal zone starts this far above the MNI origin plane (above the
#: basal ganglia, in the corona radiata / centrum semiovale).
INTERNAL_MIN_Z_MM = 8.0
#: Cortical border zones cover the cortex and the white matter within this
#: depth of it; deeper white matter belongs to the internal zone.
CORTICAL_MAX_DEPTH_MM = 10.0

#: zone id -> (hemisphere, kind, name)
ZONES = {
    1: ("left", "cortical", "ACA-MCA"), 2: ("right", "cortical", "ACA-MCA"),
    3: ("left", "cortical", "MCA-PCA"), 4: ("right", "cortical", "MCA-PCA"),
    5: ("left", "cortical", "ACA-PCA"), 6: ("right", "cortical", "ACA-PCA"),
    7: ("left", "internal", "internal"), 8: ("right", "internal", "internal"),
}

# Arterial atlas label ids (see data/arterial_labels.csv): cortical parts.
_ACA = {"L": (1,), "R": (2,)}
_MCA = {"L": (7, 9, 11, 13, 15), "R": (8, 10, 12, 14, 16)}
_PCA = {"L": (17, 19), "R": (18, 20)}
_LSA = {"L": (5,), "R": (6,)}  # lateral lenticulostriate (deep MCA)

@once
def zone_map():
    """``(zones, affine)``: uint8 border-zone ids (see :data:`ZONES`) on the
    tissue atlas grid (FSL MNI152 1 mm)."""
    from ._diskcache import cached_arrays
    from .atlases import load_atlas
    z = cached_arrays(f"border_zones_{HALF_WIDTH_MM:g}mm",
                      lambda: {"zones": _build(HALF_WIDTH_MM)[0]})["zones"]
    return z, np.asarray(load_atlas("tissue").img.affine, float)


def _build(w):
    from ._resample import resample_to_img
    from .atlases import load_atlas
    tis = load_atlas("tissue")
    art = load_atlas("arterial")
    lab = np.asarray(resample_to_img(art.img, tis.img, interpolation="nearest").dataobj).astype(np.int16)
    tl = tis.label_array()
    aff = np.asarray(tis.img.affine, float)
    zooms = np.linalg.norm(aff[:3, :3], axis=0)
    zones = np.zeros(lab.shape, np.uint8)

    def near(ids, sel):
        """Voxels within w mm of territory ``ids``, restricted to a bounding
        box around ``sel`` to keep the distance transform small."""
        src = np.isin(lab, ids)
        if not src.any() or not sel.any():
            return np.zeros_like(sel)
        idx = np.argwhere(sel)
        pad = int(np.ceil(w / zooms.min())) + 2
        lo = np.maximum(idx.min(0) - pad, 0)
        hi = np.minimum(idx.max(0) + pad + 1, lab.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        d = ndimage.distance_transform_edt(~src[sl], sampling=zooms)
        out = np.zeros_like(sel)
        out[sl] = d <= w
        return out & sel

    from .tissue import distance_maps
    to_ctx = distance_maps()[0].astype(np.float32) / 4.0
    near_cortex = to_ctx <= CORTICAL_MAX_DEPTH_MM
    # Internal zones first: where the two kinds meet, the deep white matter
    # belongs to the internal zone.
    order = sorted(ZONES, key=lambda k: ZONES[k][1] != "internal")
    for zid in order:
        hemi, kind, name = ZONES[zid]
        h = "L" if hemi == "left" else "R"
        if kind == "cortical":
            a, b = {"ACA-MCA": (_ACA, _MCA), "MCA-PCA": (_MCA, _PCA),
                    "ACA-PCA": (_ACA, _PCA)}[name]
            in_a, in_b = np.isin(lab, a[h]), np.isin(lab, b[h])
            z = (near(b[h], in_a) | near(a[h], in_b)) & near_cortex
        else:
            wm = np.isin(tl, (3, 4))
            ijk_z = np.arange(lab.shape[2]) * aff[2, 2] + aff[2, 3]
            above = (ijk_z >= INTERNAL_MIN_Z_MM)[None, None, :]
            in_c, in_d = np.isin(lab, _MCA[h]), np.isin(lab, _LSA[h])
            z = (near(_LSA[h], in_c) | near(_MCA[h], in_d)) & wm & above & ~near_cortex
        zones[z & (zones == 0)] = zid
    zones.flags.writeable = False
    return zones, aff


def _zone_label(zid: int) -> str:
    hemi, kind, name = ZONES[zid]
    return (f"{hemi} internal border zone" if kind == "internal"
            else f"{hemi} {name} cortical border zone")


def border_zones(lesion, *, threshold: float = 0.5, parts: bool = True) -> dict:
    """Share of the lesion in each border zone.

    Returns ``{"zones": [{"id", "hemisphere", "kind", "name", "label",
    "pct_of_lesion"}], "pct_in_border_zones", "fits": bool, "kind":
    "cortical" | "internal" | "mixed" | None, "n_parts_internal",
    "n_parts"}``. ``fits`` is true when at least half the lesion lies in
    border zones, or when
    three or more separate parts lie mostly in the internal zone (the
    "chain" pattern).
    """
    from .prepared import prepare
    from .atlases import load_atlas
    p = prepare(lesion, threshold)
    return p.cached(("border_zones", parts), lambda: _border_zones(p, parts, load_atlas))


def _border_zones(p, parts, load_atlas):
    zones, aff = zone_map()
    tis = load_atlas("tissue")
    gm = p.on_grid(tis.img)
    total = gm.n_voxels
    out = {"zones": [], "pct_in_border_zones": 0.0, "fits": False, "kind": None,
           "n_parts_internal": 0, "n_parts": 0}
    if not total:
        return out
    z = gm.take(zones)
    counts = np.bincount(z, minlength=len(ZONES) + 1)
    for zid in ZONES:
        if counts[zid]:
            hemi, kind, name = ZONES[zid]
            out["zones"].append({"id": zid, "hemisphere": hemi, "kind": kind, "name": name,
                                 "label": _zone_label(zid),
                                 "pct_of_lesion": round(100.0 * counts[zid] / total, 1)})
    out["zones"].sort(key=lambda d: -d["pct_of_lesion"])
    inb = 100.0 * (total - counts[0]) / total
    out["pct_in_border_zones"] = round(inb, 1)
    kinds = {d["kind"] for d in out["zones"] if d["pct_of_lesion"] >= 25}
    out["kind"] = (kinds.pop() if len(kinds) == 1 else "mixed") if kinds else None

    if parts:
        from . import qc as _qc
        cl = _qc.clusters(p)
        labels, off = cl["labels"], np.asarray(cl["offset"])
        out["n_parts"] = int(len(cl["sizes"]))
        if out["n_parts"] >= 3:
            inv = np.linalg.inv(aff)
            objs = ndimage.find_objects(labels)
            n_int = 0
            from .morphometry import part_voxels
            for i, sl in enumerate(objs):
                if sl is None:
                    continue
                ijk = part_voxels(labels, objs, i + 1, off)
                world = ijk @ p.affine[:3, :3].T + p.affine[:3, 3]
                t = np.floor(world @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(int)
                ok = np.all((t >= 0) & (t < np.array(zones.shape)), axis=1)
                zz = zones[tuple(t[ok].T)]
                if len(zz) and np.mean((zz == 7) | (zz == 8)) >= 0.5:
                    n_int += 1
            out["n_parts_internal"] = n_int
    out["fits"] = bool(inb >= 50 or out["n_parts_internal"] >= 3)
    if out["n_parts_internal"] >= 3 and out["kind"] is None:
        out["kind"] = "internal"
    return out
