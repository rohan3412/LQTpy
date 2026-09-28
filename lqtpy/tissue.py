"""Tissue composition and location classes (cortical, periventricular, ...).

Every lesion voxel is looked up in the bundled ``tissue`` atlas (one class
per voxel: cortex, cerebral white matter, deep grey matter, ventricles,
cerebellum, brainstem, CSF; see ``tools/build_tissue_map.py``), and each
separate part of the lesion gets the location class used in MS and
small-vessel reporting:

``infratentorial``      mostly brainstem or cerebellum
``cortical``            mostly in the cortex
``deep grey matter``    mostly in the basal ganglia or thalamus
``periventricular``     reaches the lateral ventricles
``juxtacortical``       white matter reaching the cortex
``deep white matter``   white matter touching neither
``outside the brain``   at least half outside the brain (checked first;
                        usually a drawing artefact)

The classes are checked in that order, so a thalamic lesion next to the
ventricle is "deep grey matter", not "periventricular". "Reaches" means
coming within ``contact_mm`` (default 3 mm) of the structure in the MNI
template. Masks drawn on a patient's scan and warped to MNI don't line up
with the template to the millimetre, so a strict "touching" test would miss
real contact. For the same reason, reports group cortical and juxtacortical
together, as the 2017 McDonald criteria do.

For a whole lesion, :func:`composition` gives the share of each tissue
class, and :func:`locate` adds a summary class: ``cortical``,
``subcortical``, ``cortico-subcortical`` or ``infratentorial``.

All of this is core tier. The only one-time cost is two distance maps, about
1 s, which :func:`lqtpy.atlases.preload` builds up front.
"""
from __future__ import annotations

import threading

import numpy as np
from scipy import ndimage

from .prepared import prepare

__all__ = ["composition", "locate", "CLASSES", "TISSUES"]

#: tissue key -> atlas label ids
TISSUES = {
    "cortex": (1, 2),
    "white_matter": (3, 4),
    "deep_grey": (5, 6),
    "ventricles": (7, 8, 9),
    "cerebellum": (10, 11),
    "brainstem": (12,),
    "csf": (13, 14),
}
TISSUE_WORDS = {
    "cortex": "cortex", "white_matter": "white matter", "deep_grey": "deep grey matter",
    "ventricles": "ventricles", "cerebellum": "cerebellum", "brainstem": "brainstem",
    "csf": "CSF spaces", "outside": "outside the brain",
}
CLASSES = ("outside the brain", "infratentorial", "cortical", "deep grey matter", "periventricular",
           "juxtacortical", "deep white matter")

_LUT = np.zeros(256, np.uint8)  # atlas label -> index into _KEYS
_KEYS = ["outside"] + list(TISSUES)
for _i, _k in enumerate(_KEYS[1:], start=1):
    _LUT[list(TISSUES[_k])] = _i

_DIST = None
_dist_lock = threading.Lock()


def _atlas():
    from .atlases import load_atlas
    return load_atlas("tissue")


def distance_maps():
    """``(to_cortex, to_lateral_ventricle, affine)``: distance in mm from
    every voxel to the nearest cortex / lateral-ventricle voxel of the
    template, stored as uint8 in quarter-millimetres (capped at 63.75 mm).
    Built once, lazily, and cached read-only."""
    global _DIST
    if _DIST is None:
        with _dist_lock:
            if _DIST is None:
                at = _atlas()
                lab = at.label_array()
                aff = np.asarray(at.img.affine, float)
                zooms = np.linalg.norm(aff[:3, :3], axis=0)
                out = []
                for ids in ((1, 2), (7, 8)):
                    d = ndimage.distance_transform_edt(~np.isin(lab, ids), sampling=zooms)
                    q = np.minimum(np.round(d * 4), 255).astype(np.uint8)
                    q.flags.writeable = False
                    out.append(q)
                _DIST = (out[0], out[1], aff)
    return _DIST


def _sample(p, ijk_lesion: np.ndarray):
    """Tissue class index and template voxel index for lesion voxels
    (``ijk_lesion`` on the lesion's own grid). Voxels outside the template's
    field of view get class 0 ("outside") and index -1."""
    at = _atlas()
    lab = at.label_array()
    aff = np.asarray(at.img.affine, float)
    world = ijk_lesion @ p.affine[:3, :3].T + p.affine[:3, 3]
    ijk = np.floor((world - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T + 0.5).astype(np.int64)
    inside = np.all((ijk >= 0) & (ijk < np.array(lab.shape)), axis=1)
    flat = np.full(len(ijk), -1, np.int64)
    flat[inside] = np.ravel_multi_index(ijk[inside].T, lab.shape)
    cls = np.zeros(len(ijk), np.uint8)
    cls[inside] = _LUT[lab.reshape(-1)[flat[inside]]]
    return cls, flat


def _shares(cls: np.ndarray) -> dict:
    n = len(cls)
    counts = np.bincount(cls, minlength=len(_KEYS))
    return {k: (round(100.0 * int(c) / n, 1) if n else 0.0) for k, c in zip(_KEYS, counts)}


def composition(lesion, threshold: float = 0.5) -> dict:
    """Percent of the lesion in each tissue class (they sum to 100):
    ``cortex, white_matter, deep_grey, ventricles, cerebellum, brainstem,
    csf, outside``."""
    p = prepare(lesion, threshold)
    if not p.n_voxels:
        return {k: 0.0 for k in _KEYS}
    return dict(p.cached(("tissue_composition",),
                         lambda: _shares(_sample(p, p.voxel_coords().astype(float))[0])))


def _summary_class(sh: dict) -> "str | None":
    """cortical / subcortical / cortico-subcortical / infratentorial, from
    the shares of the lesion's brain tissue (CSF and outside excluded)."""
    ctx = sh["cortex"]
    sub = sh["white_matter"] + sh["deep_grey"]
    infra = sh["cerebellum"] + sh["brainstem"]
    tissue = ctx + sub + infra
    if tissue < 1:
        return None
    ctx, sub, infra = (100.0 * v / tissue for v in (ctx, sub, infra))
    if infra >= 50:
        return "infratentorial"
    if ctx >= 10 and sub >= 10:
        return "cortico-subcortical"
    return "cortical" if ctx > sub else "subcortical"


def _part_class(sh: dict, d_vent: float, d_ctx: float, contact_mm: float) -> str:
    if sh["outside"] >= 50:
        return "outside the brain"
    tissue = max(1e-9, 100.0 - sh["outside"] - sh["csf"])
    frac = {k: 100.0 * sh[k] / tissue for k in ("cortex", "deep_grey", "cerebellum", "brainstem")}
    if frac["cerebellum"] + frac["brainstem"] >= 50:
        return "infratentorial"
    if frac["cortex"] >= 50:
        return "cortical"
    if frac["deep_grey"] >= 50:
        return "deep grey matter"
    if d_vent <= contact_mm:
        return "periventricular"
    if d_ctx <= contact_mm:
        return "juxtacortical"
    return "deep white matter"


def locate(lesion, *, threshold: float = 0.5, contact_mm: float = 3.0,
           merge_gaps: bool = True) -> dict:
    """Tissue composition of the whole lesion and a location class for each
    separate part (gap-aware parts, the same ones the description numbers).

    Returns ``{"composition", "class", "contact_mm", "parts", "counts"}``;
    each part has ``id, voxels, volume_mm3, class, tissue_class`` (the
    whole-lesion style class of that part), ``composition,
    dist_ventricle_mm, dist_cortex_mm, reaches_ventricle, reaches_cortex,
    max_axial_mm`` (the largest extent across an axial plane).
    """
    p = prepare(lesion, threshold)
    return p.cached(("tissue_locate", float(contact_mm), bool(merge_gaps)),
                    lambda: _locate(p, contact_mm, merge_gaps))


def _locate(p, contact_mm, merge_gaps) -> dict:
    from . import qc as _qc

    out = {"composition": composition(p), "class": None, "contact_mm": contact_mm,
           "parts": [], "counts": {c: 0 for c in CLASSES}}
    if not p.n_voxels:
        return out
    out["class"] = _summary_class(out["composition"])
    to_ctx, to_vent, _ = distance_maps()
    cl = _qc.clusters(p, merge_gaps=merge_gaps)
    labels, off = cl["labels"], np.asarray(cl["offset"])
    vox_mm3 = float(abs(np.linalg.det(p.affine[:3, :3])))
    objs = ndimage.find_objects(labels)
    for i, sl in enumerate(objs):
        if sl is None:
            continue
        sub = labels[sl] == i + 1
        ijk = np.argwhere(sub) + off + np.array([s.start for s in sl])
        cls, flat = _sample(p, ijk.astype(float))
        sh = _shares(cls)
        ok = flat >= 0
        d_v = float(to_vent.reshape(-1)[flat[ok]].min()) / 4 if ok.any() else float("inf")
        d_c = float(to_ctx.reshape(-1)[flat[ok]].min()) / 4 if ok.any() else float("inf")
        world = ijk @ p.affine[:3, :3].T + p.affine[:3, 3]
        span = world.max(0) - world.min(0)
        # Axial extent: the larger of the x and y spans, plus one voxel.
        vs = np.linalg.norm(p.affine[:3, :3], axis=0)
        max_axial = float(max(span[0] + vs[0], span[1] + vs[1]))
        c = _part_class(sh, d_v, d_c, contact_mm)
        out["counts"][c] += 1
        out["parts"].append({
            "id": i + 1, "voxels": int(sub.sum()), "volume_mm3": round(int(sub.sum()) * vox_mm3, 1),
            "class": c, "tissue_class": _summary_class(sh), "composition": sh,
            "dist_ventricle_mm": round(d_v, 2), "dist_cortex_mm": round(d_c, 2),
            "reaches_ventricle": d_v <= contact_mm, "reaches_cortex": d_c <= contact_mm,
            "max_axial_mm": round(max_axial, 1),
        })
    return out
