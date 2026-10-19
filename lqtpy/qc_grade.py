"""Grading, locating and the newer QC checks.

:func:`grade` takes the detectors' findings (from :mod:`lqtpy.qc`), adds the
newer checks (holes, rectangle-tool boxes, label values, inverted masks,
misregistration, plausibility, fragmentation, exact gaps from a drawing
tool's slice list), then for every finding measures its extent, grades it
(:data:`lqtpy.qc_model.GRADE_BANDS`), sets its display severity, and lists
where it is (``locations``) and how it could be fixed (``fix``).

:func:`assess_header` is the header-only tier: no voxel data is read.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

from .qc_model import (CATEGORY, ERROR, FATAL, INFO, WARN, DrawingContext, Finding, QCResult,
                       grade_of, severity_for, verdict_and_score)

__all__ = ["grade", "assess_header", "recognise_grid"]

_S26 = np.ones((3, 3, 3), bool)
_ORDER = {ERROR: 0, WARN: 1, INFO: 2}
_GRADE_ORDER = {"major": 0, "moderate": 1, "minor": 2, None: 3}


# --------------------------------------------------------------------------- #
# Locations
# --------------------------------------------------------------------------- #
def _names(p):
    from .qc import slice_axis_names
    return slice_axis_names(p.affine)


def _mm(p, ijk) -> "list[float]":
    v = p.affine[:3, :3] @ np.asarray(ijk, float) + p.affine[:3, 3]
    return [round(float(x), 1) for x in v]


def loc_point(p, ijk, n_voxels=None, **extra) -> dict:
    ijk = [round(float(v), 1) for v in ijk]
    d = {"kind": "point", "ijk": ijk, "mm": _mm(p, ijk)}
    if n_voxels is not None:
        d["n_voxels"] = int(n_voxels)
    d.update(extra)
    return d


def loc_slice(p, axis: int, index: int, through=None, **extra) -> dict:
    """A slice to show: ``index`` along grid ``axis``; ``ijk``/``mm`` is a
    point on that slice (through the lesion's centre by default), so a
    viewer can move its crosshair there."""
    c = list(np.asarray(through if through is not None else p.voxel_coords().mean(0), float))
    c[axis] = float(index)
    d = {"kind": "slice", "axis": int(axis), "plane": _names(p)[axis], "index": int(index),
         "ijk": [round(v, 1) for v in c], "mm": _mm(p, c)}
    d.update(extra)
    return d


def _world_to_ijk(p, mm) -> "list[float]":
    inv = np.linalg.inv(p.affine)
    v = inv[:3, :3] @ np.asarray(mm, float) + inv[:3, 3]
    return [round(float(x), 1) for x in v]


def _component_centres(p, ids) -> "dict[int, list[float]]":
    """Voxel-space centres (lesion grid) of morphometry components."""
    from .morphometry import components
    labels, lo, _ = components(p)
    if not ids:
        return {}
    c = ndimage.center_of_mass(np.ones(labels.shape), labels, index=list(ids))
    return {i: [float(v) + float(lo[k]) for k, v in enumerate(cc)] for i, cc in zip(ids, c)}


# --------------------------------------------------------------------------- #
# New detectors
# --------------------------------------------------------------------------- #
def _vox_mm3(p) -> float:
    return float(abs(np.linalg.det(p.affine[:3, :3])))


def detect_values(p, ctx=None) -> "list[Finding]":
    vi = getattr(p, "value_info", {}) or {}
    vals = vi.get("values")
    if p.was_binary or not vals:
        return []
    if vi.get("integer"):
        if ctx is not None and ctx.labels_are_colours:
            return []       # a drawing tool's pen colours, declared by the tool
        shown = ", ".join(f"{v:g}" for v in vals[:6]) + (", ..." if len(vals) > 6 else "")
        n = vi.get("n_values", len(vals))
        return [Finding("values", WARN,
                        f"The mask has {n} label values ({shown}); all of them were counted as "
                        "one lesion.",
                        {"values": vals, "n_values": n},
                        "If the labels mean different lesions or structures, split them into "
                        "separate masks.",
                        extent={"n_values": n, "volume_spread_pct": 0.0})]
    na = vi.get("n_above") or {}
    n5 = max(1, na.get("0.5", p.n_voxels))
    spread = 100.0 * (na.get("0.3", n5) - na.get("0.7", n5)) / n5
    return [Finding("values", INFO,
                    f"The mask holds fractional values ({vi.get('min', 0):.2g}-{vi.get('max', 1):.2g}), "
                    f"like a probability map; voxels above {p.threshold:g} were counted. At 0.3 or "
                    f"0.7 instead the volume would change by up to {spread / 2:.0f}%.",
                    {"threshold": p.threshold, "n_above": na},
                    "Binarise the map yourself if a different threshold is meant.",
                    extent={"volume_spread_pct": round(spread, 1)})]


def detect_inverted(p, opts) -> "list[Finding]":
    total = int(np.prod(p.shape))
    frac = p.n_voxels / max(total, 1)
    vol_cm3 = p.n_voxels * _vox_mm3(p) / 1000
    if frac >= opts["inverted_min_frac"] or vol_cm3 >= opts["inverted_min_cm3"]:
        return [Finding("inverted", ERROR,
                        f"The mask covers {100 * frac:.0f}% of the image ({vol_cm3:.0f} cm³): it "
                        "looks inverted (background marked as lesion) or is a brain mask, not a "
                        "lesion.",
                        {"image_fraction": round(frac, 3), "volume_cm3": round(vol_cm3, 1)},
                        "Invert the mask, or check that the right file was saved.",
                        extent={"image_pct": round(100 * frac, 1), "volume_cm3": round(vol_cm3, 1)},
                        affected_pct=100.0)]
    return []


def hole_voxels(sub: np.ndarray, opts) -> np.ndarray:
    """Undrawn voxels inside a lesion: what 3-D hole filling adds, plus
    pores. A porous lesion (undrawn voxels scattered through it, 15%+ of its
    core) has pores reaching its surface, so all of them count, as closing
    finds them. Otherwise only scattered pinholes deep inside count: the
    steps between the slices of a jagged slice-by-slice drawing form larger
    rings. :func:`lqtpy.qc.fix` fills exactly these voxels."""
    filled = ndimage.binary_fill_holes(sub)
    holes = filled & ~sub
    closed = ndimage.binary_closing(np.pad(sub, 2), structure=_S26, iterations=1)[2:-2, 2:-2, 2:-2]
    inner = ndimage.binary_erosion(closed | filled, structure=_S26, iterations=2)
    pores = inner & ~sub
    if pores.any():
        if 100.0 * pores.sum() / max(int(inner.sum()), 1) >= opts["porous_min_pct"]:
            pores = closed & ~sub
        else:
            pl, _ = ndimage.label(pores, structure=_S26)
            small = np.bincount(pl.ravel()) <= opts["pore_max_voxels"]
            small[0] = False
            pores = small[pl]
        holes |= pores
    return holes


def detect_holes(p, opts, mni) -> "list[Finding]":
    """Enclosed cavities inside the lesion (:func:`hole_voxels`). In MNI
    space, cavities that are CSF or ventricle in the template are left alone
    (a lesion can wrap around a ventricle)."""
    from .qc import _analysis_mask
    from .morphometry import components
    sub = _analysis_mask(p, opts)
    if not sub.any():
        return []
    holes = hole_voxels(sub, opts)
    n_h = int(holes.sum())
    if not n_h:
        return []
    _, lo, _ = components(p)
    if mni:
        from .tissue import _sample
        ijk = np.argwhere(holes) + lo
        cls, _ = _sample(p, ijk.astype(float))
        # tissue class indices: see tissue._KEYS (ventricles=4, csf=7)
        from .tissue import _KEYS
        csf = np.isin(cls, [_KEYS.index("ventricles"), _KEYS.index("csf")])
        if csf.any():
            keep = np.zeros_like(holes)
            pts = np.argwhere(holes)[~csf]
            keep[tuple(pts.T)] = True
            holes = keep
            n_h = int(holes.sum())
    if n_h < opts["holes_min_voxels"]:
        return []
    # Graded on what filling would add: the impact on every number.
    missing = 100.0 * n_h / (n_h + int(sub.sum()))
    lab, n = ndimage.label(holes, structure=_S26)
    sizes = np.bincount(lab.ravel())[1:]
    order = np.argsort(-sizes)[:10]
    cents = ndimage.center_of_mass(holes, lab, index=[int(i) + 1 for i in order])
    locs = [loc_point(p, np.asarray(c) + lo, n_voxels=sizes[i]) for i, c in zip(order, cents)]
    vox = _vox_mm3(p)
    what = (f"{n} enclosed hole{'s' if n != 1 else ''}" if n <= 20
            else "many small enclosed holes")
    return [Finding("holes", WARN,
                    f"The lesion has {what} ({n_h * vox:.0f} mm³, {missing:.0f}% of its filled "
                    "volume) that were left undrawn inside it.",
                    {"n_holes": int(n), "hole_voxels": n_h},
                    "Fill the holes in your drawing tool, or with lqtpy.qc.fix(lesion, ['holes']).",
                    extent={"missing_pct": round(missing, 1), "hole_mm3": round(n_h * vox, 1),
                            "n_holes": int(n)},
                    affected_pct=round(missing, 1), locations=locs,
                    fix={"call": "qc.fix", "ids": ["holes"]})]


def detect_boxes(p, opts) -> "list[Finding]":
    """Rectangular blocks (a rectangle/box tool): a part filling almost all
    of its bounding box. No real lesion does (a sphere fills 52%)."""
    from .morphometry import components
    labels, lo, sizes = components(p)
    vox = _vox_mm3(p)
    zooms = np.linalg.norm(p.affine[:3, :3], axis=0)
    out = []
    objs = ndimage.find_objects(labels)
    for i, sl in enumerate(objs):
        if sl is None or sizes[i] * vox < opts["box_min_mm3"]:
            continue
        dims = np.array([s.stop - s.start for s in sl])
        if np.any(dims * zooms < opts["box_min_side_mm"]):
            continue
        fill = sizes[i] / float(np.prod(dims))
        if fill >= opts["box_min_fill"]:
            centre = [(s.start + s.stop - 1) / 2 + lo[k] for k, s in enumerate(sl)]
            out.append((i + 1, fill, dims * zooms, centre, int(sizes[i])))
    if not out:
        return []
    n_box = sum(o[4] for o in out)
    pct = 100.0 * n_box / p.n_voxels
    big = out[0]
    return [Finding("box", WARN,
                    f"{'A part' if len(out) == 1 else f'{len(out)} parts'} of the mask "
                    f"{'is a' if len(out) == 1 else 'are'} rectangular block"
                    f"{'' if len(out) == 1 else 's'} ({big[2][0]:.0f} x {big[2][1]:.0f} x "
                    f"{big[2][2]:.0f} mm, {100 * big[1]:.0f}% of its box filled): it looks drawn with "
                    "a rectangle tool, not traced.",
                    {"boxes": [{"component": o[0], "fill": round(float(o[1]), 3),
                                "size_mm": [round(float(v), 1) for v in o[2]]} for o in out]},
                    "Trace the lesion's outline instead of drawing a box.",
                    extent={"affected_pct": round(pct, 1), "n_boxes": len(out),
                            "fill": round(float(big[1]), 3)},
                    affected_pct=round(pct, 1),
                    locations=[loc_point(p, o[3], n_voxels=o[4]) for o in out])]


def detect_fragmentation(p, opts) -> "list[Finding]":
    from .morphometry import components
    _, _, sizes = components(p)
    vox = _vox_mm3(p)
    n_parts = int(np.count_nonzero(sizes * vox >= opts["speck_mm3"]))
    if n_parts < opts["fragments_note"]:
        return []
    small = sizes[sizes * vox < 50]
    pct_small = 100.0 * small.sum() / p.n_voxels
    return [Finding("fragmentation", INFO,
                    f"The mask is in {n_parts} separate parts ({pct_small:.0f}% of its volume in "
                    "parts under 50 mm³). Many small lesions can be real (small-vessel disease); "
                    "if this was meant to be one lesion, the drawing broke apart.",
                    {"n_parts": n_parts},
                    "Check whether the parts are real lesions or pieces of one broken drawing.",
                    extent={"n_parts": n_parts, "small_parts_pct": round(pct_small, 1)})]


def detect_plausibility(p, opts, mni) -> "list[Finding]":
    """Suspicious but not proof: tiny, huge, only in CSF, a mirror copy."""
    from .morphometry import compute_morphometry
    out = []
    m = compute_morphometry(p)
    vol = m["volume_mm3"]
    com = m["center_of_mass_mm"]
    if vol < opts["tiny_mm3"] and m["n_components"] <= 3:
        out.append(Finding("plausibility", WARN,
                           f"The lesion is very small ({vol:.0f} mm³) for a drawn lesion: check it "
                           "isn't an accidental click or an almost empty mask.",
                           {"kind": "tiny", "volume_mm3": vol}, "",
                           grade="minor", extent={"volume_mm3": vol}))
    if vol > opts["huge_cm3"] * 1000:
        out.append(Finding("plausibility", WARN,
                           f"The lesion is very large ({vol / 1000:.0f} cm³): check it isn't a brain "
                           "or hemisphere mask saved as a lesion.",
                           {"kind": "huge", "volume_mm3": vol}, "",
                           grade="moderate", extent={"volume_cm3": round(vol / 1000, 1)}))
    if mni:
        from . import tissue
        comp = tissue.composition(p)
        csf = comp.get("ventricles", 0) + comp.get("csf", 0)
        if csf >= opts["csf_only_pct"]:
            out.append(Finding("plausibility", WARN,
                               f"{csf:.0f}% of the lesion lies in CSF or the ventricles in the MNI "
                               "template: it may be misplaced, or drawn on the wrong structure.",
                               {"kind": "csf", "csf_pct": csf}, "",
                               grade="moderate", extent={"csf_pct": csf},
                               locations=[loc_point(p, _world_to_ijk(p, com))]))
        if m["pct_left"] >= 25 and m["pct_right"] >= 25 and m["n_components"] >= 2:
            from .mirror import homologue
            h = homologue(p)
            if h["self_overlap_pct"] >= opts["mirror_min_pct"] and not _crosses_midline(p):
                out.append(Finding("plausibility", WARN,
                                   f"The left and right parts are near mirror images of each other "
                                   f"({h['self_overlap_pct']:.0f}% overlap when flipped): check the "
                                   "lesion wasn't copied to the other side.",
                                   {"kind": "mirror", "self_overlap_pct": h["self_overlap_pct"]}, "",
                                   grade="minor", extent={"mirror_overlap_pct": h["self_overlap_pct"]}))
    return out


def _crosses_midline(p) -> bool:
    """True when one connected piece spans both sides (brainstem, corpus
    callosum lesions are naturally symmetric)."""
    from .morphometry import components
    labels, lo, sizes = components(p)
    ijk = np.argwhere(labels == 1) + lo
    x = ijk @ p.affine[0, :3] + p.affine[0, 3]
    return bool((x < -2).any() and (x > 2).any())


def detect_context_gaps(p, ctx: DrawingContext) -> "Finding | None":
    """With a drawing tool's slice list: slices inside the drawn range, not
    drawn or interpolated, that are empty in the mask. Exact, and catches a
    single forgotten slice."""
    if ctx is None or not ctx.drawn_slices:
        return None
    best = None
    for axis, drawn in ctx.drawn_slices.items():
        if len(drawn) < 2:
            continue
        interp = set((ctx.interpolated_slices or {}).get(axis, []))
        moved = np.moveaxis(p.mask, axis, 0)
        lo_k, hi_k = max(0, min(drawn)), min(moved.shape[0] - 1, max(drawn))
        occ = moved[lo_k:hi_k + 1].reshape(hi_k - lo_k + 1, -1).any(1)
        empty = [lo_k + i for i, o in enumerate(occ) if not o and (lo_k + i) not in interp]
        if empty and (best is None or len(empty) > len(best[1])):
            best = (axis, empty, hi_k - lo_k + 1)
    if best is None:
        return None
    axis, empty, span = best
    plane = _names(p)[axis]
    return Finding("sparse_slices", WARN,
                   f"{len(empty)} of the {span} {plane} slices between the first and last drawn "
                   f"slice {'is' if len(empty) == 1 else 'are'} empty.",
                   {"axis": axis, "plane": plane, "gap_slices": empty, "n_gap_slices": len(empty),
                    "n_slices": span, "from_context": True},
                   "Draw the empty slices, or use your tool's interpolation.")


# --------------------------------------------------------------------------- #
# Grading
# --------------------------------------------------------------------------- #
def _sparse_extent(p, f, tier) -> None:
    ev = f.evidence
    ax = ev.get("axis", 2)
    n_sl = max(1, ev.get("n_slices", 1))
    gaps = ev.get("gap_slices", [])
    dips = ev.get("dip_slices", [])
    added = None
    if tier == "full" and p.n_voxels:
        from .qc import gap_slices_only, repair_gaps
        try:
            _, rep = repair_gaps(p, axis=ax, slices=gap_slices_only(ev))
            added = rep.get("voxels_added", 0)
        except Exception:
            added = None
        if gaps and ev.get("from_context"):
            # Interpolation can't reach a lone empty slice at a segment end;
            # fall back to the empty-slice share then.
            added = max(added or 0, int(len(gaps) * p.n_voxels / n_sl))
    if added is None:
        added = int((len(gaps) + 0.5 * len(dips)) / n_sl * p.n_voxels)
    # Nearly empty slices aren't interpolated, but they are missing volume too.
    added = max(added, int(ev.get("dip_deficit_voxels", 0)))
    missing = 100.0 * added / (p.n_voxels + added) if added else 100.0 * len(gaps) / n_sl
    f.extent = {"missing_pct": round(missing, 1), "n_gap_slices": len(gaps),
                "n_thin_slices": len(dips), "n_slices": n_sl, "plane": ev.get("plane")}
    f.affected_pct = round(missing, 1)
    order = list(dict.fromkeys(list(gaps) + list(dips)))[:20]
    f.locations = [loc_slice(p, ax, k) for k in order]
    from .qc import gap_slices_only
    only = gap_slices_only(ev)
    f.fix = {"call": "qc.repair_gaps",
             "kwargs": {"axis": int(ax), **({"slices": only} if only else {})}}


def _flat_extent(p, f) -> None:
    ev = f.evidence
    if "slab" in ev:
        pct = 100.0 * ev["slab"]["fraction"]
        ax = ev["slab"]["axis"]
        f.extent = {"flat_pct": round(pct, 1), "thickness_mm": ev["slab"]["thickness_mm"],
                    "plane": ev["slab"]["plane"]}
        c = p.voxel_coords().mean(0)
        f.locations = [loc_slice(p, ax, int(round(c[ax])))]
    else:
        comps = ev.get("components", [])
        n = sum(c["n_voxels"] for c in comps)
        pct = 100.0 * n / max(p.n_voxels, 1)
        f.extent = {"flat_pct": round(pct, 1), "n_parts": len(comps),
                    "thickness_mm": min((c["thickness_mm"] for c in comps), default=None),
                    "plane": comps[0]["plane"] if comps else None}
        f.locations = [loc_slice(p, c["axis"], int(round(c["centre_ijk"][c["axis"]])),
                                 through=c["centre_ijk"], n_voxels=c["n_voxels"]) for c in comps[:10]
                       if "centre_ijk" in c]
    f.affected_pct = round(pct, 1)


def _hollow_extent(p, f, opts) -> None:
    from .qc import _analysis_mask, _filled_volume
    ev = f.evidence
    sub = _analysis_mask(p, opts, drop_specks=False)
    names = _names(p)
    ax = next((a for a, n in names.items() if n == ev.get("plane")), 2)
    filled = _filled_volume(sub, ax)
    missing = 100.0 * max(0, filled - int(sub.sum())) / max(filled, 1)
    f.extent = {"missing_pct": round(missing, 1), "fill_gain": ev.get("fill_gain"), "plane": ev.get("plane")}
    f.affected_pct = round(missing, 1)
    c = p.voxel_coords().mean(0)
    f.locations = [loc_slice(p, ax, int(round(c[ax])))]
    f.fix = {"call": "qc.fix", "ids": ["hollow"]}


def _components_extent(p, f, comp_ids, what: str) -> None:
    from .morphometry import components
    _, _, sizes = components(p)
    vox = int(sum(int(sizes[c - 1]) for c in comp_ids))
    pct = 100.0 * vox / max(p.n_voxels, 1)
    cents = _component_centres(p, comp_ids[:20])
    f.locations = [loc_point(p, cents[c], n_voxels=sizes[c - 1]) for c in comp_ids[:20]]
    f.extent.update({"affected_pct": round(pct, 2), f"{what}_mm3": round(vox * _vox_mm3(p), 1)})
    f.affected_pct = round(pct, 2)


def _specks_ids(p, opts):
    from .morphometry import components
    from .qc import _edge_components
    _, _, sizes = components(p)
    edge = set(_edge_components(p))
    speck_vox = opts["speck_mm3"] / _vox_mm3(p)
    return [i + 1 for i, s in enumerate(sizes) if s < speck_vox and i + 1 not in edge]


def grade(p, raw: "list[Finding]", opts: dict, tier: str, ctx, mni: bool) -> "list[Finding]":
    """Measure, grade and locate every finding; add the newer checks."""
    from .qc import _edge_components
    inv = detect_inverted(p, opts)
    if inv:
        # Everything else about an inverted mask (outside the brain, touching
        # the image edge, box-shaped) is a consequence, not a separate problem.
        raw = [f for f in raw if f.id in ("header", "input")] + inv
    findings = list(raw)
    ids = {f.id for f in findings}
    ctx_gap = detect_context_gaps(p, ctx)
    if ctx_gap is not None:
        findings = [f for f in findings if f.id != "sparse_slices"] + [ctx_gap]
    findings += detect_values(p, ctx)
    # The empty slices of a gap-drawn or outline-only mask look like enclosed
    # cavities too: one problem, reported once (fixing it shows any real holes).
    if not any(f.id in ("hollow", "inverted", "sparse_slices") and f.severity != INFO
               for f in findings):
        findings += detect_holes(p, opts, mni)
    if not inv:
        findings += detect_boxes(p, opts)
        findings += detect_fragmentation(p, opts)
    if tier == "full" and not any(f.id in ("inverted", "not_mni") for f in findings):
        findings += detect_plausibility(p, opts, mni)

    out = []
    for f in findings:
        f.category = CATEGORY.get(f.id, f.category or "")
        if f.id == "sparse_slices":
            _sparse_extent(p, f, tier)
        elif f.id == "flat":
            _flat_extent(p, f)
        elif f.id == "hollow":
            _hollow_extent(p, f, opts)
        elif f.id == "edge_artifact":
            _components_extent(p, f, _edge_components(p), "edge")
            f.fix = {"call": "qc.clean", "kwargs": {"edge": True, "specks": False}}
        elif f.id == "specks":
            sid = _specks_ids(p, opts)
            f.extent = {"n_specks": len(sid)}
            _components_extent(p, f, sid, "speck")
            f.fix = {"call": "qc.clean", "kwargs": {"specks": True, "edge": False}}
        elif f.id == "strokes":
            sv = f.evidence.get("volume_mm3", 0.0)
            pct = 100.0 * sv / max(p.n_voxels * _vox_mm3(p), 1e-9)
            f.extent = {"affected_pct": round(pct, 2), "stroke_mm3": sv,
                        "largest_mm2": f.evidence["strokes"][0]["area_mm2"] if f.evidence.get("strokes") else 0}
            f.affected_pct = round(pct, 2)
            f.locations = [loc_slice(p, st["axis"], st["slice"], through=st["centre_ijk"])
                           for st in f.evidence.get("strokes", [])]
            f.fix = {"call": "qc.clean", "kwargs": {"strokes": True, "specks": False, "edge": False}}
        elif f.id == "outside_brain":
            ev = f.evidence
            pct = ev.get("outside_pct", 0.0)
            if pct >= opts["misregistered_pct"] and ev.get("attached_pct", 100.0) >= 70.0:
                f.id, f.category = "misregistered", "space"
                d = np.asarray(ev.get("direction_mm", [0, 0, 0]), float)
                ax = int(np.argmax(np.abs(d)))
                word = [("right", "left"), ("front", "back"), ("top", "bottom")][ax][0 if d[ax] > 0 else 1]
                f.message = (f"{pct:.0f}% of the lesion ({ev['outside_mm3']:.0f} mm³) lies outside "
                             f"the brain, mostly towards the {word}: the mask may be shifted relative "
                             "to MNI space (a registration problem), not just overdrawn.")
                f.suggestion = "Check the registration to MNI152 and the mask's header."
            f.extent = {"outside_pct": pct, "outside_mm3": ev.get("outside_mm3"),
                        "max_depth_mm": ev.get("max_depth_mm"), "rim_mm3": ev.get("rim_mm3"),
                        "far_mm3": ev.get("far_mm3")}
            f.affected_pct = pct
            f.locations = [loc_point(p, _world_to_ijk(p, c["mm"]), n_voxels=c["voxels"])
                           for c in ev.get("chunks", [])]
            if f.id == "outside_brain":
                f.fix = {"call": "qc.clean", "kwargs": {"outside": True, "specks": False, "edge": False}}
        elif f.id == "not_mni":
            f.extent = {"outside_pct": round(100 * f.evidence.get("outside_fraction", 0), 1)}
            f.affected_pct = 100.0
        elif f.id == "side_mismatch":
            f.extent = {"pct_left": f.evidence.get("pct_left"), "pct_right": f.evidence.get("pct_right")}
            f.affected_pct = 100.0
        elif f.id == "header":
            mm = f.evidence.get("max_disagreement_mm")
            if mm is not None:
                f.extent = {"qform_sform_mm": mm}
        # -- grade and severity ---------------------------------------------
        if f.id in FATAL:
            f.grade, f.severity = "major", ERROR
        elif f.id in ("side_mismatch", "misregistered"):
            f.grade, f.severity = "major", ERROR
        elif f.id in ("header", "input"):
            f.grade = "minor" if f.severity == WARN else None
        elif f.id == "plausibility":
            f.severity = severity_for(f.grade)
            f.severity = WARN if f.grade else INFO
        else:
            band_metric = {"strokes": "affected_pct", "edge_artifact": "affected_pct",
                           "box": "affected_pct"}.get(f.id)
            from .qc_model import GRADE_BANDS
            metric = band_metric or (GRADE_BANDS.get(f.id, (None,))[0])
            val = f.extent.get(metric) if metric else None
            g = grade_of(f.id, val)
            if f.id == "values" and f.evidence.get("n_values", 0) > 1:
                g = g or "minor"
            if f.id == "specks" and f.extent.get("affected_pct", 0) >= 5:
                g = "moderate" if g in (None, "minor") else g
            if f.id == "edge_artifact":
                g = g or "minor"
            if f.id == "outside_brain" and f.evidence.get("far_mm3"):
                g = g or "minor"        # far from the brain: stray marks, however small
            f.grade = g
            f.severity = severity_for(g)
        out.append(f)
    out.sort(key=lambda f: (_ORDER[f.severity], _GRADE_ORDER.get(f.grade, 3), -(f.affected_pct or 0)))
    return out


# --------------------------------------------------------------------------- #
# Header tier
# --------------------------------------------------------------------------- #
#: Known MNI-aligned grids: (shape, voxel mm) -> name. The affine must also
#: place the grid over the MNI brain (checked separately).
MNI_GRIDS = {
    ((182, 218, 182), 1.0): "MNI152 1 mm (FSL)",
    ((91, 109, 91), 2.0): "MNI152 2 mm (FSL)",
    ((181, 217, 181), 1.0): "MNI 1 mm (SPM/AAL)",
    ((79, 95, 79), 2.0): "MNI 2 mm (SPM)",
    ((197, 233, 189), 1.0): "MNI152 2009 1 mm",
    ((193, 229, 193), 1.0): "MNI152 2009c 1 mm",
    ((207, 256, 215), 0.74): "MNI 0.74 mm (lab grid)",
    ((99, 117, 95), 2.0): "MNI 2 mm (lab grid)",
    ((364, 436, 364), 0.5): "MNI152 0.5 mm",
}
_MNI_BRAIN_BOX = (np.array([-75.0, -110.0, -60.0]), np.array([75.0, 75.0, 85.0]))


def recognise_grid(shape, affine) -> dict:
    """Name a grid from its shape, voxel size and where it sits in world
    space. ``{"name", "mni": bool, "covers_brain": bool}``."""
    shape = tuple(int(s) for s in shape[:3])
    aff = np.asarray(affine, float)
    zooms = np.linalg.norm(aff[:3, :3], axis=0)
    vs = round(float(np.mean(zooms)), 2)
    corners = np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                        for k in (0, shape[2] - 1)]) @ aff.T
    lo, hi = corners[:, :3].min(0), corners[:, :3].max(0)
    covers = bool(np.all(lo <= _MNI_BRAIN_BOX[0] + 5) and np.all(hi >= _MNI_BRAIN_BOX[1] - 5))
    name = None
    for (shp, v), nm in MNI_GRIDS.items():
        if shp == shape and abs(v - vs) < 0.02:
            name = nm
            break
    centred = bool(np.all(np.abs((lo + hi) / 2 - np.array([0.0, -18.0, 18.0])) < 25))
    return {"name": name or "unrecognised grid", "mni": bool(name and covers),
            "covers_brain": covers, "centred_on_mni": centred, "voxel_mm": [round(float(z), 3) for z in zooms],
            "shape": list(shape)}


def assess_header(lesion) -> QCResult:
    """The header tier: grid and space, qform/sform agreement, data type,
    4-D, orientation and misnamed compression, without reading voxel data
    (except for an in-memory image). Fast enough to screen every copy of a
    lesion on a slow network share."""
    import time
    import nibabel as nib
    t0 = time.perf_counter()
    findings = []
    src = os.fspath(lesion) if isinstance(lesion, (str, os.PathLike)) else None
    if src is not None:
        try:
            img = nib.load(src)
        except Exception:
            from .formats import load_image
            img = load_image(src)
            findings.append(Finding("header", INFO,
                                    "The file's name doesn't match its compression (e.g. gzip data "
                                    "named .nii); it was read by its content.",
                                    {"misnamed_compression": True}, "Rename it to .nii.gz.",
                                    category="space"))
    elif hasattr(lesion, "header"):
        img = lesion
    else:
        from .io import as_lesion_image
        img = as_lesion_image(lesion)
    shape = img.shape
    hdr = img.header
    g = recognise_grid(shape, img.affine)
    if len(shape) == 4 and shape[3] > 1:
        findings.append(Finding("input", ERROR, f"The image is 4-D with {shape[3]} volumes, not one mask.",
                                {"shape": list(shape)}, "Save a single 3-D mask.", category="values",
                                grade="major"))
    dt = np.dtype(img.get_data_dtype())
    if dt.kind == "f":
        findings.append(Finding("values", INFO,
                                f"The data type is {dt} (floating point): values will be thresholded at 0.5.",
                                {"dtype": str(dt)}, "", category="values"))
    if not g["mni"]:
        sev = WARN if g["covers_brain"] else ERROR
        findings.append(Finding("not_mni" if sev == ERROR else "header", sev,
                                f"The grid ({shape[0]}x{shape[1]}x{shape[2]} at "
                                f"{g['voxel_mm'][0]:g} mm) isn't a recognised MNI grid"
                                + ("" if g["covers_brain"] else " and doesn't cover the MNI brain")
                                + ".", {"grid": g}, "Register the mask to MNI152 before analysis.",
                                category="space", grade="major" if sev == ERROR else "minor"))
    if hasattr(hdr, "get_sform"):
        try:
            _, sc = hdr.get_sform(coded=True)
            _, qc_ = hdr.get_qform(coded=True)
            if not sc and not qc_:
                findings.append(Finding("header", WARN, "The file has no qform or sform, so its "
                                        "orientation is a guess.", {}, "", category="space", grade="minor"))
            elif sc and qc_:
                q, s_ = hdr.get_qform(), hdr.get_sform()
                cs = np.array([[i, j, k, 1.0] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                               for k in (0, shape[2] - 1)])
                mm = float(np.abs((cs @ q.T)[:, :3] - (cs @ s_.T)[:, :3]).max())
                if mm > 2.0:
                    findings.append(Finding("header", INFO,
                                            f"The qform and sform disagree by {mm:.0f} mm; tools that "
                                            "read the qform (ITK, ANTs) will misplace the lesion.",
                                            {"max_disagreement_mm": round(mm, 1)},
                                            "Re-save so both transforms agree.", category="space",
                                            extent={"qform_sform_mm": round(mm, 1)}))
        except Exception:
            pass
    for f in findings:
        f.category = f.category or CATEGORY.get(f.id, "")
    verdict, score, worst = verdict_and_score(findings)
    return QCResult(findings=findings, verdict=verdict, score=score, worst=worst, tier="header",
                    options={"grid": g}, seconds=time.perf_counter() - t0, source=src)
