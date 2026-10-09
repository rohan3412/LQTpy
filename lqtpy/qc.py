"""Quality control for drawn lesion masks.

:func:`check` runs every check on a lesion and returns a list of
:class:`Flag` records. It never changes the mask. :func:`repair_gaps` is the
one function that does, and only when called explicitly: it returns a *new*
image plus a report of what it filled.

Checks
------
``sparse_slices``  drawn on some slices only, with the slices between left
                   empty (or, after warping to MNI, left as thin dips).
``flat``           drawn on a single slice (a thin sheet).
``hollow``         an outline drawn but never filled in.
``specks``         tiny isolated fragments (stray clicks).
``outside_brain``  a meaningful amount of lesion outside the brain. A few
                   stray voxels are ignored, and ventricles count as brain
                   (a distorted brain can push a lesion into them).
``not_mni``        the mask doesn't look like it's in MNI space.
``header``         header problems: pixdim vs affine, qform vs sform, no
                   orientation at all.
``side_mismatch``  only when the caller passes ``expected_side``: the mask
                   alone can't reveal a left/right flip.
``edge_artifact``  fragments on the outermost slice of the image, outside the
                   brain (left by drawing/export tools; seen in real data).
``strokes``        sizeable pieces only one or two voxels thick: lines or sheets
                   drawn by mistake (a stroke on the wrong plane, a crosshair),
                   which can be joined to the real lesion. ``flat`` also covers a
                   lesion that is mostly one thin slab with such pieces attached.
``input``          input notes: non-binary values thresholded, NaN/inf.

Because masks are usually MNI-space, a lesion drawn every few native slices
arrives smeared by the warp: gaps become thin dips in the per-slice area or
short empty runs, a single drawn 5 mm slice becomes a ~5 mm slab, and a
drawn outline breaks into fragments. The detectors look for those patterns.
``tests/synthetic.py`` generates exactly these artefacts (native drawings
warped to MNI with trilinear interpolation) and ``tests/test_qc.py`` checks
each detector against them, including the no-false-alarm cases.

Slice orientation is named from the *lesion grid's* axes (the axis whose
direction is closest to world x/y/z): sagittal, coronal, axial.

Gap-aware clusters
------------------
A lesion drawn with gaps falls apart into thin slabs. Counting those as
separate lesions would be wrong, so :func:`clusters` merges components that
are stacked along the gap axis, a short distance apart, with overlapping
in-plane footprints. The description uses these clusters.
"""
from __future__ import annotations

import copy
import os
import threading
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .prepared import PreparedLesion, prepare

__all__ = ["Flag", "Finding", "QCResult", "DrawingContext", "QC_VERSION", "check", "assess",
           "gap_evidence", "clusters", "repair_gaps", "clean", "brain_mask", "slice_axis_names",
           "DEFAULTS", "check_folder", "clean_folder", "problem_mask", "fix", "assess_folder",
           "rank", "assess_header", "recognise_grid", "PROBLEM_KIND", "FIXABLE"]

INFO, WARN, ERROR = "info", "warn", "error"
_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}
_S26 = np.ones((3, 3, 3), bool)

#: Tunable thresholds. Override per call: ``check(lesion, speck_mm3=5)``.
DEFAULTS = {
    "speck_mm3": 10.0,             # components smaller than this are specks
    "outside_min_mm3": 100.0,      # an outside chunk must be at least this ...
    "outside_min_frac": 0.05,      # ... and this share of the lesion ...
    "outside_min_depth_mm": 5.0,   # ... and reach this far beyond the brain (legacy)
    "outside_rim_mm": 5.0,         # overdrawing within this of the brain edge is a Note only
    "outside_note_mm3": 20.0,      # below this much outside, nothing is reported
    "not_mni_outside_frac": 0.5,   # more than this outside the brain: wrong space?
    "flat_max_mm": 7.0,            # a sheet thinner than this ...
    "flat_min_ratio": 4.0,         # ... and this many times wider: drawn on one slice
    "flat_min_voxels": 30,
    "flat_slab_max_mm": 12.0,      # or a slab up to this thick ...
    "flat_slab_min_ratio": 2.2,    # ... this many times wider ...
    "flat_min_consistency": 0.85,  # ... whose layers share one outline
    "flat_min_width_mm": 12.0,     # a flat part must be at least this wide (lacunes aren't flat)
    "flat_plate_min_share": 0.5,   # plates count only when they hold this share of the lesion: a
                                   # small second lesion seen on one thick slice is ordinary
    "sparse_min_sheets": 3,        # this many separate thin sheets in one plane: sparse slices
    "flat_plate_min_ratio": 3.0,   # a plate along a grid axis: this many times wider than thick ...
    "flat_plate_min_face": 0.35,   # ... ending in flat faces (end slices this share of the largest)
    "hollow_min_gain": 0.25,       # dilate+fill adds >= 25% volume ...
    "hollow_max_interior": 0.10,   # ... and < 10% of voxels are interior
    "gap_min_voxels": 200,         # lesions smaller than this aren't gap-tested
    "gap_dip_ratio": 0.5,          # below half of BOTH neighbours' area: a dip
    "gap_deep_ratio": 0.15,        # below this: a nearly empty slice
    "gap_min_dips": 2,
    "gap_max_mm": 12.0,            # longest empty stretch still treated as a gap ...
    "gap_long_mm": 22.0,           # ... or this long, between strongly overlapping thin slabs
    "slab_mm": 7.5,                # a "slab": this thick along one grid axis ...
    "slab_min_frac": 0.7,          # ... holding at least this share of the lesion ...
    "slab_min_span_ratio": 2.0,    # ... while the mask spans at least this many slab widths
    "stroke_min_mm2": 200.0,       # a flat patch one voxel thick, in one grid slice, this big ...
    "stroke_min_width_mm": 6.0,    # ... and solid: at least this wide at its widest point
    "stroke_line_min_mm2": 25.0,   # or a line: a thin patch at least this large ...
    "stroke_line_min_len_mm": 15.0,  # ... and this long (a stroke crossing or leaving the lesion)
    "misregistered_pct": 25.0,     # this much outside the brain: a registration problem
    "inverted_min_frac": 0.3,      # covering this share of the image ...
    "inverted_min_cm3": 1300.0,    # ... or this volume: an inverted or brain mask
    "holes_min_voxels": 3,         # enclosed holes smaller than this are ignored
    "pore_max_voxels": 8,          # undrawn pieces this small deep inside count as pinholes ...
    "porous_min_pct": 15.0,        # ... or all of them, when this much of the inside is undrawn
    "box_min_mm3": 100.0,          # a rectangular block at least this big ...
    "box_min_side_mm": 5.0,        # ... every side at least this long ...
    "box_min_fill": 0.85,          # ... filling this share of its bounding box
    "fragments_note": 50,          # this many separate parts: a Note
    "tiny_mm3": 50.0,              # plausibility: smaller than this
    "huge_cm3": 400.0,             # plausibility: larger than this
    "csf_only_pct": 70.0,          # plausibility: this much in CSF/ventricles
    "mirror_min_pct": 92.0,        # plausibility: left/right overlap when flipped (bilateral
                                   # lesions reach 70-90% by nature; a flipped copy is ~100%)
}


from .qc_model import DrawingContext, Finding, QCResult, QC_VERSION, verdict_and_score  # noqa: E402

#: Backward-compatible name: every finding is a :class:`~lqtpy.qc_model.Finding`.
Flag = Finding


# --------------------------------------------------------------------------- #
# Brain mask (atlas-derived until a dedicated MNI brain mask is bundled)
# --------------------------------------------------------------------------- #
_BRAIN = None
_brain_lock = threading.Lock()


def brain_mask():
    """``(mask, affine)``: a filled, slightly dilated MNI brain mask built
    from the bundled atlases (AAL + DKT31/CMA, excluding extra-cerebral
    CSF). Includes the ventricles. Built once (~0.6 s) and cached, read-only."""
    global _BRAIN
    if _BRAIN is not None:
        return _BRAIN
    with _brain_lock:
        if _BRAIN is not None:
            return _BRAIN
        from ._diskcache import cached_arrays
        from .atlases import load_atlas

        dk = load_atlas("dk")
        f = cached_arrays("brain_mask", lambda: {"mask": _build_brain(dk, load_atlas("aal"))})["mask"]
        _BRAIN = (f.astype(bool), np.asarray(dk.img.affine, float))
        _BRAIN[0].flags.writeable = False
        return _BRAIN


def _build_brain(dk, aal):
    from .prepared import _nearest_on_grid
    if True:
        d = dk.label_array()
        m = (d > 0) & (d != 24)  # 24 = extra-cerebral CSF
        a = aal.label_array() > 0
        nz = np.argwhere(a)
        am = _nearest_on_grid(a, np.asarray(aal.img.affine, float),
                              (tuple(nz.min(0)), tuple(nz.max(0) + 1)),
                              d.shape, np.asarray(dk.img.affine, float))
        f = ndimage.binary_fill_holes(m | am)
        for z in range(f.shape[2]):   # close sulcal CSF gaps slice by slice
            f[:, :, z] = ndimage.binary_fill_holes(f[:, :, z])
        return ndimage.binary_dilation(f, iterations=2)


_DEPTH = None


def _outside_depth():
    """Distance (mm) of every voxel outside the brain mask to the brain.
    Built once, lazily (only when some lesion extends outside)."""
    global _DEPTH
    if _DEPTH is None:
        with _brain_lock:
            if _DEPTH is None:
                bm, baff = brain_mask()
                zooms = np.linalg.norm(baff[:3, :3], axis=0)
                d = ndimage.distance_transform_edt(~bm, sampling=zooms).astype(np.float32)
                d.flags.writeable = False
                _DEPTH = d
    return _DEPTH


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
_PLANES = {0: "sagittal", 1: "coronal", 2: "axial"}


def slice_axis_names(affine) -> "dict[int, str]":
    """Map each lesion-grid axis to the slice orientation it indexes: the
    axis most aligned with world x indexes sagittal slices, y coronal, z
    axial."""
    A = np.abs(np.asarray(affine, float)[:3, :3])
    return {ax: _PLANES[int(np.argmax(A[:, ax]))] for ax in range(3)}


def _voxel_mm3(p: PreparedLesion) -> float:
    return float(abs(np.linalg.det(p.affine[:3, :3])))


def _axis_mm(p: PreparedLesion, axis: int) -> float:
    return float(np.linalg.norm(p.affine[:3, axis]))


def _edge_components(p: PreparedLesion) -> "list[int]":
    """Component ids (1-based, as in morphometry.components) that touch the
    outermost slice of the image grid on any axis AND lie mostly outside the
    brain. A real lesion in an MNI-space mask never reaches the edge of the
    image; fragments there are left behind by drawing/export tools."""
    def build():
        from .morphometry import components
        labels, lo, sizes = components(p)
        objs = ndimage.find_objects(labels)
        edge = []
        for i, sl in enumerate(objs):
            if sl is None:
                continue
            touches = any(s.start + lo[a] == 0 or s.stop + lo[a] == p.shape[a]
                          for a, s in enumerate(sl))
            if touches:
                edge.append(i + 1)
        if not edge:
            return []
        bm, baff = brain_mask()
        inv = np.linalg.inv(baff)
        out = []
        for cid in edge:
            ijk = np.argwhere(labels == cid) + lo
            world = ijk @ p.affine[:3, :3].T + p.affine[:3, 3]
            v = np.round(world @ inv[:3, :3].T + inv[:3, 3]).astype(int)
            inside = np.all((v >= 0) & (v < np.asarray(bm.shape)), axis=1)
            in_brain = np.zeros(len(v), bool)
            in_brain[inside] = bm[v[inside, 0], v[inside, 1], v[inside, 2]]
            if in_brain.mean() < 0.5:
                out.append(cid)
        return out
    return p.cached("edge_components", build)


def _analysis_mask(p: PreparedLesion, opts, *, drop_specks: bool = True) -> np.ndarray:
    """The lesion's padded-bbox mask without image-edge artefacts (and,
    by default, without specks): what the shape-based detectors look at.
    The outline detector keeps specks, since a warped outline breaks into
    speck-sized pieces."""
    from .morphometry import components
    labels, _, sizes = components(p)
    keep = np.zeros(len(sizes) + 1, bool)
    keep[1:] = (sizes * _voxel_mm3(p) >= opts["speck_mm3"]) if drop_specks else True
    keep[_edge_components(p)] = False
    return keep[labels]


_without_specks = _analysis_mask  # backward-compatible name


def _runs(flags: np.ndarray):
    """(start, stop) of each run of True in a 1-D bool array."""
    d = np.diff(np.concatenate([[0], flags.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _gap_evidence_axis(sub: np.ndarray, axis: int, max_run: int, opts, long_run: "int | None" = None) -> dict:
    """Sparse-slice evidence along one axis of ``sub`` (specks removed).

    The per-slice area profile is split into *segments* at empty stretches
    longer than ``long_run`` slices (separate lesions, not gaps). Inside a
    segment:
    - an empty run counts as a gap if the pieces either side overlap in-plane
      (the lesion continues across it) and at least one of them ends at the
      gap in a flat face (its last slice is 30%+ of its largest): two
      separate lesions, such as one in each hemisphere, taper toward the
      space between them, the slabs of a gap drawing don't. Runs longer than
      ``max_run`` (but within ``long_run``) must overlap strongly:
      thick-slice drawings on every third or fourth slice leave gaps of
      15-20 mm;
    - a *clear* gap is an empty run whose neighbouring slices are both large
      (at least 30% of the largest slice): a single forgotten slice in the
      middle of a lesion;
    - a dip is a slice below ``gap_dip_ratio`` of BOTH neighbourhoods, so a
      lesion's natural taper -- lower than one side only -- isn't a dip.
      Dips are also counted in *runs* (the neck where two blobs join is one
      run, however long), and a *deep* run goes below ``gap_deep_ratio``:
      nearly empty slices between drawn ones.
    Slice indices (on ``sub``'s axis) of gaps and dips are returned too.
    """
    long_run = max_run if long_run is None else max(long_run, max_run)
    moved = np.moveaxis(sub, axis, 0)
    prof = moved.reshape(moved.shape[0], -1).sum(1).astype(float)
    nonempty = prof > 0
    empty_runs = _runs(~nonempty)
    seg_breaks = [(a, b) for a, b in empty_runs if (b - a) > long_run]
    segments, start = [], 0
    for a, b in seg_breaks:
        if a > start:
            segments.append((start, a))
        start = b
    if start < len(prof):
        segments.append((start, len(prof)))
    n_slices = n_dips = n_gap_slices = n_gap_runs = n_clear = n_dip_runs = n_deep_runs = 0
    deficit = 0.0
    gap_idx: "list[int]" = []
    dip_idx: "list[int]" = []
    best_ratio_series = None
    peak = float(prof.max()) if prof.size else 0.0
    for s0, s1 in segments:
        seg = prof[s0:s1]
        nz = np.flatnonzero(seg)
        # At least 2 drawn slices spanning 5: a thick-slice native drawing
        # can have few non-empty slices.
        if nz.size < 2 or nz[-1] - nz[0] + 1 < 5:
            continue
        seg = seg[nz[0]:nz[-1] + 1]
        off = s0 + nz[0]
        n = seg.size
        n_slices += n
        runs = _runs(seg == 0)
        for idx, (a, b) in enumerate(runs):
            prev_start = runs[idx - 1][1] if idx > 0 else 0
            next_stop = runs[idx + 1][0] if idx + 1 < len(runs) else n
            before = moved[off + prev_start:off + a].any(axis=0)
            after = moved[off + b:off + next_stop].any(axis=0)
            bd = ndimage.binary_dilation(before, iterations=2)
            ov = np.count_nonzero(bd & after) / max(1, min(before.sum(), after.sum()))
            need = 0.3 if (b - a) <= max_run else 0.5
            pb, pa = seg[prev_start:a], seg[b:next_stop]
            # (the last two slices: a warp feathers a slab's face by a voxel)
            face = max(seg[max(a - 2, prev_start):a].max() / max(pb.max(), 1.0),
                       seg[b:min(b + 2, next_stop)].max() / max(pa.max(), 1.0))
            if ov >= need and face >= 0.3 and min(pb.max(), pa.max()) >= 0.05 * peak:
                n_gap_runs += 1
                n_gap_slices += b - a
                gap_idx += list(range(off + a, off + b))
                if min(seg[a - 1], seg[b]) >= 0.3 * peak and ov >= 0.5:
                    n_clear += 1
        w = max(3, max_run)
        left = np.array([seg[max(0, k - w):k].max() if k > 0 else 0 for k in range(n)])
        right = np.array([seg[k + 1:k + 1 + w].max() if k < n - 1 else 0 for k in range(n)])
        ref = np.minimum(left, right)
        ratio = np.where(ref > 0, seg / np.maximum(ref, 1e-9), 1.0)
        dips = (ratio < opts["gap_dip_ratio"]) & (seg > 0)
        n_dips += int(np.count_nonzero(dips))
        dip_idx += [int(off + k) for k in np.flatnonzero(dips)]
        deficit += float((ref - seg)[dips].sum())
        for a, b in _runs(dips):
            n_dip_runs += 1
            deep = ratio[a:b].min() < opts["gap_deep_ratio"]
            if deep and ref[a:b].min() >= 0.1 * peak:
                n_deep_runs += 1
                # one or two nearly empty slices inside a large lesion
                if (b - a) <= 2 and ref[a:b].min() >= 0.3 * peak:
                    n_clear += 1
        if best_ratio_series is None or n > best_ratio_series.size:
            best_ratio_series = ratio
    period, strength = None, 0.0
    if best_ratio_series is not None and best_ratio_series.size >= 8:
        x = best_ratio_series - best_ratio_series.mean()
        denom = float((x * x).sum())
        if denom > 0:
            for lag in range(2, 7):
                if lag >= x.size:
                    break
                ac = float((x[:-lag] * x[lag:]).sum() / denom)
                if ac > strength:
                    strength, period = ac, lag
    frac = (n_dips + n_gap_slices) / max(1, n_slices)
    score = frac + n_gap_runs
    return {"score": round(score, 3), "n_slices": int(n_slices), "n_dips": int(n_dips),
            "n_gap_runs": int(n_gap_runs), "n_gap_slices": int(n_gap_slices),
            "n_clear_gaps": int(n_clear), "dip_fraction": round(frac, 3),
            "n_dip_runs": int(n_dip_runs), "n_deep_runs": int(n_deep_runs),
            "dip_deficit_voxels": int(deficit),
            "period": period if strength >= 0.3 else None, "periodicity": round(strength, 3),
            "gap_slices": gap_idx, "dip_slices": dip_idx}


def gap_evidence(lesion, opts=None) -> "dict | None":
    """Strongest sparse-slice evidence over the three grid axes, or None
    when no axis shows a convincing pattern. Each axis is judged on its own
    (a lesion in pieces along one axis must be slabs *along that axis*).
    ``gap_slices``/``dip_slices`` in the result are slice indices on the
    lesion's own grid along ``axis``."""
    from .morphometry import components
    opts = {**DEFAULTS, **(opts or {})}
    p = prepare(lesion)
    if p.n_voxels < opts["gap_min_voxels"]:
        return None
    sub = _analysis_mask(p, opts)
    if not sub.any():
        return None
    _, lo, _ = components(p)
    names = slice_axis_names(p.affine)
    best = None
    for ax in range(3):
        mm = _axis_mm(p, ax)
        max_run = max(1, int(round(opts["gap_max_mm"] / mm)))
        long_run = max(max_run, int(round(opts["gap_long_mm"] / mm)))
        ev = _gap_evidence_axis(sub, ax, max_run, opts, long_run)
        span = ev["n_slices"] * mm
        if span <= opts["flat_slab_max_mm"]:
            idx = np.argwhere(sub)
            wide = min((idx[:, a].max() - idx[:, a].min() + 1) * _axis_mm(p, a)
                       for a in range(3) if a != ax)
            if wide >= opts["flat_plate_min_ratio"] * span:
                continue    # everything within one thin, wide slab: a single-slice drawing (flat)
        slabs = _slab_ratio(sub, ax)
        ev.update(axis=ax, plane=names[ax], slab_ratio=slabs)
        # Pieces along this axis that are round, separate lesions -- unless
        # the pieces are thin slabs stacked with gaps between them.
        if slabs is not None and (slabs > 0.85 or (slabs > 0.6 and not ev["n_gap_runs"])):
            continue
        gap_ok = ev["n_gap_runs"] >= 1 and slabs is not None
        # Without empty gaps, thin slices alone must form a strong, regular
        # pattern: the waist where two blobs of a real lesion join is a dip
        # too, and so are the jumps of a jagged slice-by-slice drawing.
        strong = ((gap_ok and ev["n_gap_slices"] + ev["n_dips"] >= 2)
                  or ev["n_clear_gaps"] >= 1
                  or ev["n_deep_runs"] >= 2
                  or (ev["n_dips"] >= 3 and ev["dip_fraction"] >= 0.2 and ev["periodicity"] >= 0.25
                      and ev["n_dip_runs"] >= 3)
                  or (ev["n_dips"] >= 3 and ev["dip_fraction"] >= 0.45 and ev["n_dip_runs"] >= 2))
        if strong and (best is None or ev["score"] > best["score"]):
            best = ev
    if best is not None:
        off = int(lo[best["axis"]])
        best["gap_slices"] = [int(k) + off for k in best["gap_slices"]]
        best["dip_slices"] = [int(k) + off for k in best["dip_slices"]]
    return best


def _slab_ratio(sub: np.ndarray, axis: int) -> "float | None":
    """Size-weighted mean over pieces of (extent along ``axis``) / (smaller
    in-plane extent), in voxels: ~0.2-0.4 for the slabs of a gapped drawing,
    ~1 for round lesions. ``None`` for a single piece."""
    lab, n = ndimage.label(sub, structure=_S26)
    if n < 2:
        return None
    sizes = np.bincount(lab.ravel())[1:]
    ratios = []
    for sl in ndimage.find_objects(lab):
        ext = [s.stop - s.start for s in sl]
        inplane = min(e for a, e in enumerate(ext) if a != axis)
        ratios.append(ext[axis] / max(inplane, 1))
    return round(float(np.average(ratios, weights=sizes)), 3)


def _outline_consistency(c: np.ndarray, evecs: np.ndarray) -> float:
    """How alike a thin part's 1 mm layers are across its thin axis: the
    layers 20% of the thickness in from each face are compared with the
    middle layer (both their area ratio and their overlap, IoU); the lowest
    of the four numbers is returned. A single thick slice drawn once and
    warped keeps one outline through its thickness (about 0.9-1); a real
    thin lesion tapers toward its faces (0.55-0.75 on synthetic ellipsoids
    6-10 mm thick)."""
    proj = c @ evecs                      # minor, mid, major
    t = proj[:, 0]
    uv = np.round(proj[:, 1:]).astype(int)
    tmin, tmax = float(t.min()), float(t.max())
    rng = tmax - tmin
    if rng < 2.0:
        return 1.0

    def layer(centre):
        sel = (t >= centre - 0.5) & (t < centre + 0.5)
        return {tuple(x) for x in uv[sel]}

    mid = layer(0.0)
    if not mid:
        return 0.0
    vals = []
    for centre in (tmin + 0.2 * rng, tmax - 0.2 * rng):
        a = layer(centre)
        vals.append(len(a) / len(mid))
        vals.append(len(a & mid) / max(1, len(a | mid)))
    return float(min(vals))


def _flat_components(p: PreparedLesion, labels, lo, sizes, opts):
    """Components that are thin sheets across their minor principal axis:
    either very thin and wide (``flat_max_mm``, ``flat_min_ratio``), or a
    thicker slab whose layers all share one outline (a single thick slice
    drawn once and warped; ``flat_slab_max_mm``, ``flat_slab_min_ratio``,
    ``flat_min_consistency``)."""
    out = []
    names = slice_axis_names(p.affine)
    edge = set(_edge_components(p))
    zooms = np.linalg.norm(p.affine[:3, :3], axis=0)
    for i, n in enumerate(sizes):
        if n < opts["flat_min_voxels"]:
            break
        if i + 1 in edge:
            continue
        ijk = np.argwhere(labels == i + 1) + lo
        world = ijk @ p.affine[:3, :3].T
        c = world - world.mean(0)
        evals, evecs = np.linalg.eigh(np.cov(c.T))
        proj = c @ evecs  # columns: minor .. major
        ext = proj.max(0) - proj.min(0)
        vox = float(np.abs(np.linalg.inv(p.affine[:3, :3]) @ evecs[:, 0]).max()) ** -1
        thick = ext[0] + vox
        wide = min(ext[1], ext[2]) + vox
        thin = thick <= opts["flat_max_mm"] and wide >= opts["flat_min_ratio"] * thick
        cons, plate = None, False
        if not thin and thick <= opts["flat_slab_max_mm"] and wide >= opts["flat_slab_min_ratio"] * thick:
            cons = _outline_consistency(c, evecs)
            thin = cons >= opts["flat_min_consistency"]
        gridvec = np.linalg.inv(p.affine[:3, :3]) @ evecs[:, 0]
        ax = int(np.argmax(np.abs(gridvec)))
        if not thin and thick <= opts["flat_slab_max_mm"]:
            # A plate cut along a grid axis: wide, and ending abruptly in two
            # flat faces instead of tapering (one thick slice, drawn once).
            g = ijk - ijk.min(0)
            gext = (g.max(0) + 1) * zooms
            others = [a for a in range(3) if a != ax]
            if (gext[ax] <= opts["flat_slab_max_mm"]
                    and min(gext[others]) >= opts["flat_plate_min_ratio"] * gext[ax]):
                area = np.bincount(g[:, ax])
                if min(area[0], area[-1]) >= opts["flat_plate_min_face"] * area.max():
                    thin, thick, wide, plate = True, float(gext[ax]), float(min(gext[others])), True
        if thin and wide < opts["flat_min_width_mm"]:
            thin = False
        if thin:
            centre = ijk.mean(0)
            out.append({"component": i + 1, "thickness_mm": round(float(thick), 1),
                        "width_mm": round(float(wide), 1), "plane": names[ax], "axis": ax,
                        "n_voxels": int(n), "consistency": None if cons is None else round(cons, 3),
                        "plate": plate,
                        "centre_ijk": [round(float(v), 1) for v in centre]})
    plates = sum(c["n_voxels"] for c in out if c["plate"])
    if plates < opts["flat_plate_min_share"] * max(int(np.sum(sizes)), 1):
        out = [c for c in out if not c["plate"]]
    return out


_ACROSS = {"sagittal": "from left to right", "coronal": "from front to back",
           "axial": "from top to bottom"}


def _slab(sub: np.ndarray, zooms, opts) -> "dict | None":
    """Most of the mask in one thin slab along a grid axis, though the mask
    spans much further: a lesion drawn on one slice with strokes reaching
    into others. (A pure single-slice sheet is caught by ``flat``; this finds
    it when other pieces are attached, which defeats the per-component
    thickness test.)"""
    total = int(sub.sum())
    best = None
    if not total:
        return None
    for ax in range(3):
        prof = sub.sum(axis=tuple(a for a in range(3) if a != ax)).astype(float)
        w = max(1, int(round(opts["slab_mm"] / zooms[ax])))
        if len(prof) < w:
            continue
        win = np.convolve(prof, np.ones(w), "valid")
        k = int(np.argmax(win))
        frac = float(win[k]) / total if total else 0.0
        nz = np.flatnonzero(prof)
        span = float((nz[-1] - nz[0] + 1) * zooms[ax])
        sl = [slice(None)] * 3
        sl[ax] = slice(k, k + w)
        slab = sub[tuple(sl)]
        others = [a for a in range(3) if a != ax]
        idx = np.argwhere(slab.any(axis=ax))
        width = float(min((idx[:, i].max() - idx[:, i].min() + 1) * zooms[others[i]]
                          for i in range(2))) if len(idx) else 0.0
        if (frac >= opts["slab_min_frac"] and span >= opts["slab_min_span_ratio"] * w * zooms[ax]
                and width >= 4 * w * zooms[ax]):
            if best is None or frac > best["fraction"]:
                best = {"axis": ax, "fraction": round(frac, 3), "thickness_mm": round(w * zooms[ax], 1),
                        "span_mm": round(span, 1), "width_mm": round(width, 1)}
    return best


def _drawing_axis(sub: np.ndarray, zooms, max_width_mm: float = 6.0) -> "int | None":
    """The grid axis a mask was most likely drawn on, slice by slice: the one
    whose slices leave the most *narrow* one-voxel rim area (where one drawn
    slice sticks out past both neighbours), or None when there are no rims.
    Wide solid sheets don't vote: they are what the stroke check looks for,
    and would otherwise hide themselves by making their own plane look like
    the drawing plane.

    A few long strokes can outvote the drawing plane the same way. So when
    the winning axis has nearly all its rim area on a few slices, and
    another axis has rims on a good share of its slices (a clear
    slice-by-slice drawing), that other axis is the drawing axis. A
    thick-slice drawing also has its rims on few slices, but then no other
    axis shows a drawing, and it keeps its plane."""
    s8 = np.ones((3, 3), bool)
    area, top, spread = [], [], []
    for a in range(3):
        mm = np.moveaxis(sub, a, 0)
        thin = mm.copy()
        thin[1:] &= ~mm[:-1]
        thin[:-1] &= ~mm[1:]
        others = [zooms[i] for i in range(3) if i != a]
        # A patch is narrow where one 2-D erosion by half the width limit
        # removes it entirely.
        r = max(1, int(round(max_width_mm / 2 / min(others))))
        solid = np.zeros_like(thin)
        for k in np.flatnonzero(thin.reshape(thin.shape[0], -1).any(1)):
            core = ndimage.binary_erosion(thin[k], structure=s8, iterations=r)
            if core.any():
                solid[k] = ndimage.binary_dilation(core, structure=s8, iterations=r) & thin[k]
        rim = (thin & ~solid).reshape(thin.shape[0], -1).sum(1).astype(float)
        span = np.flatnonzero(mm.reshape(mm.shape[0], -1).any(1))
        n = int(span[-1] - span[0] + 1) if span.size else 1
        total = float(rim.sum())
        k = max(3, int(round(0.15 * n)))
        area.append(total * others[0] * others[1])
        top.append(float(np.sort(rim)[::-1][:k].sum()) / total if total else 0.0)
        spread.append(float(np.count_nonzero(rim >= 2)) / n)
    if max(area) <= 0:
        return None
    best = int(np.argmax(area))
    if top[best] >= 0.8:
        drawn = [a for a in range(3) if a != best and spread[a] >= 0.4]
        if drawn:
            best = max(drawn, key=lambda a: area[a])
    return best


def _stroke_voxels(sub: np.ndarray, zooms, opts, skip_axes=()) -> "tuple[np.ndarray, list[dict]]":
    """Drawn strokes: solid flat patches exactly one voxel thick along a grid
    axis (no lesion on either side along that axis), lying in one grid
    slice, at least ``stroke_min_mm2`` in area and ``stroke_min_width_mm``
    wide. A drawing tool leaves these when a stroke lands on the wrong plane.
    A stroke that crosses a lesion shows only where it sticks out of it, as a
    long narrow strip: a one-voxel patch at least ``stroke_line_min_len_mm``
    long (and ``stroke_line_min_mm2`` in area) counts too. The rims of a
    slice-by-slice drawing don't line up into strips that long off the
    drawing plane.

    Axes in ``skip_axes`` are the planes a lesion was *drawn* in when a
    drawing problem was found there (some slices only, a single slice): its
    drawn slices are one-voxel sheets too, but they are that problem, not
    strokes. On 1,335 real masks, one-voxel patches in the drawing plane are
    common (slice-by-slice drawings leave rims where one slice sticks out),
    but they are narrow; the width test keeps them out. Returns (mask of
    stroke voxels, patches largest first)."""
    out = np.zeros_like(sub)
    patches = []
    s8 = np.ones((3, 3), bool)
    names = ("sagittal", "coronal", "axial")
    for a in range(3):
        if a in skip_axes:
            continue
        mm = np.moveaxis(sub, a, 0)
        up = np.zeros_like(mm)
        up[:-1] = mm[1:]
        dn = np.zeros_like(mm)
        dn[1:] = mm[:-1]
        thin = mm & ~up & ~dn
        others = [zooms[i] for i in range(3) if i != a]
        px = float(others[0] * others[1])
        min_px = opts["stroke_min_mm2"] / px
        line_px = opts["stroke_line_min_mm2"] / px
        om = np.moveaxis(out, a, 0)
        for k in np.flatnonzero(thin.reshape(thin.shape[0], -1).sum(1) >= min(min_px, line_px)):
            lab, n = ndimage.label(thin[k], structure=s8)
            sizes = np.bincount(lab.ravel())[1:]
            for i in np.flatnonzero(sizes >= min(min_px, line_px)):
                sel = lab == i + 1
                ij = np.argwhere(sel)
                lo, hi = ij.min(0), ij.max(0) + 1
                ext = (hi - lo) * np.array(others)
                box = np.pad(sel[lo[0]:hi[0], lo[1]:hi[1]], 1)
                width = 2.0 * float(ndimage.distance_transform_edt(box, sampling=others).max())
                sheet = sizes[i] >= min_px and width >= opts["stroke_min_width_mm"]
                line = float(ext.max()) >= opts["stroke_line_min_len_mm"]
                if not (sheet or line):
                    continue  # a narrow rim, not a drawn stroke
                om[k] |= sel
                ctr_sub = [0.0, 0.0, 0.0]
                oth = [i for i in range(3) if i != a]
                ctr_sub[a] = float(k)
                ctr_sub[oth[0]], ctr_sub[oth[1]] = (float(v) for v in ij.mean(0))
                patches.append({"axis": a, "plane": names[a], "slice": int(k), "centre_sub": ctr_sub,
                                "area_mm2": round(float(sizes[i]) * px, 1),
                                "width_mm": round(width, 1), "kind": "sheet" if sheet else "line",
                                "extent_mm": [round(float(e), 1) for e in ext]})
    return out, sorted(patches, key=lambda d: -d["area_mm2"])


def _hollow(sub: np.ndarray):
    """(gain, interior_fraction, axis) for the outline-only signature.

    A drawn outline has almost no interior voxels (nothing survives one
    erosion), and once its fragments are joined by a small dilation, filling
    the space it encloses slice by slice adds a lot. A filled lesion has a
    large interior; a thin sheet or tube encloses nothing. After an MNI warp
    an outline is broken into dots, so the dilation is tried at 2 and 3
    voxels (validated on warped synthetic outlines and clean ribbons in
    tests/qc_zoo.py).
    """
    n = int(sub.sum())
    if n == 0:
        return 0.0, 1.0, None
    interior_frac = float(ndimage.binary_erosion(sub).sum()) / n
    if interior_frac >= 0.2:          # clearly solid: skip the expensive part
        return 0.0, interior_frac, None
    best, best_ax = 0.0, None
    for it in (2, 3):
        d = ndimage.binary_dilation(np.pad(sub, it + 1), iterations=it)
        base = int(d.sum())
        for ax in range(3):
            filled = np.zeros_like(d)
            for k in range(d.shape[ax]):
                sl = [slice(None)] * 3
                sl[ax] = k
                filled[tuple(sl)] = ndimage.binary_fill_holes(d[tuple(sl)])
            gain = (int(filled.sum()) - base) / base
            if gain > best:
                best, best_ax = gain, ax
    return best, interior_frac, best_ax


def _filled_volume(sub: np.ndarray, axis: int) -> int:
    """Voxels an outline encloses (close, fill slice by slice along
    ``axis``, then undo the closing), for the hollow finding's extent."""
    best = int(sub.sum())
    for it in (2, 3):
        d = ndimage.binary_dilation(np.pad(sub, it + 1), iterations=it)
        f = np.zeros_like(d)
        for k in range(d.shape[axis]):
            sl = [slice(None)] * 3
            sl[axis] = k
            f[tuple(sl)] = ndimage.binary_fill_holes(d[tuple(sl)])
        f = ndimage.binary_erosion(f, iterations=it) | np.pad(sub, it + 1)
        best = max(best, int(f.sum()))
    return best


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #
def check(lesion, *, threshold: float = 0.5, expected_side: "str | None" = None,
          mni: bool = True, context=None, tier: str = "full", **options) -> "list[Finding]":
    """Run every QC check; returns graded findings, worst first.

    ``lesion``: anything :func:`lqtpy.prepare` accepts. ``expected_side``:
    "left"/"right" if the caller knows the clinical side. ``mni=False``
    skips the brain-mask checks (for masks not meant to be in MNI space).
    ``context``: a :class:`~lqtpy.qc_model.DrawingContext` (or its dict/JSON
    form) from a drawing tool. ``tier``: ``"fast"`` or ``"full"``.
    ``options`` override :data:`DEFAULTS`. :func:`assess` returns the same
    findings plus the mask's verdict and score.
    """
    return assess(lesion, threshold=threshold, expected_side=expected_side, mni=mni,
                  context=context, tier=tier, **options).findings


def assess(lesion, *, threshold: float = 0.5, expected_side: "str | None" = None,
           mni: bool = True, context=None, tier: str = "full", **options) -> QCResult:
    """Graded QC for one mask: findings (worst first) with category, grade,
    extent, locations and fix hints, plus the verdict and a 0-100 score.
    See :mod:`lqtpy.qc_model` and ``docs/QC.md``.

    ``tier="header"`` reads only the file header (see
    :func:`lqtpy.qc_grade.assess_header`); ``"fast"`` skips the slower
    plausibility checks and estimates gap extent without interpolating;
    ``"full"`` (default) runs everything."""
    import time as _time
    from . import qc_grade
    from .qc_model import verdict_and_score
    if tier == "header":
        return qc_grade.assess_header(lesion)
    if tier not in ("fast", "full"):
        raise ValueError(f"unknown QC tier {tier!r}; choose header, fast or full")
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown QC option(s): {sorted(unknown)}")
    opts = {**DEFAULTS, **options}
    ctx = DrawingContext.from_any(context)
    side = expected_side or (ctx.expected_side if ctx else None)
    t0 = _time.perf_counter()
    p = prepare(lesion, threshold)
    ckey = ("qc2", side, mni, tier, tuple(sorted(options.items())),
            repr(ctx.to_dict()) if ctx else None)
    findings = p.cached(ckey, lambda: qc_grade.grade(p, _check(p, side, mni, opts, ctx), opts,
                                                     tier, ctx, mni))
    findings = [copy.deepcopy(f) for f in findings]
    verdict, score, worst = verdict_and_score(findings)
    src = p.source if isinstance(p.source, (str, os.PathLike)) else None
    return QCResult(findings=findings, verdict=verdict, score=score, worst=worst, tier=tier,
                    context=ctx.to_dict() if ctx else {},
                    options={"threshold": threshold, "mni": mni, "expected_side": side, **options},
                    seconds=_time.perf_counter() - t0, source=os.fspath(src) if src else None)


def _check(p: PreparedLesion, expected_side, mni, opts, ctx=None) -> "list[Flag]":
    """The detectors. Findings come out ungraded; :mod:`lqtpy.qc_grade`
    measures, grades and locates them."""
    from .morphometry import components, compute_morphometry

    flags: "list[Flag]" = []
    vox_mm3 = _voxel_mm3(p)
    names = slice_axis_names(p.affine)
    morph = compute_morphometry(p)

    # -- input notes (non-binary values: see the "values" check) ------------- #
    if p.n_nonfinite:
        flags.append(Flag("input", WARN,
                          f"{p.n_nonfinite} voxels were NaN or infinite and were treated as 0.",
                          {"n_nonfinite": p.n_nonfinite},
                          "Check how the mask was exported."))

    # -- header ----------------------------------------------------------- #
    hi = p.header_info or {}
    if morph.get("header_zooms_mismatch"):
        flags.append(Flag("header", WARN,
                          "The header's voxel size disagrees with the affine; volumes use the affine.",
                          {"affine_voxel_mm": morph["voxel_size_mm"]},
                          "Re-save the mask so the header and affine agree."))
    if hi.get("no_orientation"):
        flags.append(Flag("header", WARN,
                          "The file has no qform or sform, so its orientation is a guess.",
                          {}, "Re-save the mask from the tool that registered it."))
    if hi.get("qform_sform_mm", 0) > 1.0:
        # Escalated to a warning below if the sform placement looks wrong.
        flags.append(Flag("header", INFO,
                          f"The file's qform and sform disagree by {hi['qform_sform_mm']:.0f} mm; "
                          "lqtpy used the sform. Viewers that use the qform will show the lesion "
                          "elsewhere.",
                          {"max_disagreement_mm": hi["qform_sform_mm"]},
                          "Re-save the mask so both transforms agree."))

    labels, lo, sizes = components(p)
    sub = _analysis_mask(p, opts, drop_specks=False)

    # -- image-edge artefacts -------------------------------------------------- #
    edge = _edge_components(p)
    if edge:
        ev = int(sum(int(sizes[c - 1]) for c in edge))
        flags.append(Flag("edge_artifact", WARN,
                          f"{len(edge)} fragment{'s' if len(edge) != 1 else ''} "
                          f"({ev * vox_mm3:.0f} mm³) lie on the outermost slice of the image, "
                          "outside the brain; they look like leftovers from the drawing or "
                          "export tool. They are counted in the lesion volume.",
                          {"n_fragments": len(edge), "voxels": ev,
                           "volume_mm3": round(ev * vox_mm3, 1)},
                          "Remove them in your drawing tool, or use lqtpy.qc.clean()."))

    # -- hollow (before specks: a broken outline looks like many specks) --- #
    gain, interior, hax = _hollow(sub)
    hollow = gain >= opts["hollow_min_gain"] and interior < opts["hollow_max_interior"]
    if hollow:
        where = f" in the {names[hax]} slices" if hax is not None else ""
        flags.append(Flag("hollow", WARN,
                          f"The lesion looks like an outline that wasn't filled in{where}; its "
                          "volume is badly underestimated.",
                          {"fill_gain": round(gain, 2), "interior_fraction": round(interior, 3),
                           "plane": names.get(hax)},
                          "Fill the outline in your drawing tool."))

    # -- specks ----------------------------------------------------------- #
    speck_vox = opts["speck_mm3"] / vox_mm3
    edge_set = set(edge)
    specks = [int(s) for i, s in enumerate(sizes) if s < speck_vox and i + 1 not in edge_set]
    if specks and not hollow:
        if len(specks) < len(sizes):
            flags.append(Flag("specks", INFO if sum(specks) * vox_mm3 < 20 else WARN,
                              f"{len(specks)} tiny fragment{'s' if len(specks) != 1 else ''} "
                              f"(under {opts['speck_mm3']:g} mm³ each) look like stray marks.",
                              {"n_specks": len(specks), "speck_voxels": sum(specks)},
                              "Remove them if they weren't meant to be lesion."))
        else:
            flags.append(Flag("specks", WARN, "The whole lesion is only a few tiny fragments.",
                              {"n_specks": len(specks), "speck_voxels": sum(specks)},
                              "Check the mask; it may be almost empty or thresholded too high."))

    # -- sparse slices (gaps) --------------------------------------------- #
    gap = gap_evidence(p, opts)
    if gap is not None:
        detail = []
        if gap["n_gap_runs"]:
            detail.append(f"{gap['n_gap_runs']} empty gap{'s' if gap['n_gap_runs'] != 1 else ''}")
        if gap["n_dips"]:
            detail.append(f"{gap['n_dips']} much thinner slice{'s' if gap['n_dips'] != 1 else ''}")
        flags.append(Flag("sparse_slices", WARN,
                          f"The lesion looks drawn on some {gap['plane']} slices only "
                          f"({' and '.join(detail)} across {gap['n_slices']} slices"
                          + (f", repeating every {gap['period']}" if gap.get("period") else "")
                          + "). Its volume and extent are probably underestimated.",
                          gap, "Draw on every slice, or interpolate with lqtpy.qc.repair_gaps()."))

    zooms = np.linalg.norm(p.affine[:3, :3], axis=0)

    # -- flat / single slice (not when it's the slabs of a gap drawing) ---- #
    if gap is None and not hollow:
        slab = _slab(sub, zooms, opts)
        if slab is not None and not any(f.id == "flat" for f in flags):
            slab["plane"] = names[slab["axis"]]
            flags.append(Flag("flat", WARN,
                              f"{100 * slab['fraction']:.0f}% of the lesion lies in one {slab['plane']} "
                              f"slab {slab['thickness_mm']:g} mm thick, although the mask reaches "
                              f"{slab['span_mm']:.0f} mm {_ACROSS.get(slab['plane'], 'across it')}: it looks "
                              f"drawn on a single "
                              f"{slab['plane']} slice (or a few), with pieces reaching into other slices.",
                              {"slab": slab},
                              "Draw the lesion on every slice it spans."))
    if gap is None and not hollow and not any(f.id == "flat" for f in flags):
        flat = _flat_components(p, labels, lo, sizes, opts)
        planes = {}
        for c in flat:
            planes.setdefault(c["axis"], []).append(c)
        stack_ax = max(planes, key=lambda a: len(planes[a])) if planes else None
        if stack_ax is not None and len(planes[stack_ax]) >= opts["sparse_min_sheets"]:
            # Several separate thin sheets in one plane are the drawn slices of
            # a gap drawing whose gaps the profile test couldn't prove.
            sheets = planes[stack_ax]
            ks = sorted(c["centre_ijk"][stack_ax] for c in sheets)
            ev = {"axis": int(stack_ax), "plane": names[stack_ax], "n_sheets": len(sheets),
                  "n_slices": int(round(ks[-1] - ks[0])) + 1, "n_gap_runs": 0, "n_gap_slices": 0,
                  "n_dips": 0, "gap_slices": [], "dip_slices": [int(round(k)) for k in ks],
                  "components": sheets}
            flags.append(Flag("sparse_slices", WARN,
                              f"{len(sheets)} parts of the lesion are separate thin sheets in the "
                              f"{names[stack_ax]} plane: it looks drawn on some slices only. Its "
                              "volume and extent are probably underestimated.",
                              ev, "Draw on every slice, or interpolate with lqtpy.qc.repair_gaps()."))
            gap = ev
        elif flat:
            big = flat[0]
            if big["component"] == 1:
                msg = (f"The lesion is a thin sheet ({big['thickness_mm']} mm thick, "
                       f"{big['width_mm']} mm across) in the {big['plane']} plane; it looks "
                       "drawn on a single slice.")
            else:
                msg = (f"{len(flat)} part{'s' if len(flat) != 1 else ''} of the lesion "
                       f"{'are thin sheets' if len(flat) != 1 else 'is a thin sheet'} in the "
                       f"{big['plane']} plane, as if drawn on a single slice.")
            flags.append(Flag("flat", WARN, msg, {"components": flat},
                              "Draw the lesion on every slice it spans."))

    # -- strokes: thin lines/sheets drawn by mistake -------------------- #
    # Planes where a drawing problem was found: their one-voxel slices are
    # that problem (reported above), not strokes.
    drawn = set()
    if ctx is not None and ctx.drawing_axis is not None:
        drawn.add(ctx.drawing_axis)
    if ctx is not None and ctx.drawn_slices:
        drawn |= set(ctx.drawn_slices)
    if gap is not None:
        drawn.add(gap["axis"])
    for f in flags:
        if f.id == "flat":
            if "slab" in f.evidence:
                drawn.add(f.evidence["slab"]["axis"])
            for c in f.evidence.get("components", []):
                drawn.add({v: k for k, v in names.items()}.get(c["plane"], -1))
    if not drawn and sub.any():
        # No drawing problem named the plane: infer it. Its slices' rims are
        # part of an ordinary slice-by-slice drawing.
        dax = _drawing_axis(sub, zooms)
        if dax is not None:
            drawn.add(dax)
    svox, strokes = ((np.zeros_like(sub), []) if hollow
                     else _stroke_voxels(sub, zooms, opts, skip_axes=drawn))
    for st in strokes:
        st["centre_ijk"] = [round(float(v + lo[i]), 1) for i, v in enumerate(st.pop("centre_sub"))]
        st["slice"] = int(st["slice"] + lo[st["axis"]])
    if strokes:
        sv = float(svox.sum()) * vox_mm3
        pl = sorted({st["plane"] for st in strokes})
        planes = [", ".join(pl[:-1]), pl[-1]] if len(pl) > 1 else pl
        flags.append(Flag("strokes", WARN,
                          f"{len(strokes)} flat sheet{'s' if len(strokes) != 1 else ''} one voxel thick "
                          f"lie{'s' if len(strokes) == 1 else ''} in single {' and '.join(planes)} "
                          f"slice{'s' if len(strokes) != 1 else ''} ({sv / 1000:.1f} cm³, "
                          f"{100 * sv / (p.n_voxels * vox_mm3):.0f}% of the mask, largest "
                          f"{strokes[0]['area_mm2']:.0f} mm²): they look like strokes drawn by "
                          "mistake, for example on the wrong plane, not lesion.",
                          {"strokes": strokes[:10], "volume_mm3": round(sv, 1)},
                          "Erase the strokes in your drawing tool, or remove them with "
                          "lqtpy.qc.clean(lesion, strokes=True)."))

    # -- brain-mask checks --------------------------------------------------- #
    if mni:
        com = morph["center_of_mass_mm"]
        in_box = -95 <= com[0] <= 95 and -130 <= com[1] <= 95 and -80 <= com[2] <= 110
        bm, baff = brain_mask()
        # Image-edge fragments are their own finding: measure what lies
        # outside the brain without them, so one artefact isn't reported twice.
        src = p
        if edge and len(edge) < len(sizes):
            import nibabel as nib
            rest = p.mask.copy()
            box = tuple(slice(int(lo[a]), int(lo[a]) + labels.shape[a]) for a in range(3))
            rest[box] &= ~np.isin(labels, edge)
            if rest.any():
                src = prepare(nib.Nifti1Image(rest.astype(np.uint8), p.affine), 0.5, name=p.name)
        on = src.on_grid((bm.shape, baff))
        n_on = on.n_voxels
        outside = on.mask & ~bm
        n_out = int(outside.sum())
        total = n_on + on.outside_fov_voxels
        out_frac = (n_out + on.outside_fov_voxels) / total if total else 0.0
        if not in_box or out_frac > opts["not_mni_outside_frac"]:
            reason = (f"its centre {tuple(com)} mm is outside the MNI brain box" if not in_box
                      else f"{100 * out_frac:.0f}% of it lies outside the brain")
            flags.append(Flag("not_mni", ERROR,
                              f"The mask probably isn't in MNI space: {reason}.",
                              {"outside_fraction": round(out_frac, 3), "center_of_mass_mm": com},
                              "Register the mask to MNI152 before analysis."))
            for f in flags:
                if f.id == "header" and "qform" in f.message:
                    f.severity = WARN
                    f.message += " The sform placement looks wrong here; check the qform."
        elif n_out:
            lab, nlab = ndimage.label(outside, structure=_S26)
            if nlab:
                depth_map = _outside_depth()
                chunk = np.bincount(lab.ravel())[1:]
                depth = np.asarray(ndimage.maximum(depth_map, lab, index=np.arange(1, nlab + 1)))
                bvox = float(abs(np.linalg.det(baff[:3, :3])))
                # Overdrawing that hugs the brain edge within ``outside_rim_mm``
                # is tolerated; everything beyond it is measured.
                beyond = outside & (depth_map > opts["outside_rim_mm"])
                n_beyond = int(beyond.sum())
                rim_pct = 100.0 * (n_out - n_beyond) / max(n_on, 1)
                if n_beyond * bvox >= opts["outside_note_mm3"] or rim_pct >= 1.0:
                    order = np.argsort(-chunk)
                    idx = [int(i) + 1 for i in order[:10]]
                    cents = ndimage.center_of_mass(outside, lab, index=idx)
                    chunks = [{"voxels": int(chunk[i - 1]), "mm3": round(float(chunk[i - 1] * bvox), 1),
                               "depth_mm": round(float(depth[i - 1]), 1),
                               "mm": [round(float(v), 1) for v in (baff[:3, :3] @ np.asarray(c) + baff[:3, 3])]}
                              for i, c in zip(idx, cents)]
                    # How much of the outside part is joined to lesion inside the
                    # brain (a shifted lesion) rather than detached (a stray blob).
                    whole, _ = ndimage.label(on.mask, structure=_S26)
                    inside_ids = np.unique(whole[on.mask & bm])
                    attached = np.isin(whole[beyond], inside_ids[inside_ids > 0])
                    attached_pct = 100.0 * attached.mean() if attached.size else 0.0
                    # Direction: from the lesion's centre towards what lies outside.
                    oc = np.argwhere(beyond if n_beyond else outside).mean(0)
                    lc = np.argwhere(on.mask).mean(0)
                    shift = baff[:3, :3] @ (oc - lc)
                    msg = (f"{n_beyond * bvox:.0f} mm³ of the lesion "
                           f"({100 * n_beyond / max(n_on, 1):.0f}%) lies outside the brain."
                           if n_beyond * bvox >= opts["outside_note_mm3"] else
                           f"The lesion is overdrawn past the brain edge by up to "
                           f"{float(depth.max()):.0f} mm ({(n_out - n_beyond) * bvox:.0f} mm³, "
                           f"{rim_pct:.0f}% of the lesion).")
                    flags.append(Flag("outside_brain", WARN, msg,
                                      {"outside_mm3": round(float(n_beyond * bvox), 1),
                                       "outside_pct": round(100.0 * n_beyond / max(n_on, 1), 2),
                                       "rim_mm3": round(float((n_out - n_beyond) * bvox), 1),
                                       "n_chunks": int(nlab), "chunks": chunks,
                                       "max_depth_mm": round(float(depth.max()), 1),
                                       "direction_mm": [round(float(v), 1) for v in shift],
                                       "attached_pct": round(float(attached_pct), 1),
                                       "rim_pct": round(float(rim_pct), 2)},
                                      "Check the mask edges against the brain."))

    # -- expected side ------------------------------------------------------ #
    if expected_side:
        side = expected_side.strip().lower()[:1]
        pl, pr = morph["pct_left"], morph["pct_right"]
        if side in ("l", "r"):
            wrong = pr if side == "l" else pl
            if wrong > 50:
                flags.append(Flag("side_mismatch", ERROR,
                                  f"Expected a {expected_side} lesion, but {wrong:.0f}% of it is on the "
                                  f"{'right' if side == 'l' else 'left'}.",
                                  {"pct_left": pl, "pct_right": pr},
                                  "Check the mask orientation (possible left/right flip)."))

    flags.sort(key=lambda f: _SEVERITY_ORDER[f.severity])
    return flags


# --------------------------------------------------------------------------- #
# Gap-aware clusters
# --------------------------------------------------------------------------- #
def clusters(lesion, *, merge_gaps: bool = True) -> dict:
    """Lesion clusters: connected components, with gap-drawn slabs merged.

    Returns ``{"labels", "offset", "sizes", "merged_from", "gap_plane"}``:
    ``labels`` is an int32 array over the lesion's padded bounding box
    (1 = largest cluster), ``merged_from[i]`` lists the raw component ids
    joined into cluster ``i + 1``. Merging only happens when
    :func:`gap_evidence` finds a sparse-slice pattern and ``merge_gaps`` is
    true; then components within ``gap_max_mm`` of each other along the gap
    axis whose in-plane footprints overlap are joined.
    """
    p = prepare(lesion)
    return p.cached(("clusters", merge_gaps), lambda: _clusters(p, merge_gaps))


def _clusters(p, merge_gaps):
    from .morphometry import components

    labels, lo, sizes = components(p)
    n = len(sizes)
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    gap = gap_evidence(p) if merge_gaps and n > 1 else None
    if gap is not None:
        ax = gap["axis"]
        max_run = int(round(DEFAULTS["gap_long_mm"] / _axis_mm(p, ax)))
        objs = ndimage.find_objects(labels)
        foot = [None] * n
        for i in range(n):
            if objs[i] is not None:
                foot[i] = (labels[objs[i]] == i + 1).any(axis=ax)
        for i in range(n):
            for j in range(i + 1, n):
                si, sj = objs[i], objs[j]
                if si is None or sj is None:
                    continue
                sep = max(sj[ax].start - si[ax].stop, si[ax].start - sj[ax].stop)
                if sep > max_run:
                    continue
                # Footprint overlap in the plane, in bbox coordinates.
                others = [a for a in range(3) if a != ax]
                fi = np.zeros([labels.shape[a] for a in others], bool)
                fj = np.zeros_like(fi)
                fi[si[others[0]], si[others[1]]] = foot[i]
                fj[sj[others[0]], sj[others[1]]] = foot[j]
                if np.any(ndimage.binary_dilation(fi, iterations=2) & fj):
                    parent[find(i + 1)] = find(j + 1)
    roots: "dict[int, list]" = {}
    for c in range(1, n + 1):
        roots.setdefault(find(c), []).append(c)
    groups = sorted(roots.values(), key=lambda g: -sum(int(sizes[c - 1]) for c in g))
    remap = np.zeros(n + 1, dtype=np.int32)
    for new_id, g in enumerate(groups, start=1):
        for c in g:
            remap[c] = new_id
    new_labels = remap[labels]
    new_labels.flags.writeable = False
    new_sizes = np.array([sum(int(sizes[c - 1]) for c in g) for g in groups], dtype=np.int64)
    return {"labels": new_labels, "offset": lo, "sizes": new_sizes,
            "merged_from": groups, "gap_plane": None if gap is None else gap["plane"]}


# --------------------------------------------------------------------------- #
# repair_gaps()
# --------------------------------------------------------------------------- #
def _sdt(mask2d: np.ndarray) -> np.ndarray:
    """Signed distance: negative inside, positive outside."""
    if not mask2d.any():
        return np.full(mask2d.shape, np.inf)
    inside = ndimage.distance_transform_edt(mask2d)
    outside = ndimage.distance_transform_edt(~mask2d)
    return outside - inside


def repair_gaps(lesion, *, axis: "int | None" = None, max_gap_mm: "float | None" = None,
                threshold: float = 0.5):
    """Fill sparse-slice gaps by shape-based interpolation. Opt-in only.

    Along the gap axis (from :func:`gap_evidence`, or ``axis`` = 0/1/2 on
    the lesion grid), the core slice of each drawn slab (a local maximum of
    the per-slice area) is a key slice; the slices between two neighbouring
    cores are filled by interpolating the cores' signed distance maps
    (shape-based interpolation). Existing voxels are always kept (the
    result is a superset of the input). The lesion's outermost tips beyond
    the first and last drawn slices are unknown and left alone.

    Returns ``(repaired_img, report)``; ``report`` says which plane was
    used, how many slices were interpolated and how much was added. When no
    gaps are found it returns an unchanged copy and says so.
    """
    import nibabel as nib

    p = prepare(lesion, threshold)
    names = slice_axis_names(p.affine)
    if axis is None:
        ev = gap_evidence(p)
        if ev is None:
            return (nib.Nifti1Image(p.mask.astype(np.uint8), p.affine),
                    {"repaired": False, "reason": "no sparse-slice pattern found",
                     "voxels_added": 0, "volume_added_mm3": 0.0})
        axis = ev["axis"]
    max_gap_mm = DEFAULTS["gap_long_mm"] if max_gap_mm is None else max_gap_mm
    mask = np.moveaxis(p.mask.copy(), axis, 0)
    prof = mask.reshape(mask.shape[0], -1).sum(1).astype(float)
    # Key slices = the core of each drawn slab: local maxima of the area
    # profile. After an MNI warp a drawn slice is a slab whose edges taper,
    # so interpolating between the tapered edges would add almost nothing;
    # interpolating between cores rebuilds the missing volume.
    n = prof.size
    keys = [k for k in range(n) if prof[k] > 0
            and prof[k] >= (prof[k - 1] if k > 0 else 0)
            and prof[k] > (prof[k + 1] if k < n - 1 else 0)]
    # Cores can be at most two gaps plus a slab apart.
    max_span = max(2, int(round(2 * max_gap_mm / _axis_mm(p, axis))))
    out = mask.copy()
    filled_slices = 0
    for a, b in zip(keys[:-1], keys[1:]):
        g = b - a - 1
        if g < 1 or g > max_span:
            continue
        da, db = _sdt(mask[a]), _sdt(mask[b])
        for k in range(1, g + 1):
            t = k / (g + 1)
            new = ((1 - t) * da + t * db) <= 0
            filled_slices += int(np.any(new & ~out[a + k]))
            out[a + k] |= new
    out = np.moveaxis(out, 0, axis)
    added = int(out.sum()) - p.n_voxels
    img = nib.Nifti1Image(out.astype(np.uint8), p.affine)
    return img, {"repaired": added > 0, "axis": int(axis), "plane": names[axis],
                 "slices_interpolated": filled_slices, "voxels_added": added,
                 "volume_added_mm3": round(added * _voxel_mm3(p), 1)}


# --------------------------------------------------------------------------- #
# clean()
# --------------------------------------------------------------------------- #
def clean(lesion, *, specks: bool = True, edge: bool = True, strokes: bool = False,
          outside: bool = False, threshold: float = 0.5, **options):
    """Remove image-edge artefacts, specks, (``strokes=True``) thin strokes
    and (``outside=True``) voxels beyond the brain edge (past the
    ``outside_rim_mm`` tolerance, MNI masks only). Opt-in only.

    Returns ``(clean_img, report)`` with a *new* binary image; the input is
    never modified. ``report`` lists what was removed.
    """
    import nibabel as nib
    from .morphometry import components

    opts = {**DEFAULTS, **options}
    p = prepare(lesion, threshold)
    labels, lo, sizes = components(p)
    drop = set()
    if edge:
        drop |= set(_edge_components(p))
    n_specks = 0
    if specks:
        speck_vox = opts["speck_mm3"] / _voxel_mm3(p)
        for i, n in enumerate(sizes):
            if n < speck_vox and i + 1 not in drop and len(sizes) > 1:
                drop.add(i + 1)
                n_specks += 1
    out = p.mask.copy()
    if drop:
        rm = np.isin(labels, sorted(drop))
        sl = tuple(slice(lo[a], lo[a] + labels.shape[a]) for a in range(3))
        out[sl] &= ~rm
    n_strokes = 0
    if strokes:
        # Repeat: removing a one-voxel sheet can expose the next layer of a
        # stroke that was two voxels thick in places.
        zooms = np.linalg.norm(p.affine[:3, :3], axis=0)
        found0 = next((f for f in check(p) if f.id == "strokes"), None)
        keep_axes = set(range(3)) - {st["axis"] for st in (found0.evidence["strokes"] if found0 else [])}
        for _ in range(3 if found0 else 0):
            svox, found = _stroke_voxels(out, zooms, opts, skip_axes=keep_axes)
            if not found:
                break
            n_strokes += len(found)
            out &= ~svox
    n_outside = 0
    if outside:
        from .qc_tools import _outside_voxels
        ov = _outside_voxels(p, opts) & out
        n_outside = int(ov.sum())
        out &= ~ov
    removed = p.n_voxels - int(out.sum())
    return (nib.Nifti1Image(out.astype(np.uint8), p.affine),
            {"removed_components": len(drop), "removed_specks": n_specks,
             "removed_strokes": n_strokes, "removed_outside_voxels": n_outside,
             "removed_edge_fragments": len(drop) - n_specks, "voxels_removed": removed,
             "volume_removed_mm3": round(removed * _voxel_mm3(p), 1)})


# --------------------------------------------------------------------------- #
# Folders
# --------------------------------------------------------------------------- #
def check_folder(folder, *, threshold: float = 0.5, recursive: bool = False,
                 expected_sides: "dict | None" = None) -> "list[dict]":
    """Run :func:`check` on every lesion file in ``folder``.

    One record per file: ``file`` (path), ``name``, ``worst`` (ok / info /
    warn / error), ``flags`` (list of flag dicts), ``volume_mm3`` and
    ``error`` (a message when the file couldn't be read, else None),
    ``verdict`` and ``score`` (see :class:`QCResult`).
    ``expected_sides`` maps names to "left"/"right" for the flip check."""
    import os
    from .formats import is_lesion_file, strip_extension
    from .morphometry import compute_morphometry
    files = []
    if recursive:
        for root, _, fs in os.walk(folder):
            files += [os.path.join(root, f) for f in fs if is_lesion_file(f)]
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if is_lesion_file(f)]
    rows = []
    for path in sorted(files):
        name = strip_extension(os.path.basename(path))
        row = {"file": path, "name": name, "worst": "ok", "flags": [], "volume_mm3": None,
               "error": None, "verdict": None, "score": None}
        try:
            p = prepare(path, threshold)
            fl = check(p, expected_side=(expected_sides or {}).get(name))
            row["flags"] = [f.to_dict() for f in fl]
            row["volume_mm3"] = compute_morphometry(p)["volume_mm3"]
            row["verdict"], row["score"], _ = verdict_and_score(fl)
            sev = {f.severity for f in fl}
            row["worst"] = "error" if ERROR in sev else WARN if WARN in sev else (
                INFO if INFO in sev else "ok")
        except Exception as exc:  # one unreadable file must not stop the folder
            row["error"] = str(exc).splitlines()[0]
            row["worst"] = "error"
        rows.append(row)
    return rows


def clean_folder(src, dst, *, specks: bool = True, edge: bool = True, threshold: float = 0.5,
                 only_flagged: bool = True) -> "list[dict]":
    """Write cleaned copies (:func:`clean`) of every lesion file in ``src``
    into ``dst``. Never overwrites: ``dst`` must differ from ``src``, and a
    file that already exists in ``dst`` is skipped. With ``only_flagged``
    (default), files with nothing to remove are not copied.

    Returns one record per file: ``file``, ``out`` (written path or None),
    ``status`` (written / nothing to remove / exists / failed) and the
    clean report."""
    import os
    import nibabel as nib
    from .formats import is_lesion_file, strip_extension
    if os.path.abspath(src) == os.path.abspath(dst):
        raise ValueError("The output folder must differ from the input folder; lqtpy never "
                         "overwrites masks.")
    os.makedirs(dst, exist_ok=True)
    out = []
    for f in sorted(os.listdir(src)):
        if not is_lesion_file(f):
            continue
        path = os.path.join(src, f)
        target = os.path.join(dst, strip_extension(f) + ".nii.gz")
        rec = {"file": path, "out": None, "status": "", "report": None}
        if os.path.exists(target):
            rec["status"] = "exists"
            out.append(rec)
            continue
        try:
            img, rep = clean(path, specks=specks, edge=edge, threshold=threshold)
            rec["report"] = rep
            if rep["voxels_removed"] or not only_flagged:
                nib.save(img, target)
                rec["out"], rec["status"] = target, "written"
            else:
                rec["status"] = "nothing to remove"
        except Exception as exc:
            rec["status"] = f"failed: {str(exc).splitlines()[0]}"
        out.append(rec)
    return out


# Helpers for tools built on lqtpy (see lqtpy.qc_tools and lqtpy.qc_grade).
from .qc_tools import (FIXABLE, PROBLEM_KIND, assess_folder, fix, problem_mask,  # noqa: E402
                       rank)
from .qc_grade import assess_header, recognise_grid  # noqa: E402
