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

import threading
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import ndimage

from .prepared import PreparedLesion, prepare

__all__ = ["Flag", "check", "gap_evidence", "clusters", "repair_gaps", "clean", "brain_mask",
           "slice_axis_names", "DEFAULTS", "check_folder", "clean_folder"]

INFO, WARN, ERROR = "info", "warn", "error"
_SEVERITY_ORDER = {ERROR: 0, WARN: 1, INFO: 2}
_S26 = np.ones((3, 3, 3), bool)

#: Tunable thresholds. Override per call: ``check(lesion, speck_mm3=5)``.
DEFAULTS = {
    "speck_mm3": 10.0,             # components smaller than this are specks
    "outside_min_mm3": 100.0,      # an outside chunk must be at least this ...
    "outside_min_frac": 0.05,      # ... and this share of the lesion ...
    "outside_min_depth_mm": 5.0,   # ... and reach this far beyond the brain
    "not_mni_outside_frac": 0.5,   # more than this outside the brain: wrong space?
    "flat_max_mm": 7.0,            # a sheet thinner than this ...
    "flat_min_ratio": 4.0,         # ... and this many times wider: drawn on one slice
    "flat_min_voxels": 30,
    "hollow_min_gain": 0.25,       # dilate+fill adds >= 25% volume ...
    "hollow_max_interior": 0.10,   # ... and < 10% of voxels are interior
    "gap_min_voxels": 200,         # lesions smaller than this aren't gap-tested
    "gap_dip_ratio": 0.5,          # below half of BOTH neighbours' area: a dip
    "gap_min_dips": 2,
    "gap_max_mm": 12.0,            # longest empty stretch still treated as a gap
}


@dataclass
class Flag:
    """One QC finding."""

    id: str
    severity: str           # "info" | "warn" | "error"
    message: str            # one plain sentence for reports
    evidence: dict = field(default_factory=dict)
    suggestion: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


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


def _gap_evidence_axis(sub: np.ndarray, axis: int, max_run: int, opts) -> dict:
    """Sparse-slice evidence along one axis of ``sub`` (specks removed).

    The per-slice area profile is split into *segments* at empty stretches
    longer than ``max_run`` slices (those are separate lesions, not gaps).
    Inside a segment:
    - an empty run counts as a gap only if the slices either side of it
      overlap in-plane (the lesion continues across it);
    - a dip is a slice below ``gap_dip_ratio`` of BOTH neighbourhoods (up
      to ``max_run`` slices each side), so a lesion's natural taper -- lower
      than one side only -- isn't a dip.
    """
    moved = np.moveaxis(sub, axis, 0)
    prof = moved.reshape(moved.shape[0], -1).sum(1).astype(float)
    nonempty = prof > 0
    empty_runs = _runs(~nonempty)
    # Segment boundaries at long empty stretches (and at the ends).
    seg_breaks = [(a, b) for a, b in empty_runs if (b - a) > max_run]
    segments, start = [], 0
    for a, b in seg_breaks:
        if a > start:
            segments.append((start, a))
        start = b
    if start < len(prof):
        segments.append((start, len(prof)))
    n_slices = n_dips = n_gap_slices = n_gap_runs = 0
    best_ratio_series = None
    for s0, s1 in segments:
        seg = prof[s0:s1]
        nz = np.flatnonzero(seg)
        # At least 3 drawn slices spanning 5: a thick-slice native drawing
        # can have few non-empty slices.
        if nz.size < 3 or nz[-1] - nz[0] + 1 < 5:
            continue
        seg = seg[nz[0]:nz[-1] + 1]
        off = s0 + nz[0]
        n = seg.size
        n_slices += n
        # Empty runs inside the segment that the lesion continues across:
        # the in-plane footprints of the pieces either side must overlap.
        runs = _runs(seg == 0)
        for idx, (a, b) in enumerate(runs):
            prev_start = runs[idx - 1][1] if idx > 0 else 0
            next_stop = runs[idx + 1][0] if idx + 1 < len(runs) else n
            before = moved[off + prev_start:off + a].any(axis=0)
            after = moved[off + b:off + next_stop].any(axis=0)
            bd = ndimage.binary_dilation(before, iterations=2)
            ov = np.count_nonzero(bd & after) / max(1, min(before.sum(), after.sum()))
            if ov >= 0.3:
                n_gap_runs += 1
                n_gap_slices += b - a
        # Dips against both sides.
        w = max(3, max_run)
        left = np.array([seg[max(0, k - w):k].max() if k > 0 else 0 for k in range(n)])
        right = np.array([seg[k + 1:k + 1 + w].max() if k < n - 1 else 0 for k in range(n)])
        ref = np.minimum(left, right)
        ratio = np.where(ref > 0, seg / np.maximum(ref, 1e-9), 1.0)
        dips = (ratio < opts["gap_dip_ratio"]) & (seg > 0)
        n_dips += int(np.count_nonzero(dips))
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
            "dip_fraction": round(frac, 3),
            "period": period if strength >= 0.3 else None, "periodicity": round(strength, 3)}


def gap_evidence(lesion, opts=None) -> "dict | None":
    """Strongest sparse-slice evidence over the three grid axes, or None
    when no axis shows a convincing pattern. Each axis is judged on its own
    (a lesion in pieces along one axis must be slabs *along that axis*)."""
    opts = {**DEFAULTS, **(opts or {})}
    p = prepare(lesion)
    if p.n_voxels < opts["gap_min_voxels"]:
        return None
    sub = _analysis_mask(p, opts)
    if not sub.any():
        return None
    names = slice_axis_names(p.affine)
    best = None
    for ax in range(3):
        max_run = max(1, int(round(opts["gap_max_mm"] / _axis_mm(p, ax))))
        ev = _gap_evidence_axis(sub, ax, max_run, opts)
        slabs = _slab_ratio(sub, ax)
        ev.update(axis=ax, plane=names[ax], slab_ratio=slabs)
        if slabs is not None and slabs > 0.6:
            continue  # pieces along this axis are separate round lesions
        gap_ok = ev["n_gap_runs"] >= 1 and slabs is not None
        # Without empty gaps, thin slices alone must form a strong pattern:
        # the waist where two blobs of a real lesion join is a dip too.
        strong = ((gap_ok and ev["n_gap_slices"] + ev["n_dips"] >= 2)
                  or (ev["n_dips"] >= 3 and ev["dip_fraction"] >= 0.2)
                  or (ev["n_dips"] >= 3 and ev["periodicity"] >= 0.4
                      and ev["dip_fraction"] >= 0.1))
        if strong and (best is None or ev["score"] > best["score"]):
            best = ev
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


def _flat_components(p: PreparedLesion, labels, lo, sizes, opts):
    """Components that are thin sheets across their minor principal axis."""
    out = []
    names = slice_axis_names(p.affine)
    edge = set(_edge_components(p))
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
        if thick <= opts["flat_max_mm"] and wide >= opts["flat_min_ratio"] * thick:
            gridvec = np.linalg.inv(p.affine[:3, :3]) @ evecs[:, 0]
            ax = int(np.argmax(np.abs(gridvec)))
            out.append({"component": i + 1, "thickness_mm": round(float(thick), 1),
                        "width_mm": round(float(wide), 1), "plane": names[ax],
                        "n_voxels": int(n)})
    return out


def _hollow(sub: np.ndarray):
    """(gain, interior_fraction, axis) for the outline-only signature.

    A drawn outline has almost no interior voxels (nothing survives one
    erosion), and once its fragments are joined by a 2-voxel dilation,
    filling the space it encloses slice by slice adds a lot. A filled lesion
    has a large interior; a thin sheet or tube encloses nothing. After an
    MNI warp an outline is broken into dots, which is why the dilation is
    needed (validated on warped synthetic outlines in tests/test_qc.py).
    """
    n = int(sub.sum())
    if n == 0:
        return 0.0, 1.0, None
    interior_frac = float(ndimage.binary_erosion(sub).sum()) / n
    if interior_frac >= 0.2:          # clearly solid: skip the expensive part
        return 0.0, interior_frac, None
    d = ndimage.binary_dilation(np.pad(sub, 3), iterations=2)
    base = int(d.sum())
    best, best_ax = 0.0, None
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


# --------------------------------------------------------------------------- #
# check()
# --------------------------------------------------------------------------- #
def check(lesion, *, threshold: float = 0.5, expected_side: "str | None" = None,
          mni: bool = True, **options) -> "list[Flag]":
    """Run every QC check; returns flags sorted by severity (errors first).

    ``lesion``: anything :func:`lqtpy.prepare` accepts. ``expected_side``:
    "left"/"right" if the caller knows the clinical side. ``mni=False``
    skips the brain-mask checks (for masks not meant to be in MNI space).
    ``options`` override :data:`DEFAULTS`.
    """
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown QC option(s): {sorted(unknown)}")
    opts = {**DEFAULTS, **options}
    p = prepare(lesion, threshold)
    key = ("qc", expected_side, mni, tuple(sorted(options.items())))
    return list(p.cached(key, lambda: _check(p, expected_side, mni, opts)))


def _check(p: PreparedLesion, expected_side, mni, opts) -> "list[Flag]":
    from .morphometry import components, compute_morphometry

    flags: "list[Flag]" = []
    vox_mm3 = _voxel_mm3(p)
    names = slice_axis_names(p.affine)
    morph = compute_morphometry(p)

    # -- input notes ------------------------------------------------------ #
    if not p.was_binary:
        flags.append(Flag("input", INFO,
                          f"The input wasn't binary; voxels above {p.threshold:g} were counted as lesion.",
                          {"threshold": p.threshold}))
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

    # -- flat / single slice (not when it's the slabs of a gap drawing) ---- #
    if gap is None and not hollow:
        flat = _flat_components(p, labels, lo, sizes, opts)
        if flat:
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

    # -- brain-mask checks --------------------------------------------------- #
    if mni:
        com = morph["center_of_mass_mm"]
        in_box = -95 <= com[0] <= 95 and -130 <= com[1] <= 95 and -80 <= com[2] <= 110
        bm, baff = brain_mask()
        on = p.on_grid((bm.shape, baff))
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
                chunk = np.bincount(lab.ravel())[1:]
                depth = ndimage.maximum(_outside_depth(), lab, index=np.arange(1, nlab + 1))
                bvox = float(abs(np.linalg.det(baff[:3, :3])))
                min_vox = max(opts["outside_min_mm3"] / bvox, opts["outside_min_frac"] * n_on)
                # A rim of overdrawing hugging the brain edge is ignored; a
                # chunk reaching well beyond it is flagged.
                big = chunk[(chunk >= min_vox) & (np.asarray(depth) > opts["outside_min_depth_mm"])]
                if big.size:
                    flags.append(Flag("outside_brain", WARN,
                                      f"{big.sum() * bvox:.0f} mm³ of the lesion "
                                      f"({100 * big.sum() / max(n_on, 1):.0f}%) lies outside the brain.",
                                      {"outside_mm3": round(float(big.sum() * bvox), 1),
                                       "n_chunks": int(big.size),
                                       "ignored_stray_voxels": int(chunk.sum() - big.sum())},
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
        max_run = int(round(DEFAULTS["gap_max_mm"] / _axis_mm(p, ax)))
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
    max_gap_mm = DEFAULTS["gap_max_mm"] if max_gap_mm is None else max_gap_mm
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
def clean(lesion, *, specks: bool = True, edge: bool = True, threshold: float = 0.5, **options):
    """Remove image-edge artefacts and/or specks. Opt-in only.

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
    removed = p.n_voxels - int(out.sum())
    return (nib.Nifti1Image(out.astype(np.uint8), p.affine),
            {"removed_components": len(drop), "removed_specks": n_specks,
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
    ``error`` (a message when the file couldn't be read, else None).
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
               "error": None}
        try:
            p = prepare(path, threshold)
            fl = check(p, expected_side=(expected_sides or {}).get(name))
            row["flags"] = [f.to_dict() for f in fl]
            row["volume_mm3"] = compute_morphometry(p)["volume_mm3"]
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
