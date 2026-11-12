"""QC helpers for tools built on lqtpy (drawing apps, registries, pipelines).

- :func:`problem_mask`  the voxels a finding is about, on the mask's own grid,
  for highlighting in a viewer or erasing as one undoable edit. What the
  voxels mean is in :data:`PROBLEM_KIND`: ``"remove"`` (stray voxels:
  specks, strokes, edge fragments, outside the brain), ``"add"`` (voxels
  that are missing: gaps, holes, an unfilled outline) or ``"inspect"``
  (the part to look at: a single-slice part, a box).
- :func:`fix`  applies the fixes lqtpy can make safely and returns a *new*
  image plus exactly which voxels were added and removed, so a caller can
  show a preview or apply it as an undoable edit. Never overwrites.
- :func:`assess_folder`  QC for every mask in a folder, with a summary
  (counts by finding and grade, and by verdict), as plain dicts / JSON.
- :func:`rank`  orders copies of the same lesion, best first (verdict, then
  score, then finer voxels), to pick the one to analyse.
"""
from __future__ import annotations

import os

import numpy as np
from scipy import ndimage

__all__ = ["problem_mask", "PROBLEM_KIND", "FIXABLE", "fix", "fix_plan", "assess_folder", "rank",
           "export_problems"]

PROBLEM_KIND = {
    "specks": "remove", "strokes": "remove", "edge_artifact": "remove", "outside_brain": "remove",
    "sparse_slices": "add", "holes": "add", "hollow": "add", "partial_gaps": "add",
    "flat": "inspect", "box": "inspect", "misregistered": "inspect",
}
#: Findings :func:`fix` can repair on its own.
FIXABLE = ("edge_artifact", "specks", "strokes", "outside_brain", "holes", "hollow", "sparse_slices",
           "partial_gaps")
_S26 = np.ones((3, 3, 3), bool)
#: Fixes judged against the MNI brain: never made for a mask that isn't in MNI space.
_MNI_ONLY = ("outside_brain",)


def _finding(p, finding, **kw):
    from . import qc
    if isinstance(finding, str):
        found = [f for f in qc.check(p, **kw) if f.id == finding]
        if not found:
            return None
        return found[0]
    return finding


def _grid_holes(mask: np.ndarray, opts) -> np.ndarray:
    """:func:`lqtpy.qc_grade.hole_voxels` of a whole-grid mask, worked out in
    the mask's box plus 4 voxels (the closing and erosions in it reach 3;
    beyond that the result is empty, as on the whole grid)."""
    from .qc_grade import hole_voxels
    out = np.zeros(mask.shape, bool)
    lo, hi = [], []
    for a in range(mask.ndim):
        idx = np.flatnonzero(mask.any(axis=tuple(b for b in range(mask.ndim) if b != a)))
        if not idx.size:
            return out
        lo.append(max(0, int(idx[0]) - 4))
        hi.append(min(mask.shape[a], int(idx[-1]) + 5))
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    out[box] = hole_voxels(mask[box], opts)
    return out


def _place(p, sub: np.ndarray, lo) -> np.ndarray:
    out = np.zeros(p.shape, bool)
    sl = tuple(slice(int(lo[a]), int(lo[a]) + sub.shape[a]) for a in range(3))
    out[sl] = sub
    return out


def _outside_voxels(p, opts, rim_only: bool = False) -> np.ndarray:
    """Lesion voxels (own grid) outside the brain mask, beyond the rim
    tolerance (or, with ``rim_only``, any outside voxel)."""
    from .qc import _outside_depth, brain_mask
    bm, baff = brain_mask()
    depth = _outside_depth()
    ijk = p.voxel_coords()
    world = ijk @ p.affine[:3, :3].T + p.affine[:3, 3]
    inv = np.linalg.inv(baff)
    b = np.floor(world @ inv[:3, :3].T + inv[:3, 3] + 0.5).astype(int)
    ok = np.all((b >= 0) & (b < np.array(bm.shape)), axis=1)
    out = ~ok
    inb = b[ok]
    outside = ~bm[tuple(inb.T)]
    if not rim_only:
        outside &= depth[tuple(inb.T)] > opts["outside_rim_mm"]
    out[ok] = outside
    m = np.zeros(p.shape, bool)
    m[tuple(ijk[out].T)] = True
    return m


def problem_mask(lesion, finding, *, threshold: float = 0.5, mni: bool = True,
                 **options) -> np.ndarray:
    """Boolean mask (the lesion's own grid) of the voxels ``finding`` is
    about; ``finding`` is a :class:`~lqtpy.qc_model.Finding` or a finding
    id. See :data:`PROBLEM_KIND` for what the voxels mean. An id that isn't
    present, or a finding about the whole mask (space, values,
    plausibility), gives an empty mask."""
    from . import qc
    from .morphometry import components
    from .prepared import prepare
    p = prepare(lesion, threshold)
    opts = {**qc.DEFAULTS, **options}
    f = _finding(p, finding, threshold=threshold, mni=mni)
    if f is None or (not mni and f.id == "outside_brain"):
        return np.zeros(p.shape, bool)
    labels, lo, sizes = components(p)
    fid = f.id
    if fid == "edge_artifact":
        return _place(p, np.isin(labels, qc._edge_components(p)), lo)
    if fid == "specks":
        from .qc_grade import _specks_ids
        return _place(p, np.isin(labels, _specks_ids(p, opts)), lo)
    if fid == "box":
        comps = [b["component"] for b in f.evidence.get("boxes", [])]
        return _place(p, np.isin(labels, comps), lo)
    if fid == "flat":
        if "slab" in f.evidence:
            ax = f.evidence["slab"]["axis"]
            c = p.voxel_coords()
            mid = int(round(c[:, ax].mean()))
            w = max(1, int(round(f.evidence["slab"]["thickness_mm"] / np.linalg.norm(p.affine[:3, ax]))))
            m = np.zeros(p.shape, bool)
            sl = [slice(None)] * 3
            sl[ax] = slice(max(0, mid - w), mid + w + 1)
            m[tuple(sl)] = p.mask[tuple(sl)]
            return m
        comps = [c["component"] for c in f.evidence.get("components", [])]
        return _place(p, np.isin(labels, comps), lo)
    if fid == "strokes":
        sub = qc._analysis_mask(p, opts)
        axes = {st["axis"] for st in f.evidence.get("strokes", [])}
        zooms = np.linalg.norm(p.affine[:3, :3], axis=0)
        svox, _ = qc._stroke_voxels(sub, zooms, opts, skip_axes=set(range(3)) - axes)
        return _place(p, svox, lo)
    if fid == "outside_brain":
        return _outside_voxels(p, opts, rim_only=f.grade is None)
    if fid in ("holes", "hollow", "sparse_slices", "partial_gaps"):
        img, rep = fix(p, [fid], threshold=threshold)
        return rep["added"]
    return np.zeros(p.shape, bool)


def export_problems(lesion, out_dir, *, threshold: float = 0.5, context=None, mni: bool = True,
                    **options) -> "list[dict]":
    """Write one overlay mask per finding into ``out_dir`` (created if
    needed), on the lesion's own grid, to load over it in MRIcroGL or
    ITK-SNAP while fixing: ``<lesion>_<finding id>.nii.gz``, 1 where the
    finding's voxels are (:func:`problem_mask`). Findings about the whole
    mask have no voxels and get no file. Never overwrites a file; returns
    ``[{"id", "kind", "voxels", "path"}]`` for the files written."""
    import nibabel as nib
    from . import qc
    from .prepared import prepare
    p = prepare(lesion, threshold)
    os.makedirs(out_dir, exist_ok=True)
    out, seen = [], set()
    for f in qc.check(p, context=context, mni=mni):
        if f.id in seen:
            continue
        seen.add(f.id)
        m = problem_mask(p, f, threshold=threshold, mni=mni, **options)
        if not m.any():
            continue
        path = os.path.join(out_dir, f"{p.name}_{f.id}.nii.gz")
        if os.path.exists(path):
            raise FileExistsError(f"{path} already exists; lqtpy never overwrites a file")
        nib.save(nib.Nifti1Image(m.astype(np.uint8), p.affine), path)
        out.append({"id": f.id, "kind": PROBLEM_KIND.get(f.id, "inspect"),
                    "voxels": int(m.sum()), "path": path})
    return out


def fix_plan(lesion, *, threshold: float = 0.5, context=None, ids=None, mni: bool = True,
             **options) -> "list[dict]":
    """Each fix lqtpy can make, worked out on its own, for a preview: one
    entry per fixable finding the mask has (or per id in ``ids``), in
    :data:`FIXABLE` order::

        {"id", "label", "action", "advice", "adds_voxels",
         "voxels_added", "voxels_removed", "added", "removed"}

    ``added`` / ``removed`` are boolean masks on the mask's own grid. Every
    entry starts from the original mask, so after applying one, plan again
    (fixes can overlap: filling gaps can also fill a hole). A fix that
    would change nothing is left out. Nothing is written. ``mni=False``
    (a mask not in MNI space): nothing is judged against the MNI brain, so
    no "outside the brain" fix is offered."""
    from . import qc
    from .prepared import prepare
    from .qc_advice import advice_for
    p = prepare(lesion, threshold)
    found = {f.id: f for f in qc.check(p, context=context, mni=mni)}
    wanted = [i for i in FIXABLE if (i in found if ids is None else i in ids)
              and (mni or i not in _MNI_ONLY)]
    out = []
    for fid in wanted:
        _, rep = fix(p, [fid], threshold=threshold, context=context, mni=mni, **options)
        if not (rep["voxels_added"] or rep["voxels_removed"]):
            continue
        f = found.get(fid)
        adv = advice_for(f if f is not None else fid)
        out.append({"id": fid, "label": f.label if f is not None else "",
                    "action": adv["action"], "advice": adv["text"],
                    "adds_voxels": bool(rep["voxels_added"]),
                    "voxels_added": rep["voxels_added"], "voxels_removed": rep["voxels_removed"],
                    "added": rep["added"], "removed": rep["removed"]})
    return out


def fix(lesion, ids=None, *, threshold: float = 0.5, context=None, mni: bool = True, **options):
    """Repair what lqtpy can repair safely: strokes, specks, edge fragments,
    voxels beyond the brain edge (removed), gaps between drawn slices, partly
    drawn slices, enclosed holes and unfilled outlines (filled).

    ``ids``: finding ids to fix (default: every fixable finding the mask
    has). ``mni=False`` (a mask not in MNI space, e.g. drawn on the patient's
    own scan): the check skips the MNI brain, and an ``outside_brain`` fix is
    skipped even when asked for. Returns ``(new_image, report)``; the input is never changed.
    ``report``: ``applied`` / ``skipped`` (with reasons), ``voxels_added``,
    ``voxels_removed``, and boolean masks ``added`` and ``removed`` on the
    mask's own grid (for previews and undo)."""
    import nibabel as nib
    from . import qc
    from .prepared import prepare
    p = prepare(lesion, threshold)
    opts = {**qc.DEFAULTS, **options}
    found = {f.id: f for f in qc.check(p, context=context, mni=mni)}
    wanted = list(ids) if ids is not None else [i for i in FIXABLE if i in found]
    cur = p.mask.copy()
    applied, skipped = [], {}
    for fid in wanted:
        if fid not in FIXABLE:
            skipped[fid] = "not something lqtpy can repair (redraw or re-register)"
            continue
        if fid not in found and ids is None:
            continue
        if not mni and fid in _MNI_ONLY:
            skipped[fid] = "only for masks in MNI space (the brain's edge is the MNI template's)"
            continue
        before = cur.copy()
        cp = p._derived(cur.copy(), name=p.name, threshold=0.5)
        if fid in ("edge_artifact", "specks", "strokes", "outside_brain"):
            img, _ = qc.clean(cp, specks=fid == "specks", edge=fid == "edge_artifact",
                              strokes=fid == "strokes", outside=fid == "outside_brain")
            cur = np.asarray(img.dataobj) > 0
        elif fid == "sparse_slices":
            f = found.get(fid)
            ax = f.evidence.get("axis") if f else None
            img, rep = qc.repair_gaps(cp, axis=ax,
                                      slices=qc.gap_slices_only(f.evidence) if f else None)
            cur = np.asarray(img.dataobj) > 0
            if f is not None and f.evidence.get("from_context"):
                # A lone empty slice at a segment's edge: copy the nearer
                # neighbour's outline into it.
                moved = np.moveaxis(cur, ax, 0)
                for k in f.evidence.get("gap_slices", []):
                    if moved[k].any():
                        continue
                    nb = [moved[j] for j in (k - 1, k + 1) if 0 <= j < moved.shape[0] and moved[j].any()]
                    if len(nb) == 2 and (nb[0] & nb[1]).any():
                        moved[k] = nb[0] & nb[1]
                    elif nb:
                        moved[k] = nb[0] | nb[-1]
        elif fid == "holes":
            cur = cur | _grid_holes(cur, opts)
        elif fid == "partial_gaps":
            cur = cur | qc._partial_gap_voxels(cp, **options)
        elif fid == "hollow":
            f = found.get(fid)
            names = qc.slice_axis_names(p.affine)
            ax = next((a for a, n in names.items() if f and n == f.evidence.get("plane")), 2)
            from ._morph import fill_holes
            sub, off = cp.cropped(pad=4)
            d = ndimage.binary_dilation(sub, iterations=3)
            fl = ndimage.binary_erosion(fill_holes(d, axis=ax), iterations=3) | sub
            sl3 = tuple(slice(int(off[a]), int(off[a]) + sub.shape[a]) for a in range(3))
            cur[sl3] |= fl
        if (cur != before).any():
            applied.append(fid)
        else:
            skipped[fid] = "nothing to change"
    added = cur & ~p.mask
    removed = p.mask & ~cur
    return (nib.Nifti1Image(cur.astype(np.uint8), p.affine),
            {"applied": applied, "skipped": skipped, "voxels_added": int(added.sum()),
             "voxels_removed": int(removed.sum()), "added": added, "removed": removed})


# --------------------------------------------------------------------------- #
# Folders and copies
# --------------------------------------------------------------------------- #
def _assess_one(args):
    path, tier, kw = args
    from . import qc
    try:
        return {"file": path, "name": os.path.basename(path),
                "result": qc.assess(path, tier=tier, **kw).to_dict()}
    except Exception as exc:
        return {"file": path, "name": os.path.basename(path),
                "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}


def assess_folder(folder, *, tier: str = "full", recursive: bool = False, jobs: int = 1,
                  **kw) -> dict:
    """QC every lesion file in ``folder``. Returns ``{"results": [{"file",
    "name", "result" (QCResult.to_dict()) or "error"}], "summary": {"n",
    "errors", "verdicts", "by_finding": {"id:grade": count}, "qc_version"}}``.
    ``jobs > 1``, ``-1`` or ``"auto"`` uses worker processes."""
    from .formats import is_lesion_file
    from .qc_model import QC_VERSION
    files = []
    if recursive:
        for root, _, fs in os.walk(folder):
            files += [os.path.join(root, f) for f in fs if is_lesion_file(f)]
    else:
        files = [os.path.join(folder, f) for f in os.listdir(folder) if is_lesion_file(f)]
    files.sort()
    args = [(f, tier, kw) for f in files]
    from ._jobs import run_tasks
    results = run_tasks(_assess_one, args, jobs)
    verdicts, by = {}, {}
    for r in results:
        if "result" not in r:
            continue
        v = r["result"]["verdict"]
        verdicts[v] = verdicts.get(v, 0) + 1
        for f in r["result"]["findings"]:
            k = f"{f['id']}:{f['grade'] or 'note'}"
            by[k] = by.get(k, 0) + 1
    return {"results": results,
            "summary": {"n": len(results), "errors": sum(1 for r in results if "error" in r),
                        "verdicts": verdicts,
                        "by_finding": dict(sorted(by.items(), key=lambda kv: -kv[1])),
                        "qc_version": QC_VERSION, "tier": tier}}


_VERDICT_RANK = {"usable": 0, "usable with caution": 1, "fix before use": 2, "unusable": 3}


def rank(copies, *, tier: str = "full", **kw) -> "list[dict]":
    """Order copies of one lesion, best first: verdict, then score, then
    finer voxels. ``copies``: paths or images. Returns ``[{"index",
    "source", "verdict", "score", "voxel_mm", "findings"}]``; ``index`` is
    the position in ``copies``."""
    from . import qc
    from .prepared import prepare
    out = []
    for i, c in enumerate(copies):
        try:
            r = qc.assess(c, tier=tier, **kw)
            p = prepare(c)
            vox = float(np.mean(np.linalg.norm(p.affine[:3, :3], axis=0)))
            out.append({"index": i, "source": os.fspath(c) if isinstance(c, (str, os.PathLike)) else None,
                        "verdict": r.verdict, "score": r.score, "voxel_mm": round(vox, 3),
                        "findings": [f"{f.id}:{f.grade or 'note'}" for f in r.findings]})
        except Exception as exc:
            out.append({"index": i, "source": os.fspath(c) if isinstance(c, (str, os.PathLike)) else None,
                        "verdict": "unusable", "score": 0, "voxel_mm": None,
                        "findings": [f"error:{type(exc).__name__}"]})
    out.sort(key=lambda r: (_VERDICT_RANK.get(r["verdict"], 9), -r["score"], r["voxel_mm"] or 99))
    return out
