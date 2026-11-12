"""Compare two masks of a lesion: a follow-up scan against the baseline, or
two copies of one lesion drawn by different people.

    c = lqtpy.compare("baseline.nii.gz", "followup.nii.gz")
    c["volume"]["change_pct"], c["overlap"]["dice"], c["regions"]["gained"]

Both masks must be in the same space (MNI, as for every lqtpy analysis).
Overlap is measured on each mask's own grid when the two share one (exact),
otherwise on the 1 mm MNI grid both are resampled to (nearest neighbour).
Region changes use one atlas (default ``dk``). Plain JSON types only.
"""
from __future__ import annotations

import numpy as np

__all__ = ["compare", "COMPARE_VERSION"]

COMPARE_VERSION = "1.0"


def _r(x, n=2):
    return None if x is None else round(float(x), n)


def compare(a, b, *, threshold: float = 0.5, atlas: str = "dk", min_change_pct: float = 5.0,
            top: int = 10) -> dict:
    """What changed from ``a`` to ``b`` (anything :func:`lqtpy.prepare` takes).

    Returns ``{"compare_version", "a", "b", "grid", "volume", "overlap",
    "centroid_shift_mm", "regions"}``:

    - ``volume``: ``{a_mm3, b_mm3, change_mm3, change_pct}``
    - ``overlap``: ``{dice, jaccard, shared_mm3, only_a_mm3, only_b_mm3}``
    - ``regions``: on ``atlas``: ``gained`` (damaged only in b), ``lost``
      (only in a), ``changed`` (both, percent damaged moved by at least
      ``min_change_pct`` points), each ``[{name, a_pct, b_pct}]`` sorted by
      the size of the change, at most ``top``.
    """
    from .api import atlas_overlap
    from .atlases import load_atlas
    from .prepared import prepare
    pa, pb = prepare(a, threshold), prepare(b, threshold)
    same = pa.shape == pb.shape and np.allclose(pa.affine, pb.affine, atol=1e-4)
    if same:
        fa, fb = np.flatnonzero(pa.mask.ravel()), np.flatnonzero(pb.mask.ravel())
        vox = float(abs(np.linalg.det(pa.affine[:3, :3])))
        grid = "own"
    else:
        ref = load_atlas("tissue").img
        fa, fb = pa.on_grid(ref).flat, pb.on_grid(ref).flat
        vox = float(abs(np.linalg.det(np.asarray(ref.affine)[:3, :3])))
        grid = "mni_1mm"
    shared = np.intersect1d(fa, fb, assume_unique=True).size
    na, nb = fa.size, fb.size
    va, vb = pa.n_voxels * abs(np.linalg.det(pa.affine[:3, :3])), \
        pb.n_voxels * abs(np.linalg.det(pb.affine[:3, :3]))
    ca, cb = pa.world_coords().mean(0), pb.world_coords().mean(0)

    ra = {r["RegionName"]: float(r["PercentDamage"]) for r in atlas_overlap(pa, atlas)
          if r["LesionVoxels"] > 0}
    rb = {r["RegionName"]: float(r["PercentDamage"]) for r in atlas_overlap(pb, atlas)
          if r["LesionVoxels"] > 0}

    def rows(names):
        out = [{"name": n, "a_pct": _r(ra.get(n, 0.0)), "b_pct": _r(rb.get(n, 0.0))} for n in names]
        out.sort(key=lambda x: -abs(x["b_pct"] - x["a_pct"]))
        return out[:top]

    gained = rows(set(rb) - set(ra))
    lost = rows(set(ra) - set(rb))
    changed = rows(n for n in set(ra) & set(rb) if abs(rb[n] - ra[n]) >= min_change_pct)
    return {
        "compare_version": COMPARE_VERSION,
        "a": {"name": pa.name}, "b": {"name": pb.name},
        "grid": grid,
        "volume": {"a_mm3": _r(va, 1), "b_mm3": _r(vb, 1), "change_mm3": _r(vb - va, 1),
                   "change_pct": _r(100.0 * (vb - va) / va, 1) if va else None},
        "overlap": {"dice": _r(2.0 * shared / (na + nb), 4) if na + nb else None,
                    "jaccard": _r(shared / (na + nb - shared), 4) if na + nb - shared else None,
                    "shared_mm3": _r(shared * vox, 1), "only_a_mm3": _r((na - shared) * vox, 1),
                    "only_b_mm3": _r((nb - shared) * vox, 1)},
        "centroid_shift_mm": _r(np.linalg.norm(cb - ca), 1),
        "regions": {"atlas": atlas, "min_change_pct": min_change_pct,
                    "gained": gained, "lost": lost, "changed": changed},
    }
