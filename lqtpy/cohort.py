"""Cohort-level checks and statistics for group runs.

- :func:`find_duplicates`: identical or near-identical masks under
  different names (the same file saved twice, or copied into two folders).
- :func:`find_outliers`: subjects far from the rest of the group in volume
  or location (a wrong file, a mask in another space, a flipped mask).
- :func:`compare`: two sub-cohorts (e.g. aphasia vs no aphasia) region by
  region: how many are affected in each, mean damage, the difference, and a
  simple test (Fisher's exact test on affected / not affected, Mann-Whitney
  U on percent damaged, both with Benjamini-Hochberg FDR q-values).

These are screening statistics for exploring a cohort, not a replacement
for a lesion-symptom mapping analysis (no covariates, no lesion-volume
correction, no permutation-based family-wise control).
"""
from __future__ import annotations

import hashlib

import numpy as np

__all__ = ["Signature", "signature", "find_duplicates", "find_outliers", "compare", "fdr"]


class Signature:
    """What duplicate detection needs from one subject, kept small: an exact
    hash, the voxel count, centre of mass, and the 2 mm-grid voxel ids."""

    __slots__ = ("name", "digest", "n", "com", "coarse")

    def __init__(self, name, digest, n, com, coarse):
        self.name, self.digest, self.n, self.com, self.coarse = name, digest, n, com, coarse

    def to_dict(self):
        return {"name": self.name, "digest": self.digest, "n": self.n, "com": list(self.com),
                "coarse": self.coarse.tolist()}

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["digest"], d["n"], tuple(d["com"]),
                   np.asarray(d["coarse"], np.int32))


def signature(name, flat_idx: np.ndarray, shape, affine) -> Signature:
    """From the subject's voxel ids on the reference grid."""
    flat_idx = np.asarray(flat_idx, np.int64)
    digest = hashlib.sha1(np.sort(flat_idx).astype(np.int64).tobytes()).hexdigest()
    ijk = np.column_stack(np.unravel_index(flat_idx, shape))
    com = tuple(float(v) for v in (affine[:3, :3] @ ijk.mean(0) + affine[:3, 3])) if len(ijk) else (0, 0, 0)
    c = ijk // 2
    cs = tuple((np.asarray(shape) + 1) // 2)
    coarse = np.unique(np.ravel_multi_index(c.T, cs)).astype(np.int32) if len(c) else np.zeros(0, np.int32)
    return Signature(name, digest, int(len(flat_idx)), com, coarse)


def find_duplicates(sigs: "list[Signature]", *, min_dice: float = 0.95,
                    max_com_mm: float = 5.0, max_volume_diff: float = 0.1) -> "list[dict]":
    """Pairs of identical or near-identical masks.

    Identical: same voxels on the reference grid. Near-identical: centres
    within ``max_com_mm``, volumes within ``max_volume_diff``, and a Dice
    overlap of at least ``min_dice`` on a 2 mm grid."""
    out = []
    by_digest: "dict[str, list]" = {}
    for s in sigs:
        by_digest.setdefault(s.digest, []).append(s.name)
    exact = set()
    for names in by_digest.values():
        if len(names) > 1:
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    out.append({"a": names[i], "b": names[j], "dice": 1.0, "kind": "identical"})
                    exact.add((names[i], names[j]))
    order = sorted(range(len(sigs)), key=lambda i: sigs[i].n)
    for pos, i in enumerate(order):
        a = sigs[i]
        for j in order[pos + 1:]:
            b = sigs[j]
            if b.n > a.n * (1 + max_volume_diff):
                break
            if (a.name, b.name) in exact or (b.name, a.name) in exact or a.digest == b.digest:
                continue
            if np.linalg.norm(np.subtract(a.com, b.com)) > max_com_mm:
                continue
            inter = np.intersect1d(a.coarse, b.coarse, assume_unique=True).size
            dice = 2.0 * inter / max(a.coarse.size + b.coarse.size, 1)
            if dice >= min_dice:
                out.append({"a": a.name, "b": b.name, "dice": round(dice, 3),
                            "kind": "near-identical"})
    return out


def _robust_z(x: np.ndarray) -> np.ndarray:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-9:
        sd = np.std(x)
        return np.zeros_like(x) if sd < 1e-9 else (x - med) / sd
    return 0.6745 * (x - med) / mad


def find_outliers(subjects: "list[dict]", *, z_limit: float = 3.5, min_n: int = 10) -> "dict[str, list]":
    """``{name: [{"feature", "value", "z", "message"}]}`` for subjects far
    from the group (robust z-scores on log volume and on the distance of the
    centre of mass from the group's median centre). Needs ``min_n``
    subjects; returns ``{}`` otherwise."""
    if len(subjects) < min_n:
        return {}
    names = [s["name"] for s in subjects]
    vol = np.log10(np.array([max(s["morphometry"]["volume_mm3"], 1.0) for s in subjects]))
    com = np.array([s["morphometry"]["center_of_mass_mm"] for s in subjects], float)
    dist = np.linalg.norm(com - np.median(com, axis=0), axis=1)
    zv, zd = _robust_z(vol), _robust_z(dist)
    out: "dict[str, list]" = {}
    for i, n in enumerate(names):
        items = []
        if abs(zv[i]) > z_limit:
            word = "much larger" if zv[i] > 0 else "much smaller"
            items.append({"feature": "volume", "value": round(10 ** vol[i] / 1000, 2),
                          "z": round(float(zv[i]), 1),
                          "message": f"Volume is {word} than the rest of the group (robust z = {zv[i]:.1f})."})
        if zd[i] > z_limit:
            items.append({"feature": "location", "value": round(float(dist[i]), 1),
                          "z": round(float(zd[i]), 1),
                          "message": f"Location is far from the rest of the group ({dist[i]:.0f} mm from "
                                     f"the group's median centre; robust z = {zd[i]:.1f}). Check the "
                                     "file and its space."})
        if items:
            out[n] = items
    return out


def fdr(p: "list[float]") -> "list[float]":
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = len(p)
    if not n:
        return []
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(q, 1.0)
    return [round(float(v), 4) for v in out]


def compare(per_subject: "list[list[dict] | None]", groups: "list[str | None]", a: str, b: str,
            key: str, value: str) -> "list[dict]":
    """Region-by-region comparison of two groups.

    ``per_subject[i]`` is subject i's records (e.g. its region table),
    ``groups[i]`` its group label. Subjects in neither group are ignored.
    Rows: ``Name, NAffectedA, NA, PctAffectedA, MeanA, NAffectedB, NB,
    PctAffectedB, MeanB, DiffMean, PFisher, QFisher, PMannWhitney,
    QMannWhitney``; means include unaffected subjects as 0. Sorted by the
    Mann-Whitney p."""
    from scipy import stats
    ia = [i for i, g in enumerate(groups) if g == a]
    ib = [i for i, g in enumerate(groups) if g == b]
    if not ia or not ib:
        return []
    names = sorted({r[key] for i in ia + ib for r in (per_subject[i] or [])})
    vals = {}
    for side, idx in (("A", ia), ("B", ib)):
        m = np.zeros((len(idx), len(names)))
        pos = {n: k for k, n in enumerate(names)}
        for row, i in enumerate(idx):
            for r in per_subject[i] or []:
                m[row, pos[r[key]]] = float(r[value])
        vals[side] = m
    rows = []
    for k, n in enumerate(names):
        xa, xb = vals["A"][:, k], vals["B"][:, k]
        ka, kb = int((xa > 0).sum()), int((xb > 0).sum())
        p_f = float(stats.fisher_exact([[ka, len(ia) - ka], [kb, len(ib) - kb]])[1])
        if np.all(xa == xa[0]) and np.all(xb == xb[0]) and xa[0] == xb[0]:
            p_u = 1.0
        else:
            p_u = float(stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue)
        rows.append({"Name": n, "NAffectedA": ka, "NA": len(ia),
                     "PctAffectedA": round(100.0 * ka / len(ia), 1), "MeanA": round(float(xa.mean()), 2),
                     "NAffectedB": kb, "NB": len(ib),
                     "PctAffectedB": round(100.0 * kb / len(ib), 1), "MeanB": round(float(xb.mean()), 2),
                     "DiffMean": round(float(xa.mean() - xb.mean()), 2),
                     "PFisher": round(p_f, 4), "PMannWhitney": round(p_u, 4)})
    for col, qcol in (("PFisher", "QFisher"), ("PMannWhitney", "QMannWhitney")):
        for r, q in zip(rows, fdr([r[col] for r in rows])):
            r[qcol] = q
    rows.sort(key=lambda r: (r["PMannWhitney"], -abs(r["DiffMean"])))
    return rows
