"""Build lqtpy/data/shape_reference.json from a cohort feature file.

Input: a JSON-lines file of per-lesion features as written by the cohort scan
(one record per lesion: ``ok``, ``qc`` flags, ``mask_hash``,
``shape_per_component``). Output: **aggregate quantiles only** -- per
lesion-size band, the 5th..95th percentiles of each shape measure and the
number of lesions behind them. No per-lesion values, identifiers or masks
are written.

Lesions are excluded when QC found problems that distort shape
(sparse_slices, flat, hollow, not_mni), and exact duplicate masks are
counted once. Shape values are those of each lesion's largest component.

Run: python tools/calibrate_shape_cohort.py <features.jsonl> [--label "..."]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "lqtpy", "data", "shape_reference.json")
MEASURES = ("sphericity", "surface_regularity", "solidity", "elongation", "flatness")
#: Size bands by largest-component volume (mm^3). Shape statistics depend on
#: size on a voxel grid, so each lesion is compared within its band.
BANDS = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 60000), (60000, float("inf"))]
PCTS = list(range(5, 100, 5))
MIN_PER_BAND = 30
EXCLUDE_FLAGS = {"sparse_slices", "flat", "hollow", "not_mni"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--label", default="clinical lesion database (mixed conditions)")
    args = ap.parse_args()
    seen, rows, excluded = set(), [], {"qc": 0, "duplicate": 0, "not_characterised": 0}
    with open(args.features) as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            if r["mask_hash"] in seen:
                excluded["duplicate"] += 1
                continue
            seen.add(r["mask_hash"])
            if any(q["id"] in EXCLUDE_FLAGS and q["severity"] != "info" for q in r["qc"]):
                excluded["qc"] += 1
                continue
            comps = r.get("shape_per_component") or []
            if not comps or comps[0].get("sphericity") is None:
                excluded["not_characterised"] += 1
                continue
            c = comps[0]
            vol = c["n_voxels"] * r["morph"]["voxel_volume_mm3"]
            rows.append((vol, c))
    bands = []
    for lo, hi in BANDS:
        sel = [c for v, c in rows if lo <= v < hi]
        band = {"min_mm3": lo, "max_mm3": None if hi == float("inf") else hi, "n": len(sel)}
        if len(sel) >= MIN_PER_BAND:
            for m in MEASURES:
                vals = np.array([c[m] for c in sel if c.get(m) is not None], float)
                band[m] = [round(float(x), 4) for x in np.percentile(vals, PCTS)]
        bands.append(band)
    out = {
        "description": "Reference percentiles (5..95 step 5) of lesion shape measures, "
                       "per lesion-size band (largest component). Aggregates only.",
        "reference": args.label,
        "built": datetime.date.today().isoformat(),
        "n_lesions": len(rows),
        "excluded": excluded,
        "percentiles": PCTS,
        "bands": bands,
    }
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"wrote {OUT}: {len(rows)} lesions; per band n = {[b['n'] for b in bands]}; excluded {excluded}")


if __name__ == "__main__":
    main()
