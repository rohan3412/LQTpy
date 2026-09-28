"""Print lqtpy's latency table (docs/ROADMAP.md, Phase 2 budgets).

Median of several runs, warm caches, a 15 mm-radius synthetic lesion on the
1 mm FSL MNI grid. Informational: CI runs it but doesn't gate on absolute
times (machines differ).

Run: python tools/bench.py
"""
from __future__ import annotations

import os
import sys
import time
import warnings

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")
import lqtpy  # noqa: E402
from lqtpy import api, qc  # noqa: E402
from lqtpy.describe import describe  # noqa: E402

AFF = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
SHAPE = (182, 218, 182)


def lesion(center=(-40, 10, 5), r=15.0):
    ijk = np.indices(SHAPE).reshape(3, -1).T.astype(float)
    xyz = ijk @ AFF[:3, :3].T + AFF[:3, 3]
    m = (np.linalg.norm(xyz - np.asarray(center), axis=1) <= r).reshape(SHAPE)
    return nib.Nifti1Image(m.astype(np.uint8), AFF)


def med(fn, n=7):
    ts = []
    for _ in range(n):
        t = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t)
    return 1000 * float(np.median(ts))


def main():
    t = time.perf_counter()
    lqtpy.preload()
    qc.brain_mask()
    print(f"preload (atlases, tract index, brain mask, tissue maps): {1000 * (time.perf_counter() - t):.0f} ms\n")
    L = lesion()
    rows = [
        ("prepare", lambda: lqtpy.prepare(L)),
        ("lesion_stats", lambda: api.lesion_stats(L)),
        ("atlas_overlap aal (resampled)", lambda: api.atlas_overlap(L, "aal")),
        ("atlas_overlap dk (same grid)", lambda: api.atlas_overlap(L, "dk")),
        ("network_rollup_for schaefer", lambda: api.network_rollup_for(L, "schaefer")),
        ("lobe_overlap aal", lambda: api.lobe_overlap(L, "aal")),
        ("tract_proxy_overlap", lambda: api.tract_proxy_overlap(L)),
        ("qc.check", lambda: qc.check(L)),
        ("tissue.locate", lambda: lqtpy.tissue.locate(L)),
        ("describe (stroke, no shape)", lambda: describe(L)),
    ]

    def panel():
        p = lqtpy.prepare(L)
        api.lesion_stats(p)
        api.atlas_overlap(p, "aal")
        api.network_rollup_for(p, "schaefer")
        api.lobe_overlap(p, "dk")
        api.tract_proxy_overlap(p)

    print(f"{'call (raw input each time)':38s} {'median ms':>10s}")
    for name, fn in rows:
        print(f"{name:38s} {med(fn):10.1f}")
    print(f"{'5-call panel on one prepared lesion':38s} {med(panel, 5):10.1f}")
    t = time.perf_counter()
    lqtpy.analyze_single(L, write=False)
    print(f"{'analyze_single (core, no plots)':38s} {1000 * (time.perf_counter() - t):10.1f}")


if __name__ == "__main__":
    main()
