"""Run the synthetic QC library and print how the checks do.

    python tools/qc_zoo_report.py            # every case
    python tools/qc_zoo_report.py --fast     # the CI subset
    python tools/qc_zoo_report.py --json out.json

Per case: the expected finding, whether it was found, its grade, the
extent reported, whether its first location is near the true place, and
any false alarms (Check or above on a clean lesion). A summary gives the
detection rate, grade agreement, false alarms and location hits.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
warnings.simplefilter("ignore")

import numpy as np  # noqa: E402

GRADES = {None: 0, "minor": 1, "moderate": 2, "major": 3}


def run_case(c) -> dict:
    from lqtpy import qc
    t = time.perf_counter()
    try:
        if hasattr(qc, "assess"):
            res = qc.assess(c.img(), **c.kwargs)
            found = res.findings
        else:
            found = qc.check(c.img(), **c.kwargs)
        err = None
    except Exception as exc:  # a crash is a result too
        found, err = [], f"{type(exc).__name__}: {exc}"
    dt = time.perf_counter() - t
    ids = {f.id: f for f in found}
    hard = [f for f in found if f.severity in ("warn", "error")]
    out = {"name": c.name, "group": c.group, "seconds": round(dt, 3), "error": err,
           "found": [(f.id, f.severity, getattr(f, "grade", None)) for f in found]}
    if c.expect:
        hit = next((ids[e] for e in c.expect if e in ids), None)
        out["detected"] = hit is not None
        out["id"] = hit.id if hit else None
        g = getattr(hit, "grade", None) if hit else None
        out["grade"] = g
        out["grade_ok"] = (hit is not None and (c.grade is None or GRADES.get(g, 0) >= GRADES[c.grade]))
        out["extent"] = getattr(hit, "extent", None) if hit else None
        if c.near_mm is not None and hit is not None and getattr(hit, "locations", None):
            loc = hit.locations[0]
            mm = loc.get("mm")
            if isinstance(mm, (list, tuple)) and len(mm) == 3:
                out["loc_mm_err"] = round(float(np.linalg.norm(np.subtract(mm, c.near_mm))), 1)
    if c.forbid:
        out["false_alarms"] = [f.id for f in hard if f.id in c.forbid]
    out["undetectable"] = c.undetectable
    return out


def main():
    import qc_zoo
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fast", action="store_true", help="only the CI subset")
    ap.add_argument("--json", help="write per-case results here")
    a = ap.parse_args()
    cases = [c for c in qc_zoo.all_cases() if c.fast or not a.fast]
    rows = [run_case(c) for c in cases]
    w = max(len(r["name"]) for r in rows)
    for r in rows:
        if "detected" in r:
            mark = "ok  " if r["detected"] else ("n/a " if r["undetectable"] else "MISS")
            extra = f"{r['id'] or '-':15s} grade {str(r['grade']):8s}{'' if r['grade_ok'] else ' LOW'}"
            if "loc_mm_err" in r:
                extra += f"  loc {r['loc_mm_err']} mm"
        elif r.get("false_alarms"):
            mark, extra = "FA  ", ",".join(r["false_alarms"])
        elif r["undetectable"]:
            mark, extra = "n/a ", "(known limitation)"
        else:
            mark, extra = "ok  ", ""
        err = f"  ERROR {r['error']}" if r["error"] else ""
        print(f"{mark} {r['name']:{w}s}  {extra}{err}  [{r['seconds']:.2f}s]")
    exp = [r for r in rows if "detected" in r and not r["undetectable"]]
    neg = [r for r in rows if "false_alarms" in r]
    locs = [r for r in rows if "loc_mm_err" in r]
    print()
    print(f"detected      {sum(r['detected'] for r in exp)}/{len(exp)}")
    print(f"grade ok      {sum(r['grade_ok'] for r in exp)}/{len(exp)}")
    print(f"false alarms  {sum(bool(r['false_alarms']) for r in neg)}/{len(neg)} clean or problem cases")
    if locs:
        print(f"location      {sum(r['loc_mm_err'] <= 15 for r in locs)}/{len(locs)} within 15 mm")
    print(f"errors        {sum(bool(r['error']) for r in rows)}")
    t = np.array([r["seconds"] for r in rows])
    print(f"time per case median {np.median(t):.2f}s, max {t.max():.2f}s")
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
