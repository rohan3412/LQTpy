"""Mask-quality survey of a folder of lesion masks (numbers only).

    python tools/qc_survey.py FOLDER [--out survey.json] [--baseline old.json] [--jobs 8]

Runs QC on every lesion file (in parallel) and prints how often each finding
and grade occurs, the verdicts, and the extents. Files are identified by a
short SHA-1 of their name, never by the name itself, so the output can be
shared even when file names hold patient names. With ``--baseline`` the
counts are compared with an earlier survey.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def file_id(name: str) -> str:
    return hashlib.sha1(name.encode()).hexdigest()[:10]


def one(args) -> dict:
    folder, name, tier = args
    sys.path.insert(0, ROOT)
    warnings.simplefilter("ignore")
    import lqtpy
    from lqtpy import qc
    out = {"id": file_id(name)}
    t = time.perf_counter()
    try:
        if hasattr(qc, "assess"):
            r = qc.assess(os.path.join(folder, name), tier=tier)
            out.update(verdict=r.verdict, score=r.score,
                       findings=[{"id": f.id, "sev": f.severity, "grade": f.grade,
                                  "affected_pct": f.affected_pct, "extent": f.extent,
                                  "locations": f.locations[:1]} for f in r.findings])
        else:
            fl = qc.check(os.path.join(folder, name))
            out["findings"] = [{"id": f.id, "sev": f.severity, "ev": f.evidence} for f in fl]
        out["vol"] = lqtpy.lesion_stats(os.path.join(folder, name))["volume_mm3"]
    except Exception as exc:
        # Error texts can quote the path; keep only the kind of error.
        msg = str(exc).splitlines()[0] if str(exc) else ""
        msg = msg.replace(os.path.join(folder, name), "<file>").replace(folder, "<folder>")
        msg = msg.replace(name, "<file>").replace(os.path.splitext(name)[0], "<file>")
        out["error"] = f"{type(exc).__name__}: {msg[:100]}" if "'" not in msg else type(exc).__name__
    out["seconds"] = round(time.perf_counter() - t, 3)
    return out


def summarize(rows) -> dict:
    ok = [r for r in rows if "error" not in r]
    s = {"n": len(rows), "errors": len(rows) - len(ok)}
    s["by_id_grade"] = dict(collections.Counter(
        f"{f['id']}:{f.get('grade') or f['sev']}" for r in ok for f in r["findings"]).most_common())
    s["verdicts"] = dict(collections.Counter(r.get("verdict", "?") for r in ok))
    s["with_any_check"] = sum(1 for r in ok if any(f["sev"] in ("warn", "error") for f in r["findings"]))
    return s


def main():
    sys.path.insert(0, ROOT)
    from lqtpy.formats import is_lesion_file
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--out", help="write per-file results (hashed ids) here")
    ap.add_argument("--baseline", help="compare with an earlier --out file")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    ap.add_argument("--tier", default="full", choices=("header", "fast", "full"))
    a = ap.parse_args()
    names = sorted(f for f in os.listdir(a.folder) if is_lesion_file(f))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        rows = list(ex.map(one, [(a.folder, n, a.tier) for n in names], chunksize=4))
    s = summarize(rows)
    print(f"{s['n']} masks in {time.time() - t0:.0f} s, {s['errors']} unreadable")
    print("verdicts:", s["verdicts"])
    print("masks with Check or Problem:", s["with_any_check"])
    print("findings (id:grade):")
    for k, v in s["by_id_grade"].items():
        print(f"  {k:32s} {v}")
    if a.baseline:
        with open(a.baseline, encoding="utf-8") as fh:
            old = summarize(json.load(fh))
        print("compared with the baseline:")
        keys = sorted(set(old["by_id_grade"]) | set(s["by_id_grade"]))
        for k in keys:
            o, n = old["by_id_grade"].get(k, 0), s["by_id_grade"].get(k, 0)
            if o != n:
                print(f"  {k:32s} {o:5d} -> {n:5d}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, default=str)


if __name__ == "__main__":
    main()
