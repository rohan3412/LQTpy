"""Does a finding's grade track how much fixing it changes the results?

    python tools/qc_impact.py FOLDER --survey survey.json [--per 25] [--out impact.json]

For masks from a survey (``tools/qc_survey.py``) with a fixable finding,
applies ``qc.fix`` for that one finding and measures what changes:

- the volume (% change);
- the largest change in any AAL region's percent damage (percentage points);
- how much the lesion's distribution over AAL regions moves (total-variation
  distance of "percent of lesion", 0-100).

It prints the medians per finding and grade. "Major" should mean the numbers
materially change and "minor" that they barely do. Masks are identified by
the survey's hashed ids only.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import statistics
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))


def _overlap(img) -> dict:
    import lqtpy
    return {r["RegionName"]: (r["PercentDamage"], r["PercentOfLesion"])
            for r in lqtpy.atlas_overlap(img, "aal")}


def one(args) -> dict:
    path, fid, grade, mid = args
    warnings.simplefilter("ignore")
    import lqtpy
    from lqtpy import qc
    out = {"id": mid, "finding": fid, "grade": grade}
    try:
        p = lqtpy.prepare(path)
        new, rep = qc.fix(p, [fid])
        if fid not in rep["applied"]:
            out["error"] = "not applied"
            return out
        v0 = lqtpy.lesion_stats(p)["volume_mm3"]
        v1 = lqtpy.lesion_stats(new)["volume_mm3"]
        a, b = _overlap(p), _overlap(new)
        keys = set(a) | set(b)
        dmg = max((abs(a.get(k, (0, 0))[0] - b.get(k, (0, 0))[0]) for k in keys), default=0.0)
        tv = 0.5 * sum(abs(a.get(k, (0, 0))[1] - b.get(k, (0, 0))[1]) for k in keys)
        out.update(volume_change_pct=round(100 * (v1 - v0) / max(v0, 1e-9), 2),
                   max_damage_change_pp=round(dmg, 2), distribution_shift=round(tv, 2))
    except Exception as exc:                     # noqa: BLE001 - one bad file
        out["error"] = type(exc).__name__          # the text can quote the file name
    return out


def main(argv=None) -> int:
    from lqtpy.formats import is_lesion_file
    from lqtpy.qc_tools import FIXABLE
    from qc_survey import file_id
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder")
    ap.add_argument("--survey", required=True)
    ap.add_argument("--per", type=int, default=25, help="masks per finding and grade")
    ap.add_argument("--only", nargs="*", help="finding ids to measure (default: every fixable one)")
    ap.add_argument("--out")
    ap.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    a = ap.parse_args(argv)
    paths = {file_id(n): os.path.join(a.folder, n) for n in os.listdir(a.folder) if is_lesion_file(n)}
    with open(a.survey, encoding="utf-8") as fh:
        rows = json.load(fh)
    strata = collections.defaultdict(list)
    for r in rows:
        if r["id"] not in paths or "error" in r:
            continue
        for f in r["findings"]:
            if f["id"] in FIXABLE and f["sev"] != "info" and (not a.only or f["id"] in a.only):
                strata[(f["id"], f["grade"])].append(r["id"])
    rng = random.Random(3)
    jobs = []
    for (fid, g), ids in sorted(strata.items()):
        for mid in rng.sample(ids, min(a.per, len(ids))):
            jobs.append((paths[mid], fid, g, mid))
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        res = list(ex.map(one, jobs, chunksize=2))
    ok = [r for r in res if "error" not in r]
    print(f"{len(ok)} fixes measured ({len(res) - len(ok)} skipped)")
    print(f"{'finding':15s} {'grade':9s} {'n':>3s} {'volume %':>9s} {'max dmg pp':>10s} {'shift':>6s}")
    by = collections.defaultdict(list)
    for r in ok:
        by[(r["finding"], r["grade"])].append(r)
    order = {"minor": 0, "moderate": 1, "major": 2}
    for (fid, g), rs in sorted(by.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 3))):
        med = lambda k: statistics.median(abs(r[k]) for r in rs)  # noqa: E731
        print(f"{fid:15s} {g or '-':9s} {len(rs):3d} {med('volume_change_pct'):9.1f} "
              f"{med('max_damage_change_pp'):10.1f} {med('distribution_shift'):6.1f}")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
