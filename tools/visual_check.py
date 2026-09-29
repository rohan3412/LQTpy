"""Visual regression check for the reports.

Renders fixed synthetic reports (single stroke lesion, small-vessel lesions,
a small group), screenshots each in three states -- dashboard dark,
dashboard light, clinical -- with headless Edge/Chrome, and compares them
with the baselines in ``tests/visual/baseline/``.

    python tools/visual_check.py            # compare; writes diffs on failure
    python tools/visual_check.py --update   # accept the current look as baseline

A page passes when the mean absolute pixel difference is under 1.5 (of
255) and under 1% of pixels differ by more than 40. Timestamps, run times
and versions are fixed before rendering so only design changes show up.
Needs a Chromium-family browser and Pillow (``pip install pillow``).
``tests/test_visual.py`` runs this when ``LQTPY_VISUAL=1``.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
BASELINE = os.path.join(ROOT, "tests", "visual", "baseline")
STATES = {"dark": "#dashboard-dark", "light": "#dashboard-light", "clinical": "#clinical-light"}
WIDTH, HEIGHT = 1280, 1800


def _freeze(res):
    """Remove everything that changes between runs."""
    res["timestamp"] = "2026-01-01T09:00:00"
    res["elapsed"] = "1.0 s"
    man = res.get("manifest") or {}
    man["lqtpy_version"] = "0.0.0"
    man["created"] = "2026-01-01T09:00:00"
    run = man.get("run") or {}
    run["environment"] = {"lqtpy": "0.0.0", "python": "3", "numpy": "x", "scipy": "x",
                          "nibabel": "x", "platform": "x"}
    for i in run.get("inputs", []):
        i["path"] = None
        i["sha256"] = "0" * 64
    if "source_path" in res:
        res["source_path"] = None
    for s in res.get("subjects", []):
        s.pop("report_path", None)
    return res


def pages(out_dir: str) -> "dict[str, str]":
    """Render the sample reports; returns ``{name: html path}``."""
    import warnings
    warnings.simplefilter("ignore")
    import lqtpy
    import synthetic as syn
    from lqtpy.report import render_group, render_single
    os.makedirs(out_dir, exist_ok=True)
    stroke = syn.img(syn.blobs([((-40, 10, 10), 14), ((-35, -10, 15), 10)]))
    svd = syn.img(syn.blobs([((-20, 10, 20), 3), ((22, -10, 25), 3), ((-28, -30, 28), 4),
                             ((15, 30, 10), 3), ((0, -60, -35), 4)]))
    out = {}
    for name, les, prof in (("stroke", stroke, "stroke"), ("small_vessel", svd, "small_vessel")):
        res = lqtpy.analyze_single(les, write=False, plots=True, profile=prof, viewer=False)
        res["name"] = name
        path = os.path.join(out_dir, f"{name}.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_single(_freeze(res)))
        out[name] = path
    grp = lqtpy.analyze_group([("stroke", stroke), ("svd", svd),
                               ("lacune", syn.img(syn.ellipsoid((-24, 4, 8), (5, 5, 7))))],
                              write=False, plots=True)
    path = os.path.join(out_dir, "group.html")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(render_group(_freeze(grp)))
    out["group"] = path
    return out


def screenshot(html_path: str, png_path: str, state: str) -> str:
    import pathlib
    from lqtpy.report.pdf import find_browser
    exe = find_browser()
    if not exe:
        raise RuntimeError("no Chromium-family browser found")
    import time
    url = pathlib.Path(html_path).resolve().as_uri() + STATES[state]
    out = os.path.abspath(png_path)
    for _attempt in range(2):
        if os.path.exists(out):
            os.remove(out)
        with tempfile.TemporaryDirectory() as prof:
            subprocess.run([exe, "--headless=new", "--disable-gpu", f"--user-data-dir={prof}",
                            f"--window-size={WIDTH},{HEIGHT}", "--hide-scrollbars",
                            "--force-device-scale-factor=1", "--virtual-time-budget=3000",
                            f"--screenshot={out}", url], capture_output=True, timeout=120)
            # Edge on Windows can return before the file is written.
            last, deadline = -1, time.monotonic() + 30
            while time.monotonic() < deadline:
                size = os.path.getsize(out) if os.path.exists(out) else -1
                if size > 0 and size == last:
                    return png_path
                last = size
                time.sleep(0.5)
    raise RuntimeError(f"no screenshot written for {html_path} ({state})")


def compare(a: str, b: str, diff_path: "str | None" = None) -> "tuple[bool, float, float]":
    from PIL import Image, ImageChops
    import numpy as np
    ia, ib = Image.open(a).convert("RGB"), Image.open(b).convert("RGB")
    if ia.size != ib.size:
        return False, 255.0, 1.0
    d = np.asarray(ImageChops.difference(ia, ib), dtype=np.int16)
    mean = float(d.mean())
    frac = float((d.max(axis=2) > 40).mean())
    ok = mean < 1.5 and frac < 0.01
    if not ok and diff_path:
        Image.fromarray((np.clip(d.max(axis=2) * 4, 0, 255)).astype("uint8")).save(diff_path)
    return ok, mean, frac


def run(update: bool = False) -> int:
    work = tempfile.mkdtemp(prefix="lqtpy-visual-")
    os.makedirs(BASELINE, exist_ok=True)
    failed = 0
    for name, html in pages(work).items():
        for state in STATES:
            shot = screenshot(html, os.path.join(work, f"{name}-{state}.png"), state)
            base = os.path.join(BASELINE, f"{name}-{state}.png")
            if update or not os.path.exists(base):
                os.replace(shot, base)
                print(f"baseline  {name}-{state}")
                continue
            ok, mean, frac = compare(shot, base, os.path.join(work, f"{name}-{state}-diff.png"))
            print(f"{'ok  ' if ok else 'FAIL'}      {name}-{state}  mean {mean:.2f}  changed {100 * frac:.2f}%")
            failed += not ok
    if failed:
        print(f"\n{failed} view(s) changed; screenshots and diffs in {work}")
    return 1 if failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--update", action="store_true", help="Accept the current look as baseline.")
    sys.exit(run(ap.parse_args().update))
