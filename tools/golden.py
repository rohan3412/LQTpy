"""Golden-output harness: prove a change leaves lqtpy's outputs identical.

Records every output lqtpy produces for a fixed set of inputs (QC, overlap,
descriptions, figures, fixes, full reports with every file written, group
runs, the command line) as JSON, with timestamps and run times masked, then
compares two recordings.

    # 1. a checkout of the version to compare against, e.g. the last release
    git worktree add ../lqtpy-baseline v0.7.0
    # 2. record both (run them one after the other, not side by side:
    #    group and command-line cases use one fixed working folder)
    python tools/golden.py record OUT_BASE --tree ../lqtpy-baseline
    python tools/golden.py record OUT_NEW
    # 3. compare
    python tools/golden.py diff OUT_BASE OUT_NEW

``record OUT [zoo|group|cli|masks|all]`` (default ``all`` = zoo, group,
cli, plus masks when ``--masks`` is given):

- ``zoo``: the synthetic QC library (``tests/qc_zoo.py``), about 70 cases.
- ``group``: a group run with metadata, comparison and every extra, and an
  anonymised one without subject reports.
- ``cli``: ``single``, ``describe``, ``qc`` (file and folder), ``group``.
- ``masks``: real masks from ``--masks FOLDER`` (``.nii`` / ``.nii.gz``).
  They are named in the recordings by a hash of the file name, never by
  name or path. Keep that folder outside the repository.

``--tree PATH`` imports lqtpy from another checkout; the test cases always
come from this checkout's ``tests/``, so both recordings see the same inputs.
``--only TEXT`` records only cases whose name contains TEXT.

Exit status of ``diff``: 0 when every recording is identical, 1 otherwise.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
#: One fixed working folder, so file paths inside outputs are the same in every run.
WORK = os.path.join(tempfile.gettempdir(), "lqtpy-golden-work")

DROP_KEYS = {"compute_seconds", "elapsed", "timestamp", "analysed", "generated", "created",
             "seconds", "t_load", "duration_s", "elapsed_s", "started", "finished", "wall_s"}

_TS = [
    (re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?"), "<TS>"),
    (re.compile(r"analysed in [0-9.]+ ?(ms|s|min)"), "analysed in <T>"),
    (re.compile(r"\bin [0-9.]+ ?(ms|s|min)\b"), "in <T>"),
    (re.compile(r"\"(elapsed|compute_seconds|seconds|timestamp)\": ?[^,}\n]+"), r'"\1": <T>'),
    (re.compile(r"(elapsed|compute_seconds|seconds),[0-9.]+"), r"\1,<T>"),
]


def norm_text(s: str) -> str:
    for rx, rep in _TS:
        s = rx.sub(rep, s)
    s = s.replace("\\\\", "/").replace("\\", "/")
    s = s.replace(WORK.replace("\\", "/"), "<WORK>")
    s = re.sub(r"[A-Za-z]:/Users/[^\"'<>\s]*?/Temp/tmp[a-z0-9_]+", "<TMP>", s)
    return s


def _setup(tree):
    root = os.path.abspath(tree) if tree else REPO
    sys.path.insert(0, root)
    sys.path.insert(0, os.path.join(REPO, "tests"))          # cases: always from this checkout
    import lqtpy
    got = os.path.normcase(os.path.abspath(lqtpy.__file__))
    if not got.startswith(os.path.normcase(root)):
        sys.exit(f"lqtpy was imported from {got}, not from {root}: an installed copy wins "
                 "(editable install?). Run from a Python without it, or uninstall it.")
    return root


def _clean(o):
    """JSON-able, deterministic form: numpy -> python, timing keys dropped,
    big arrays and long strings hashed."""
    import numpy as np
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in sorted(o.items(), key=lambda kv: str(kv[0]))
                if str(k) not in DROP_KEYS}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        if o.size > 64:
            return {"__array__": list(o.shape), "dtype": str(o.dtype),
                    "sha": hashlib.sha1(np.ascontiguousarray(o).tobytes()).hexdigest()[:16]}
        return _clean(o.tolist())
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, float):
        return o if np.isfinite(o) else str(o)
    if isinstance(o, str) and len(o) > 300:
        return {"__str__": len(o), "sha": hashlib.sha1(norm_text(o).encode("utf-8")).hexdigest()[:16]}
    if hasattr(o, "to_dict"):
        return _clean(o.to_dict())
    if hasattr(o, "__dict__") and not isinstance(o, type):
        return _clean({k: v for k, v in vars(o).items() if not k.startswith("_")})
    if isinstance(o, str):
        return norm_text(o)
    if isinstance(o, (int, bool)) or o is None:
        return o
    return repr(type(o))


def _files(d: str) -> dict:
    import numpy as np
    import nibabel as nib
    out = {}
    for dp, _dn, fn in os.walk(d):
        for f in sorted(fn):
            p = os.path.join(dp, f)
            rel = os.path.relpath(p, d).replace("\\", "/")
            with open(p, "rb") as fh:
                data = fh.read()
            if f.endswith((".html", ".csv", ".json", ".txt", ".svg", ".md", ".tsv", ".log")):
                text = norm_text(data.decode("utf-8", errors="replace"))
                out[rel] = hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]
            elif f.endswith((".nii", ".nii.gz")):
                try:
                    img = nib.load(p)
                    arr = np.asanyarray(img.dataobj)
                    out[rel] = "nii:" + hashlib.sha1(
                        arr.tobytes() + str(img.affine.tolist()).encode()).hexdigest()[:16]
                except Exception:
                    out[rel] = "bin:" + hashlib.sha1(data).hexdigest()[:16]
            else:
                out[rel] = "bin:" + hashlib.sha1(data).hexdigest()[:16]
    return out


def _img_hash(img) -> str:
    import numpy as np
    arr = np.asanyarray(img.dataobj)
    return hashlib.sha1(np.ascontiguousarray(arr).tobytes()
                        + str(np.round(img.affine, 6).tolist()).encode()).hexdigest()[:16]


def _try(fn):
    try:
        return _clean(fn())
    except Exception as exc:                 # errors are outputs too
        return {"__error__": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _fresh(name):
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d)
    return d


def record_case(name: str, img, kwargs: dict, timings: dict, *, real: bool = False) -> dict:
    import nibabel as nib
    import lqtpy
    from lqtpy import api, figures, qc
    out = {}
    t0 = time.perf_counter()
    P = lqtpy.prepare(img, name=name)
    out["prepare"] = _clean({"n": P.n_voxels, "bbox": P.bbox, "was_binary": P.was_binary,
                             "n_nonfinite": P.n_nonfinite, "value_info": P.value_info,
                             "header_info": P.header_info, "shape": P.shape})
    side = kwargs.get("expected_side")
    for tier in ("header", "fast", "full"):
        out[f"assess_{tier}"] = _try(lambda: qc.assess(P, tier=tier, **kwargs).to_dict())
    out["check"] = _try(lambda: [f.to_dict() for f in qc.check(P, **kwargs)])
    out["gap_evidence"] = _try(lambda: qc.gap_evidence(P))
    out["clusters"] = _try(lambda: dict(qc.clusters(P).items()))

    def fixed(fn):
        im, rep = fn()
        return {"img": _img_hash(im), "report": rep}
    out["fix"] = _try(lambda: fixed(lambda: qc.fix(P)))
    out["clean"] = _try(lambda: fixed(lambda: qc.clean(P)))
    out["clean_all"] = _try(lambda: fixed(lambda: qc.clean(P, strokes=True, outside=True)))
    out["repair_gaps"] = _try(lambda: fixed(lambda: qc.repair_gaps(P)))
    out["lesion_stats"] = _try(lambda: api.lesion_stats(P))
    for key in ("aal", "schaefer", "dk", "jhu", "arterial", "tissue"):
        out[f"overlap_{key}"] = _try(lambda: api.atlas_overlap(P, key))
        out[f"coverage_{key}"] = _try(lambda: api.atlas_coverage(P, key))
    out["net_schaefer"] = _try(lambda: api.network_rollup_for(P, "schaefer"))
    out["net_arterial"] = _try(lambda: api.network_rollup_for(P, "arterial"))
    out["lobes_aal"] = _try(lambda: api.lobe_overlap(P, "aal"))
    out["lobes_dk"] = _try(lambda: api.lobe_overlap(P, "dk"))
    out["tracts"] = _try(lambda: api.tract_proxy_overlap(P))
    out["tissue"] = _try(lambda: lqtpy.tissue.locate(P))
    out["watershed"] = _try(lambda: lqtpy.watershed.border_zones(P))
    out["structures"] = _try(lambda: lqtpy.structures.distances(P))
    out["cross"] = _try(lambda: lqtpy.structures.cross_section_load(P))
    out["mirror"] = _try(lambda: lqtpy.mirror.homologue(P))
    for prof in ("stroke", "small_vessel", "tumour", "research"):
        out[f"describe_{prof}"] = _try(lambda: (lambda d: {"marked": d.marked, "facts": d.facts})(
            lqtpy.describe(P, profile=prof, shape=True)))
    from lqtpy.measures import sensitivity, shape
    out["shape"] = _try(lambda: shape.compute(P))
    out["sensitivity"] = _try(lambda: sensitivity.compute(P) if hasattr(sensitivity, "compute") else None)
    out["fig_glass"] = _try(lambda: hashlib.sha1(figures.glass_brain_svg(P).encode()).hexdigest()[:16])
    out["fig_slices"] = _try(lambda: hashlib.sha1(figures.slices_svg(P).encode()).hexdigest()[:16])
    out["fig_thumb"] = _try(lambda: hashlib.sha1(figures.thumbnail_uri(P).encode()).hexdigest()[:16])
    out["fig_viewer"] = _try(lambda: figures.viewer_data(P))
    if real and int(hashlib.sha1(name.encode()).hexdigest(), 16) % 3 == 0:
        # The Registry's glass-brain PNG thumbnails (slow: a third of the real masks).
        from lqtpy.plotting import glass_brain_png, use_agg
        use_agg()
        for theme in ("light", "dark"):
            out[f"glass_png_{theme}"] = _try(lambda: hashlib.sha1(
                (glass_brain_png(img, theme=theme) or "").encode()).hexdigest()[:16])
    out["rank"] = _try(lambda: [{k: v for k, v in r.items() if k != "seconds"}
                                for r in qc.rank([P, img])])
    out["api_fresh"] = _try(lambda: {
        "stats": api.lesion_stats(img), "aal": api.atlas_overlap(img, "aal"),
        "net": api.network_rollup_for(img, "arterial"), "tracts": api.tract_proxy_overlap(img),
        "discon": api.tract_disconnection(img) if lqtpy.datasets.discon_index_available() else None})
    timings["calls"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    d = _fresh("case")
    out["analyze"] = _try(lambda: lqtpy.analyze_single(lqtpy.prepare(img, name=name), out_dir=d,
                                                        plots=True, extras="all", expected_side=side))
    out["analyze_files"] = _files(d)
    if real:
        # The Registry's own call: a .nii.gz path, the sensitivity extra, a report.
        d = _fresh("case_path")
        src = os.path.join(d, name + ".nii.gz")
        nib.save(img, src)
        out["analyze_path"] = _try(lambda: lqtpy.analyze_single(
            src, out_dir=os.path.join(d, "out"), plots=True, verbose=False, extras="sensitivity"))
        out["analyze_path_files"] = _files(os.path.join(d, "out"))
        out["fix_path"] = _try(lambda: fixed(lambda: qc.fix(src)))
    timings["analyze"] = time.perf_counter() - t0
    return out


def zoo_cases():
    import qc_zoo
    for c in qc_zoo.all_cases():
        kw = {k: v for k, v in (c.kwargs or {}).items() if k in ("expected_side", "mni", "context")}
        yield "zoo_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", c.name), c.img(), kw


def mask_cases(folder):
    """Real masks from a folder, named by a hash of the file name only."""
    import nibabel as nib
    import numpy as np
    for f in sorted(os.listdir(folder)):
        if f.lower().endswith((".nii", ".nii.gz")):
            img = nib.load(os.path.join(folder, f))
            img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
            yield "real_r" + hashlib.sha1(f.encode()).hexdigest()[:10], img, {}


def _write_inputs(cases, src, prefix):
    import nibabel as nib
    os.makedirs(src)
    for i, c in enumerate(cases):
        nib.save(c.img(), os.path.join(src, f"{prefix}{i:02d}.nii.gz"))


def record_group() -> dict:
    import lqtpy
    import qc_zoo
    cases = list(qc_zoo.clean_cases())[:8]
    d = _fresh("group")
    src = os.path.join(d, "in")
    _write_inputs(cases, src, "subj")
    meta = os.path.join(d, "meta.csv")
    with open(meta, "w", encoding="utf-8") as fh:
        fh.write("subject,group,age\n")
        for i in range(len(cases)):
            fh.write(f"subj{i:02d},{'A' if i % 2 else 'B'},{50 + i}\n")
    res = {}
    o1 = os.path.join(d, "out1")
    res["group1"] = _clean(lqtpy.analyze_group(src, out_dir=o1, plots=True, metadata=meta,
                                               group_by="group", compare="group", extras="all"))
    res["group1_files"] = _files(o1)
    o2 = os.path.join(d, "out2")
    res["group2"] = _clean(lqtpy.analyze_group(src, out_dir=o2, plots=True, anonymise=True,
                                               subject_reports=False))
    res["group2_files"] = _files(o2)
    return res


def record_cli(root) -> dict:
    import qc_zoo
    cases = list(qc_zoo.clean_cases())[:2] + list(qc_zoo.problem_cases())[:3]
    d = _fresh("cli")
    src = os.path.join(d, "in")
    _write_inputs(cases, src, "m")
    f0 = os.path.join(src, "m00.nii.gz")
    env = dict(os.environ, PYTHONPATH=root, PYTHONIOENCODING="utf-8", NO_COLOR="1")
    cmds = {"single": ["single", f0, "-o", os.path.join(d, "single")],
            "describe": ["describe", f0],
            "qc_file": ["qc", os.path.join(src, "m03.nii.gz")],
            "qc_folder": ["qc", src, "-o", os.path.join(d, "qcpage")],
            "group": ["group", src, "-o", os.path.join(d, "group")]}
    res = {}
    for k, args in cmds.items():
        p = subprocess.run([sys.executable, "-m", "lqtpy"] + args, capture_output=True, text=True,
                           env=env, cwd=d, encoding="utf-8", errors="replace")
        res[k] = {"rc": p.returncode, "stdout": norm_text(p.stdout),
                  "stderr_tail": norm_text(p.stderr)[-400:]}
    for sub in ("single", "qcpage", "group"):
        if os.path.isdir(os.path.join(d, sub)):
            res[sub + "_files"] = _files(os.path.join(d, sub))
    return res


def record(args) -> int:
    root = _setup(args.tree)
    import lqtpy
    os.makedirs(args.out, exist_ok=True)
    lqtpy.preload()
    what = args.what
    timings = {}
    gens = []
    if what in ("zoo", "all"):
        gens.append((zoo_cases(), False))
    if what == "masks" or (what == "all" and args.masks):
        if not args.masks:
            sys.exit("record masks needs --masks FOLDER")
        gens.append((mask_cases(args.masks), True))
    for gen, real in gens:
        for name, img, kw in gen:
            if args.only and args.only not in name:
                continue
            t = {}
            rec = record_case(name, img, kw, t, real=real)
            timings[name] = t
            fn = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:90] + ".json"
            with open(os.path.join(args.out, fn), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True, ensure_ascii=False)
            print(f"{name[:60]:60s} calls {t['calls']:6.2f}s analyze {t['analyze']:6.2f}s", flush=True)
    for part, fn in (("group", record_group), ("cli", lambda: record_cli(root))):
        if what in (part, "all") and not args.only:
            t0 = time.perf_counter()
            rec = fn()
            with open(os.path.join(args.out, f"_{part}.json"), "w", encoding="utf-8") as fh:
                json.dump(rec, fh, indent=1, sort_keys=True, ensure_ascii=False)
            timings["_" + part] = {"total": time.perf_counter() - t0}
            print(f"{part} {timings['_' + part]['total']:.2f}s", flush=True)
    with open(os.path.join(args.out, f"_timings_{what}.json"), "w", encoding="utf-8") as fh:
        json.dump(timings, fh, indent=1)
    shutil.rmtree(WORK, ignore_errors=True)
    return 0


def _walk(a, b, path, out):
    if type(a) is not type(b):
        out.append((path, repr(a)[:120], repr(b)[:120]))
    elif isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing" if k not in a else "present",
                            "missing" if k not in b else "present"))
            else:
                _walk(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append((path, f"len {len(a)}", f"len {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            _walk(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append((path, repr(a)[:120], repr(b)[:120]))


def diff(args) -> int:
    A, B = args.a, args.b
    files = sorted(f for f in os.listdir(A) if f.endswith(".json") and not f.startswith("_timings"))
    n_same, diffs = 0, {}
    for f in files:
        if not os.path.exists(os.path.join(B, f)):
            diffs[f] = [("file", "present", "missing")]
            continue
        with open(os.path.join(A, f), encoding="utf-8") as fa, open(os.path.join(B, f), encoding="utf-8") as fb:
            a, b = json.load(fa), json.load(fb)
        out = []
        _walk(a, b, "", out)
        if out:
            diffs[f] = out
        else:
            n_same += 1
    extra = sorted(f for f in os.listdir(B) if f.endswith(".json") and not f.startswith("_timings")
                   and f not in files)
    for f in extra:
        diffs[f] = [("file", "missing", "present")]
    print(f"{n_same}/{len(files)} recordings identical" + (f", {len(extra)} only in B" if extra else ""))
    for f, out in diffs.items():
        print(f"\n{f}: {len(out)} differences")
        for p, x, y in out[:12]:
            print(f"  {p}\n     A {x}\n     B {y}")
    return 0 if not diffs else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="golden.py", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("record", help="Record every output into a folder.")
    r.add_argument("out")
    r.add_argument("what", nargs="?", default="all", choices=("zoo", "group", "cli", "masks", "all"))
    r.add_argument("--tree", help="Import lqtpy from this checkout instead of this one.")
    r.add_argument("--masks", help="Folder of real masks (kept outside the repository).")
    r.add_argument("--only", help="Only cases whose name contains this text.")
    d = sub.add_parser("diff", help="Compare two recordings.")
    d.add_argument("a")
    d.add_argument("b")
    args = ap.parse_args(argv)
    return record(args) if args.cmd == "record" else diff(args)


if __name__ == "__main__":
    sys.exit(main())
