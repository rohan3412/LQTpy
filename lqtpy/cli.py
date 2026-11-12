"""Command-line interface.

    lqtpy single   LESION [-o OUT]            full report for one lesion
    lqtpy group    FOLDER|FILES [-o OUT]      group report + one report per subject
    lqtpy describe LESION                     print the plain-language description
    lqtpy qc       LESION|FOLDER              quality flags (a folder gets an HTML page)
    lqtpy clean    LESION -o NEW              remove image-edge artefacts and specks (opt-in)
    lqtpy repair   LESION -o NEW              interpolate sparse-slice gaps (opt-in)
    lqtpy warp     NATIVE --transform T -o NEW  native-space mask -> MNI
    lqtpy rerun    REPORT.html                repeat the analysis a report records
    lqtpy watch    FOLDER -o OUT              rebuild the group report as masks arrive
    lqtpy serve                               upload page for colleagues without Python
    lqtpy fetch    suit                       download an optional atlas
    lqtpy new-plugin NAME                     starter package for an extra measure
    lqtpy config                              show the settings file in use
    lqtpy cache    [--clear|--prune]          where template maps are cached
    lqtpy doctor                              check this installation
    lqtpy demo                                report for a made-up lesion
    lqtpy install-shortcut                    Windows "Send to" entry that makes a report

Defaults come from ``lqtpy.toml`` (see :mod:`lqtpy.config`); options on the
command line win. Errors print one line and exit with 2 (a problem with
the input) or 1 (an environment problem, such as missing data or an
optional dependency). ``--quiet`` prints only the report path.
"""
from __future__ import annotations

import argparse
import os
import sys

from ._version import __version__
from .errors import DataUnavailableError, LQTpyError, MissingDependencyError

_PROFILES = ("stroke", "small_vessel", "tumour", "research")


def _table_atlases() -> "list[str]":
    """Atlases that make region tables (``tissue`` is internal)."""
    from .atlases import list_atlases
    return [a for a in list_atlases() if a != "tissue"]


def _transform(t):
    """One --transform stays a path (as before); several are an ANTs list."""
    if not t:
        return None
    return t[0] if len(t) == 1 else list(t)


def _jobs_arg(value):
    from ._jobs import parse_jobs
    try:
        return parse_jobs(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def _add_analysis(p, cfg, *, group: bool):
    from .atlases import DEFAULT_ATLASES
    atl = cfg.get("atlases") or DEFAULT_ATLASES
    p.add_argument("--atlas", "-a", action="append", choices=_table_atlases(), metavar="ATLAS",
                   help=f"Atlas to use (repeatable): {', '.join(_table_atlases())}. "
                        f"Default: {', '.join(atl)}.")
    p.add_argument("--out", "-o", default=None, help="Output folder." + (
        f" Default: {cfg['out_dir']}/<name>." if cfg.get("out_dir") else " Default: lqtpy_output/<name>."))
    p.add_argument("--threshold", "-t", type=float, default=cfg.get("threshold", 0.5),
                   help="A voxel is lesion when its value is greater than this (default %(default)s).")
    p.add_argument("--profile", choices=_PROFILES, default=cfg.get("profile", "stroke"),
                   help="Description style (default: %(default)s).")
    p.add_argument("--extras", default=cfg.get("extras", "none"),
                   help='Extra measures: "none" (default), "all", or a comma list, e.g. "shape".')
    p.add_argument("--no-tracts", action="store_true", help="Skip white-matter tract overlap.")
    p.add_argument("--no-disconnection", action="store_true",
                   help="Skip streamline disconnection even if an index is installed.")
    p.add_argument("--no-qc", action="store_true", help="Skip mask quality checks.")
    p.add_argument("--no-describe", action="store_true", help="Skip the written description.")
    p.add_argument("--no-viewer", action="store_true", help="Leave out the slice viewer (smaller report).")
    p.add_argument("--qc-examples", action="store_true",
                   help='Add a "What does this look like?" example picture under each quality '
                        "finding in the report.")
    p.add_argument("--flagged-voxels", action="store_true",
                   help='Add a "Show flagged voxels" switch to the slice viewer (off when the '
                        "report opens): the voxels quality findings are about, in colour.")
    p.add_argument("--anonymise", "--anonymize", nargs="?", const=True,
                   default=cfg.get("anonymise", False), metavar="KEY_FILE",
                   help="Write IDs instead of file names everywhere; the private key file "
                        "maps them back (default <out>/anonymisation_key.csv).")
    p.add_argument("--quiet", "-q", action="store_true", help="Print only the report path.")
    p.add_argument("--pdf", action="store_true", default=cfg.get("pdf", False),
                   help="Also save the report as PDF (needs Edge/Chrome/Chromium).")
    p.add_argument("--xlsx", action="store_true",
                   help="Also write the tables and the data dictionary as one Excel workbook "
                        "(needs openpyxl).")
    if group:
        p.add_argument("--pdf-bundle", action="store_true",
                       help="PDFs of the group and every subject, merged into one with bookmarks "
                            "(the merge needs pypdf).")
        p.add_argument("--no-subject-reports", action="store_true",
                       help="Don't write one report per subject.")
        p.add_argument("--metadata", "-m", help="CSV/TSV with one row per subject (age, condition, "
                                                "side, scores).")
        p.add_argument("--subject-column", help="Metadata column naming the subject.")
        p.add_argument("--group-by", help="Metadata column to summarise the cohort by.")
        p.add_argument("--compare", nargs="+", metavar="COLUMN [A B]",
                       help="Compare two groups: a metadata column with two values, or COLUMN A B.")
        p.add_argument("--side-column", default="auto",
                       help='Metadata column with the lesion (or symptom) side for the flip check; '
                            '"auto" finds one, "none" turns it off.')
        p.add_argument("--jobs", "-j", type=_jobs_arg, default=cfg.get("jobs", 1),
                       help='Worker processes (-1: one per CPU; "auto": one process for a few '
                            'files, else one per CPU less one). Default %(default)s.')
        p.add_argument("--bids", action="store_true",
                       help="The folder is a BIDS dataset: find its lesion masks (label-lesion / "
                            "desc-lesion), take metadata from participants.tsv, and write a BIDS "
                            "derivative (default <dataset>/derivatives/lqtpy).")
        p.add_argument("--incremental", action="store_true",
                       help="Reuse results from an earlier run into the same folder for unchanged files.")
    else:
        p.add_argument("--per-label", action="store_true",
                       help="A mask with several label values: one report per label "
                            "(default: the labels are merged into one lesion).")
        p.add_argument("--max-labels", type=int, default=20, metavar="N",
                       help="--per-label: refuse masks with more than N label values (default 20).")
        p.add_argument("--expected-side", choices=("left", "right"),
                       help="Flag the lesion if most of it is on the other side.")
        p.add_argument("--native", help="The same lesion as a native-space mask: drawing checks "
                                        "(gaps, single slice, outline) run on it.")
        p.add_argument("--transform", action="append",
                       help="LESION is a native-space mask: warp it to MNI with this "
                            "4x4 matrix (text file, native mm -> MNI mm), deformation field, "
                            "or ANTs transforms (repeat in ANTs order: --transform 1Warp.nii.gz "
                            "--transform 0GenericAffine.mat), then analyse.")


def build_parser(cfg: "dict | None" = None) -> argparse.ArgumentParser:
    cfg = cfg or {}
    p = argparse.ArgumentParser(
        prog="lqtpy",
        description="Lesion morphometry, atlas overlap, quality checks and plain-language "
                    "descriptions, with self-contained HTML reports.",
        epilog="Settings file: lqtpy.toml (see `lqtpy config`). Docs: README.md.")
    p.add_argument("--version", action="version", version=f"lqtpy {__version__}")
    p.add_argument("--config", help="Settings file to use instead of the lqtpy.toml search.")
    sub = p.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    s = sub.add_parser("single", help="Analyse one lesion and write a report.")
    s.add_argument("lesion", help="Lesion file (.nii, .nii.gz, .mgz, .nrrd, .dcm)")
    _add_analysis(s, cfg, group=False)

    g = sub.add_parser("group", help="Analyse a group of lesions.")
    g.add_argument("lesions", nargs="+", help="Lesion files, or one folder of lesions.")
    _add_analysis(g, cfg, group=True)

    d = sub.add_parser("describe", help="Print the plain-language description of a lesion.")
    d.add_argument("lesion")
    d.add_argument("--profile", choices=_PROFILES, default=cfg.get("profile", "stroke"))
    d.add_argument("--atlas", default="dk", choices=("dk", "aal"),
                   help="Atlas for the lobe/region hierarchy (default dk).")
    d.add_argument("--threshold", "-t", type=float, default=cfg.get("threshold", 0.5))
    d.add_argument("--shape", action="store_true", help="Add shape descriptors (an extra).")
    d.add_argument("--no-shape", action="store_true", help=argparse.SUPPRESS)  # old flag

    q = sub.add_parser("qc", help="Quality flags for a lesion, or a page for a whole folder.")
    q.add_argument("lesion", help="A lesion file or a folder of them.")
    q.add_argument("--threshold", "-t", type=float, default=cfg.get("threshold", 0.5))
    q.add_argument("--expected-side", choices=("left", "right"))
    q.add_argument("--out", "-o", help="Folder: where to write the HTML page "
                                       "(default <folder>/lqtpy_qc.html).")
    q.add_argument("--recursive", "-r", action="store_true", help="Folder: include subfolders.")
    q.add_argument("--clean-to", metavar="NEW_FOLDER",
                   help="Folder: write cleaned copies of flagged masks here (never overwrites).")
    q.add_argument("--json", nargs="?", const="-", metavar="FILE",
                   help="Print the graded result as JSON (or write it to FILE); for a folder, "
                        "every mask plus a summary.")
    q.add_argument("--tier", choices=("header", "fast", "full"), default="full",
                   help="header: header only (no voxel read); fast: skip the slower checks; "
                        "full (default): everything.")
    q.add_argument("--context", metavar="JSON_FILE",
                   help="Drawing context from a drawing tool (drawn slices, drawing plane, base).")
    q.add_argument("--jobs", "-j", type=_jobs_arg, default=cfg.get("jobs", 1),
                   help='Folder (with --json): worker processes, or "auto".')
    q.add_argument("--export-problems", metavar="DIR",
                   help="Lesion file: write one overlay mask per finding into DIR, to open over "
                        "the lesion in MRIcroGL or ITK-SNAP while fixing.")

    c = sub.add_parser("clean", help="Remove image-edge artefacts and specks into a new mask (opt-in).")
    c.add_argument("lesion")
    c.add_argument("--out", "-o", required=True, help="Path for the cleaned .nii.gz")
    c.add_argument("--keep-specks", action="store_true", help="Only remove image-edge artefacts.")
    c.add_argument("--strokes", action="store_true",
                   help="Also remove thin strokes (lines or sheets drawn by mistake).")
    c.add_argument("--threshold", "-t", type=float, default=0.5)

    r = sub.add_parser("repair", help="Interpolate sparse-slice gaps into a new mask (opt-in).")
    r.add_argument("lesion")
    r.add_argument("--out", "-o", required=True, help="Path for the repaired .nii.gz")
    r.add_argument("--threshold", "-t", type=float, default=0.5)

    w = sub.add_parser("warp", help="Warp a native-space mask to MNI.")
    w.add_argument("lesion")
    w.add_argument("--transform", required=True, action="append",
                   help="4x4 matrix text file (native mm -> MNI mm, RAS), deformation field image, "
                        "or ANTs transforms (repeat in ANTs order).")
    w.add_argument("--out", "-o", required=True, help="Path for the MNI .nii.gz")
    w.add_argument("--threshold", "-t", type=float, default=0.5)

    rr = sub.add_parser("rerun", help="Repeat the analysis recorded in a report.")
    rr.add_argument("report", help="An LQTpy report .html")
    rr.add_argument("--out", "-o", help="Output folder (default: 'rerun' next to the report).")
    rr.add_argument("--search", action="append", metavar="FOLDER",
                    help="Find moved inputs here by checksum (repeatable).")
    rr.add_argument("--allow-changed", action="store_true",
                    help="Run even if an input file has changed since the report.")
    rr.add_argument("--quiet", "-q", action="store_true")

    wa = sub.add_parser("watch", help="Rebuild a group report whenever masks are added or changed.")
    wa.add_argument("folder")
    wa.add_argument("--out", "-o", required=True)
    wa.add_argument("--interval", type=float, default=10.0, help="Seconds between checks (default 10).")
    wa.add_argument("--metadata", "-m")
    wa.add_argument("--profile", choices=_PROFILES, default=cfg.get("profile", "stroke"))
    wa.add_argument("--jobs", "-j", type=_jobs_arg, default=cfg.get("jobs", 1))

    sv = sub.add_parser("serve", help="Upload page: colleagues get a report without Python.")
    sv.add_argument("--host", default="127.0.0.1",
                    help="Address to listen on (default 127.0.0.1: this computer only).")
    sv.add_argument("--port", type=int, default=8765)
    sv.add_argument("--max-mb", type=int, default=300, help="Largest upload in MB (default 300).")
    sv.add_argument("--open", action="store_true", help="Open the page in the browser.")

    f = sub.add_parser("fetch", help="Download an optional atlas.")
    f.add_argument("what", choices=("suit",), help="suit: cerebellar lobules (Diedrichsen 2009).")

    npl = sub.add_parser("new-plugin", help="Write a starter package for an extra measure.")
    npl.add_argument("name", help="Measure name, e.g. fractal")
    npl.add_argument("--dir", default=".", help="Where to create lqtpy-<name>/ (default: here).")

    sub.add_parser("config", help="Show the settings file in use and its values.")
    ca = sub.add_parser("cache", help="Show (or clear) the template-map cache.")
    ca.add_argument("--clear", action="store_true", help="Delete this version's cache.")
    ca.add_argument("--prune", action="store_true",
                    help="Delete the caches of older lqtpy versions (never this one's).")

    sub.add_parser("doctor", help="Check this installation and say how to fix each gap.")
    dm = sub.add_parser("demo", help="Make a report for a synthetic lesion and open it.")
    dm.add_argument("--out", "-o", help="Output folder (default: a temporary folder).")
    dm.add_argument("--no-open", action="store_true", help="Don't open the report in the browser.")
    sc = sub.add_parser("install-shortcut",
                        help="Windows: add a 'Send to' entry in Explorer that makes a report.")
    sc.add_argument("--remove", action="store_true", help="Remove the entry again.")
    return p


def _out_dir(args, cfg, name):
    if args.out:
        return args.out
    if cfg.get("out_dir"):
        return os.path.join(cfg["out_dir"], name)
    return None


def _examples_style():
    from .report.style import ReportStyle
    return ReportStyle(qc_examples=True)


def _analysis_kwargs(args, cfg) -> dict:
    extras = None if args.extras in (None, "", "none") else args.extras
    return dict(atlases=args.atlas or cfg.get("atlases"), threshold=args.threshold,
                tracts=not args.no_tracts and cfg.get("tracts", True), plots=True,
                verbose=not args.quiet, qc=not args.no_qc, describe=not args.no_describe,
                profile=args.profile, extras=extras,
                disconnection=False if (args.no_disconnection or cfg.get("disconnection") is False) else "auto",
                viewer=not args.no_viewer and cfg.get("viewer", True),
                flagged_voxels=args.flagged_voxels,
                anonymise=args.anonymise,
                **({"report_style": _examples_style()} if args.qc_examples else {}))


def _run(args, cfg) -> int:
    from . import _term
    if args.cmd in ("single", "group"):
        from .analysis import analyze_group, analyze_single
        kw = _analysis_kwargs(args, cfg)
        if args.cmd == "single" and args.per_label:
            from .formats import strip_extension
            from .labels import analyze_labels
            if args.native or args.transform:
                raise ValueError("--per-label works on MNI masks (not with --native or --transform)")
            name = strip_extension(os.path.basename(args.lesion))
            out = analyze_labels(args.lesion, out_dir=_out_dir(args, cfg, name), name=name,
                                 max_labels=args.max_labels,
                                 expected_side=args.expected_side, pdf=args.pdf, **kw)
            for v, r in out["labels"].items():
                if args.xlsx and r.get("report_path"):
                    from .xlsx import write_workbook
                    write_workbook(os.path.dirname(r["report_path"]))
                print(_term.path(r["report_path"]) if args.quiet else
                      f"label {v}: {_term.path(r.get('report_path', ''))}")
            if not args.quiet and out["index_path"]:
                print(f"Index of labels: {_term.path(out['index_path'])}")
            return 0
        if args.cmd == "single":
            from .formats import strip_extension
            res = analyze_single(args.lesion, out_dir=_out_dir(args, cfg, strip_extension(
                os.path.basename(args.lesion))), expected_side=args.expected_side, pdf=args.pdf,
                native=args.native, transform=_transform(args.transform), **kw)
        else:
            inp = args.lesions[0] if len(args.lesions) == 1 else args.lesions
            bids_masks = None
            if args.bids:
                from . import bids
                if len(args.lesions) != 1 or not os.path.isdir(args.lesions[0]):
                    raise ValueError("--bids takes one folder: the BIDS dataset")
                if args.anonymise:
                    raise ValueError("--bids keeps the dataset's sub-<id> names; "
                                     "leave out --anonymise")
                root = args.lesions[0]
                if not args.out:
                    args.out = os.path.join(root, "derivatives", "lqtpy")
                bids_masks = bids.find_masks(root, exclude=args.out)
                if not bids_masks:
                    raise ValueError(f"no lesion masks (label-lesion, desc-lesion) found in {root}")
                inp = [(n, path) for n, path, _ in bids_masks]
                if not args.metadata:
                    args.metadata = bids.participants(root, bids_masks) or None
            compare = None
            if args.compare:
                if len(args.compare) not in (1, 3):
                    raise ValueError("--compare takes COLUMN, or COLUMN A B")
                compare = args.compare[0] if len(args.compare) == 1 else tuple(args.compare)
            res = analyze_group(inp, out_dir=_out_dir(args, cfg, "group"),
                                subject_reports=not args.no_subject_reports and cfg.get("subject_reports", True),
                                pdf="bundle" if args.pdf_bundle else args.pdf,
                                metadata=args.metadata, subject_column=args.subject_column,
                                group_by=args.group_by, compare=compare,
                                side_column=None if args.side_column == "none" else args.side_column,
                                n_jobs=args.jobs, incremental=args.incremental, **kw)
            if bids_masks is not None:
                from . import bids
                files = bids.write_derivatives(args.lesions[0], args.out, bids_masks, res)
                if not args.quiet:
                    print(f"BIDS derivative: {_term.path(args.out)}  ({len(files)} files)")
        if args.xlsx:
            from .xlsx import write_workbook
            if not res.get("report_path"):
                raise ValueError("--xlsx needs the tables the run writes (not with plots off)")
            book = write_workbook(os.path.dirname(res["report_path"]))
            if not args.quiet:
                print(f"Excel workbook: {_term.path(book)}")
        if args.quiet:
            print(_term.path(res.get("report_path", "")) if res.get("report_path") else "(no report written)")
        return 0
    if args.cmd == "describe":
        from .describe import describe
        print(describe(args.lesion, profile=args.profile, atlas=args.atlas,
                       threshold=args.threshold, shape=args.shape and not args.no_shape).text)
        return 0
    if args.cmd == "qc":
        from . import qc
        if os.path.isdir(args.lesion):
            return _qc_folder(args)
        ctx = None
        if args.context:
            import json as _json
            with open(args.context, encoding="utf-8") as fh:
                ctx = _json.load(fh)
        kw = {} if args.tier == "header" else dict(threshold=args.threshold,
                                                    expected_side=args.expected_side, context=ctx)
        if args.export_problems:
            from .qc_tools import export_problems
            files = export_problems(args.lesion, args.export_problems, threshold=args.threshold,
                                    context=ctx)
            for f in files:
                print(f"{f['id']:<14} {f['kind']:<8} {f['voxels']:>9,} voxels  {_term.path(f['path'])}")
            if not files:
                print("No finding marks voxels: nothing to export.")
        res = qc.assess(args.lesion, tier=args.tier, **kw)
        if args.json:
            text = res.to_json(indent=1)
            if args.json == "-":
                print(text)
            else:
                with open(args.json, "w", encoding="utf-8") as fh:
                    fh.write(text)
            return 1 if res.worst == "error" else 0
        _term.header(f"quality of {os.path.basename(args.lesion)}")
        _term.field("Quality", _term.quality(res.worst if res.worst != "ok" else "ok")
                    + _term.paint(f" · {res.verdict} · score {res.score}", "dim"))
        for f in res.findings:
            label = _term.quality(f.severity) + (_term.paint(f", {f.grade}", "dim") if f.grade else "")
            where = ""
            if f.locations:
                loc = f.locations[0]
                where = (f" [{loc['plane']} slice {loc['index']}]" if loc["kind"] == "slice"
                         else f" [at {', '.join(f'{v:.0f}' for v in loc['mm'])} mm]")
            _term.field("", f"{label}  {f.message}{_term.paint(where, 'dim')}"
                        + (_term.paint(f"  -> {f.suggestion}", "dim") if f.suggestion else ""))
        return 1 if res.worst == "error" else 0
    if args.cmd == "clean":
        import nibabel as nib
        from . import qc
        img, rep = qc.clean(args.lesion, specks=not args.keep_specks, strokes=args.strokes,
                            threshold=args.threshold)
        if not rep["voxels_removed"]:
            print("Nothing to remove.")
            return 0
        _refuse_overwrite(args.lesion, args.out)
        nib.save(img, args.out)
        print(f"Removed {rep['removed_edge_fragments']} edge fragment(s), "
              f"{rep['removed_specks']} speck(s) and {rep.get('removed_strokes', 0)} stroke(s), "
              f"{rep['volume_removed_mm3']:.0f} mm³. "
              f"Wrote {_term.path(args.out)}")
        return 0
    if args.cmd == "repair":
        import nibabel as nib
        from . import qc
        img, rep = qc.repair_gaps(args.lesion, threshold=args.threshold)
        if not rep["repaired"]:
            print(f"Nothing repaired: {rep.get('reason', 'no gaps between drawn slices')}.")
            return 0
        _refuse_overwrite(args.lesion, args.out)
        nib.save(img, args.out)
        print(f"Interpolated {rep['slices_interpolated']} {rep['plane']} slices, adding "
              f"{rep['volume_added_mm3']:.0f} mm³. Wrote {_term.path(args.out)}")
        return 0
    if args.cmd == "warp":
        import nibabel as nib
        from .native import warp_to_mni
        _refuse_overwrite(args.lesion, args.out)
        img = warp_to_mni(args.lesion, _transform(args.transform), threshold=args.threshold)
        nib.save(img, args.out)
        print(f"Wrote {_term.path(args.out)}")
        return 0
    if args.cmd == "rerun":
        from .repro import rerun
        res = rerun(args.report, out_dir=args.out, search=args.search,
                    allow_changed=args.allow_changed, verbose=not args.quiet)
        if args.quiet:
            print(_term.path(res.get("report_path", "")))
        return 0
    if args.cmd == "watch":
        from .watch import watch
        watch(args.folder, args.out, interval=args.interval, metadata=args.metadata,
              profile=args.profile, n_jobs=args.jobs)
        return 0
    if args.cmd == "serve":
        from .web import serve
        serve(args.host, args.port, max_mb=args.max_mb, open_browser=args.open)
        return 0
    if args.cmd == "fetch":
        from .datasets import fetch_suit
        d = fetch_suit()
        print(f"Downloaded the cerebellar lobule atlas to {_term.path(d)}. Use it with --atlas suit.")
        return 0
    if args.cmd == "new-plugin":
        from .scaffold import new_plugin
        root = new_plugin(args.name, args.dir)
        print(f"Wrote {_term.path(root)}\n  Next: pip install -e {_term.path(root)}\n"
              f"        lqtpy single lesion.nii.gz --extras {args.name}")
        return 0
    if args.cmd == "config":
        from .config import _user_path
        _term.header("settings")
        _term.field("File", _term.path(cfg["_path"]) if cfg.get("_path") else
                    "none found (./lqtpy.toml, LQTPY_CONFIG, " + _term.path(_user_path()) + ")")
        for k, v in cfg.items():
            if k != "_path":
                _term.field(k, str(v))
        return 0
    if args.cmd == "doctor":
        from .doctor import run_checks
        checks = run_checks()
        for status, what, fix in checks:
            _term.field(_term.quality({"ok": "ok", "note": "info", "problem": "error"}[status]),
                        what + (_term.paint(f"  -> {fix}", "dim") if fix else ""))
        bad = sum(1 for c in checks if c[0] == "problem")
        notes = sum(1 for c in checks if c[0] == "note")
        print(f"\n{bad} problem(s), {notes} optional part(s) missing.")
        return 1 if bad else 0
    if args.cmd == "demo":
        from .demo import make_demo
        report = make_demo(args.out, open_browser=not args.no_open, verbose=False)
        print(f"Demo report (a made-up lesion): {_term.path(report)}")
        return 0
    if args.cmd == "install-shortcut":
        from . import shortcut
        if args.remove:
            gone = shortcut.remove()
            print(f"Removed {_term.path(gone)}" if gone else "No Send to entry to remove.")
        else:
            print(f"Added {_term.path(shortcut.install())}\n"
                  "Right-click a lesion file > Send to > LQTpy report.")
        return 0
    if args.cmd == "cache":
        from . import _diskcache
        d = _diskcache.cache_dir()
        if args.prune:
            gone = _diskcache.prune()
            if not d:
                print("The disk cache is off.")
            elif not gone:
                print("No caches of older lqtpy versions to remove.")
            else:
                for p, size in gone:
                    print(f"Removed {_term.path(p)}  ({size / 2**20:.0f} MB)")
                print(f"Freed {sum(s for _, s in gone) / 2**20:.0f} MB.")
        elif args.clear:
            _diskcache.clear()
            print(f"Cleared {_term.path(d)}" if d else "The disk cache is off.")
        else:
            size = _diskcache._tree_size(d) if d and os.path.isdir(d) else 0
            print(f"{_term.path(d)}  ({size / 2**20:.0f} MB)" if d else "The disk cache is off (LQTPY_NO_DISK_CACHE).")
            old = _diskcache.old_versions()
            if old:
                print(f"Caches of {len(old)} older version(s): {sum(s for _, s in old) / 2**20:.0f} MB. "
                      f"`lqtpy cache --prune` removes them.")
        return 0
    raise AssertionError(args.cmd)


def _refuse_overwrite(src, out):
    if os.path.exists(out) and os.path.samefile(src, out):
        raise ValueError("The output must be a new file; lqtpy never overwrites the input mask.")


def _qc_folder(args) -> int:
    from . import _term, qc
    from .report.qcpage import render_qc_folder
    if args.json:
        import json as _json
        res = qc.assess_folder(args.lesion, tier=args.tier, recursive=args.recursive,
                               jobs=args.jobs, **({} if args.tier == "header" else
                                                  {"threshold": args.threshold}))
        text = _json.dumps(res, indent=1, default=str)
        if args.json == "-":
            print(text)
        else:
            with open(args.json, "w", encoding="utf-8") as fh:
                fh.write(text)
        return 1 if res["summary"]["verdicts"].get("unusable") else 0
    rows = qc.check_folder(args.lesion, threshold=args.threshold, recursive=args.recursive)
    cleaned = None
    if args.clean_to:
        cleaned = qc.clean_folder(args.lesion, args.clean_to, threshold=args.threshold)
    out = args.out or os.path.join(args.lesion, "lqtpy_qc.html")
    if os.path.isdir(out):
        out = os.path.join(out, "lqtpy_qc.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(render_qc_folder(rows, os.path.abspath(args.lesion), cleaned))
    counts = {k: sum(1 for r in rows if r["worst"] == k) for k in ("ok", "info", "warn", "error")}
    _term.header(f"quality of {len(rows)} masks")
    _term.field("OK", str(counts["ok"] + counts["info"]))
    _term.field(_term.quality("warn"), str(counts["warn"]))
    _term.field(_term.quality("error"), str(counts["error"]))
    if cleaned is not None:
        n = sum(1 for c in cleaned if c["status"] == "written")
        _term.field("Cleaned", f"{n} copies in {_term.path(args.clean_to)}")
    _term.field("Page", _term.path(out))
    return 1 if counts["error"] else 0


def _safe_stdout() -> None:
    """Never crash on a console that can't encode "³": piped output is UTF-8,
    and anything unencodable is replaced instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8", errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_stdout()
    argv = list(sys.argv[1:] if argv is None else argv)
    cfg_path = None
    if "--config" in argv:
        i = argv.index("--config")
        cfg_path = argv[i + 1] if i + 1 < len(argv) else None
    try:
        from .config import load_config
        cfg, used = load_config(cfg_path)
        cfg["_path"] = used
    except (ValueError, OSError) as exc:
        if "doctor" not in argv:           # doctor reports a bad settings file instead
            print(f"lqtpy: settings file: {exc}", file=sys.stderr)
            return 2
        cfg = {"_path": None}
    args = build_parser(cfg).parse_args(argv)
    try:
        return _run(args, cfg)
    except KeyboardInterrupt:
        return 130
    except (MissingDependencyError, DataUnavailableError) as exc:
        print(f"lqtpy: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"lqtpy: {exc}", file=sys.stderr)
        return 2
    except (LQTpyError, ValueError, FileExistsError) as exc:
        print(f"lqtpy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
