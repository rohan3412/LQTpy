"""Command-line interface.

    lqtpy single  lesion.nii.gz -o out           full report for one lesion
    lqtpy group   lesions/ -o out                group report + one report per subject
    lqtpy describe lesion.nii.gz                 print the plain-language description
    lqtpy qc      lesion.nii.gz                  print mask quality flags
    lqtpy repair  lesion.nii.gz -o fixed.nii.gz  interpolate sparse-slice gaps (opt-in)
    lqtpy clean   lesion.nii.gz -o clean.nii.gz  remove image-edge artefacts and specks (opt-in)

Errors print one line and exit with 2 (a problem with the input) or 1 (an
environment problem, such as missing data or an optional dependency).
"""
from __future__ import annotations

import argparse
import sys

from ._version import __version__
from .errors import LQTpyError, MissingDependencyError, DataUnavailableError

_PROFILES = ("stroke", "small_vessel", "tumour", "research")


def _add_analysis(p, *, group: bool):
    from .atlases import DEFAULT_ATLASES, list_atlases

    p.add_argument("--atlas", "-a", action="append", choices=list_atlases(),
                   help="Atlas to use (repeatable). Default: all bundled "
                        f"({', '.join(DEFAULT_ATLASES)}).")
    p.add_argument("--out", "-o", default=None, help="Output directory.")
    p.add_argument("--threshold", "-t", type=float, default=0.5,
                   help="A voxel is lesion when its value is greater than this (default 0.5).")
    p.add_argument("--profile", choices=_PROFILES, default="stroke",
                   help="Description style (default: stroke).")
    p.add_argument("--extras", default="all",
                   help='Extra measures: "all" (default), "none", or a comma list, e.g. "shape".')
    p.add_argument("--no-tracts", action="store_true", help="Skip white-matter tract overlap.")
    p.add_argument("--no-disconnection", action="store_true",
                   help="Skip streamline disconnection even if an index is installed.")
    p.add_argument("--no-qc", action="store_true", help="Skip mask quality checks.")
    p.add_argument("--no-describe", action="store_true", help="Skip the written description.")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    p.add_argument("--pdf", action="store_true",
                   help="Also save the report as PDF (needs Edge/Chrome/Chromium).")
    if group:
        p.add_argument("--no-subject-reports", action="store_true",
                       help="Don't write one report per subject.")
    else:
        p.add_argument("--expected-side", choices=("left", "right"),
                       help="Flag the lesion if most of it is on the other side.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lqtpy",
        description="Lesion morphometry, atlas overlap, quality checks and plain-language "
                    "descriptions, with self-contained HTML reports.")
    p.add_argument("--version", action="version", version=f"lqtpy {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single", help="Analyse one lesion and write a report.")
    s.add_argument("lesion", help="Path to a lesion .nii/.nii.gz")
    _add_analysis(s, group=False)

    g = sub.add_parser("group", help="Analyse a group of lesions.")
    g.add_argument("lesions", nargs="+", help="Lesion files, or a single directory of lesions.")
    _add_analysis(g, group=True)

    d = sub.add_parser("describe", help="Print the plain-language description of a lesion.")
    d.add_argument("lesion")
    d.add_argument("--profile", choices=_PROFILES, default="stroke")
    d.add_argument("--atlas", default="dk", choices=("dk", "aal"),
                   help="Atlas for the lobe/region hierarchy (default dk).")
    d.add_argument("--threshold", "-t", type=float, default=0.5)
    d.add_argument("--no-shape", action="store_true", help="Leave out shape descriptors.")

    q = sub.add_parser("qc", help="Print mask quality flags.")
    q.add_argument("lesion")
    q.add_argument("--threshold", "-t", type=float, default=0.5)
    q.add_argument("--expected-side", choices=("left", "right"))

    c = sub.add_parser("clean", help="Remove image-edge artefacts and specks into a new mask (opt-in).")
    c.add_argument("lesion")
    c.add_argument("--out", "-o", required=True, help="Path for the cleaned .nii.gz")
    c.add_argument("--keep-specks", action="store_true", help="Only remove image-edge artefacts.")
    c.add_argument("--threshold", "-t", type=float, default=0.5)

    r = sub.add_parser("repair", help="Interpolate sparse-slice gaps into a new mask (opt-in).")
    r.add_argument("lesion")
    r.add_argument("--out", "-o", required=True, help="Path for the repaired .nii.gz")
    r.add_argument("--threshold", "-t", type=float, default=0.5)
    return p


def _run(args) -> int:
    if args.cmd in ("single", "group"):
        from .analysis import analyze_group, analyze_single
        from .plotting import use_agg

        # The CLI is headless, so forcing Agg here is right; library callers
        # opt in themselves (lqtpy never forces a backend on their behalf).
        use_agg()
        common = dict(atlases=args.atlas, out_dir=args.out, threshold=args.threshold,
                      tracts=not args.no_tracts, plots=True, verbose=not args.quiet,
                      qc=not args.no_qc, describe=not args.no_describe, profile=args.profile,
                      extras=None if args.extras == "none" else args.extras,
                      disconnection=False if args.no_disconnection else "auto", pdf=args.pdf)
        if args.cmd == "single":
            res = analyze_single(args.lesion, expected_side=args.expected_side, **common)
        else:
            inp = args.lesions[0] if len(args.lesions) == 1 else args.lesions
            res = analyze_group(inp, subject_reports=not args.no_subject_reports, **common)
        print(res.get("report_path", "(no report written)"))
        return 0
    if args.cmd == "describe":
        from .describe import describe
        print(describe(args.lesion, profile=args.profile, atlas=args.atlas,
                       threshold=args.threshold, shape=not args.no_shape).text)
        return 0
    if args.cmd == "qc":
        from . import qc
        flags = qc.check(args.lesion, threshold=args.threshold, expected_side=args.expected_side)
        if not flags:
            print("No quality problems found.")
        for f in flags:
            print(f"[{f.severity}] {f.message}" + (f"  -> {f.suggestion}" if f.suggestion else ""))
        return 1 if any(f.severity == "error" for f in flags) else 0
    if args.cmd == "clean":
        import nibabel as nib
        from . import qc
        img, rep = qc.clean(args.lesion, specks=not args.keep_specks, threshold=args.threshold)
        if not rep["voxels_removed"]:
            print("Nothing to remove.")
            return 0
        nib.save(img, args.out)
        print(f"Removed {rep['removed_edge_fragments']} edge fragment(s) and "
              f"{rep['removed_specks']} speck(s), {rep['volume_removed_mm3']:.0f} mm³. Wrote {args.out}")
        return 0
    if args.cmd == "repair":
        import nibabel as nib
        from . import qc
        img, rep = qc.repair_gaps(args.lesion, threshold=args.threshold)
        if not rep["repaired"]:
            print(f"Nothing repaired: {rep.get('reason', 'no gaps between drawn slices')}.")
            return 0
        nib.save(img, args.out)
        print(f"Interpolated {rep['slices_interpolated']} {rep['plane']} slices, adding "
              f"{rep['volume_added_mm3']:.0f} mm³. Wrote {args.out}")
        return 0
    raise AssertionError(args.cmd)


def _safe_stdout() -> None:
    """Never crash on a console that can't encode "³" (Windows cp1252):
    unencodable characters are replaced instead of raising."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv=None) -> int:
    _safe_stdout()
    args = build_parser().parse_args(argv)
    try:
        return _run(args)
    except (MissingDependencyError, DataUnavailableError) as exc:
        print(f"lqtpy: {str(exc).splitlines()[0]}", file=sys.stderr)
        return 1
    except FileNotFoundError as exc:
        print(f"lqtpy: {exc}", file=sys.stderr)
        return 2
    except (LQTpyError, ValueError) as exc:
        print(f"lqtpy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
