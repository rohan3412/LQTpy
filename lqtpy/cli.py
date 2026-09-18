"""Command-line interface: ``python -m lqtpy single|group ...``."""
from __future__ import annotations

import argparse
import sys

from .analysis import analyze_single, analyze_group
from .atlases import DEFAULT_ATLASES, list_atlases
from .plotting import use_agg


def _add_common(p):
    p.add_argument("--atlas", "-a", action="append", choices=list_atlases(),
                   help="Atlas to use (repeatable). Default: all "
                        f"({', '.join(DEFAULT_ATLASES)}).")
    p.add_argument("--out", "-o", default=None, help="Output directory.")
    p.add_argument("--threshold", "-t", type=float, default=0.5,
                   help="Binarization threshold for non-binary inputs (default 0.5).")
    p.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output.")
    p.add_argument("--no-tracts", action="store_true",
                   help="Skip white-matter tract overlap (on by default).")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="lqtpy",
        description="Fast lesion morphometry + atlas-overlap reporting.")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("single", help="Analyse one lesion NIfTI.")
    s.add_argument("lesion", help="Path to a lesion .nii/.nii.gz")
    _add_common(s)

    g = sub.add_parser("group", help="Analyse a group of lesions.")
    g.add_argument("lesions", nargs="+",
                   help="Lesion files, or a single directory of lesions.")
    _add_common(g)
    return p


def main(argv=None) -> int:
    # The CLI is a headless program, so forcing the Agg backend before any
    # rendering happens is correct here. Library callers opt in themselves
    # via plotting.use_agg() -- lqtpy never forces this on their behalf.
    use_agg()
    args = build_parser().parse_args(argv)
    atlases = args.atlas or DEFAULT_ATLASES
    verbose = not args.quiet
    tracts = not args.no_tracts
    # The CLI's whole point is the self-contained HTML report, so it opts
    # into plots=True explicitly -- analyze_single/analyze_group default to
    # plots=False (no matplotlib/nilearn/pandas import) for in-process,
    # library-style callers such as a web handler.
    if args.cmd == "single":
        res = analyze_single(args.lesion, atlases=atlases, out_dir=args.out,
                             threshold=args.threshold, tracts=tracts,
                             plots=True, verbose=verbose)
    else:
        inp = args.lesions[0] if len(args.lesions) == 1 else args.lesions
        res = analyze_group(inp, atlases=atlases, out_dir=args.out,
                            threshold=args.threshold, tracts=tracts,
                            plots=True, verbose=verbose)
    print(res.get("report_path", "(no report written)"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
