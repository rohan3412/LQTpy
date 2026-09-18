"""Build the real per-tract streamline disconnection index from the HCP-842
population tractography atlas (70 named ``*.trk.gz`` bundles + a
``tract_info.csv`` giving each tract's fixed reference streamline count).

The built ``.npz`` is NOT committed to this repo (~18 MB) and is not
package data -- it's meant to be built once, locally, into a per-user data
home (``lqtpy.datasets.get_data_home()``, overridable with
``LQTPY_DATA_HOME``) and loaded from there by
``lqtpy.api.tract_disconnection``.

Usage
-----
    python tools/build_discon_index.py --source <All_Tracts dir> --out <index path>

``<All_Tracts dir>`` is a directory containing the 70 ``{short_code}.trk.gz``
bundle files plus ``tract_info.csv`` (columns: ``tract_names``,
``tract_counts``, ``tract_pathways``). No path under this tool is
hard-coded -- pass whatever directory you have the atlas in.

################################################################################
# THE ~120 MM TRAP -- see lqtpy/_discon_index.py's module docstring.         #
# This tool is the ONE place that reads raw TRK points and must apply the   #
# origin shift before any voxel lookup. Do not "simplify" this by loading   #
# via `reference='same', to_space=Space.RASMM)` -- see the warning in       #
# lqtpy/_discon_index.py for why that's the most likely way to silently     #
# reintroduce a ~120 mm error.                                              #
################################################################################
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time

import numpy as np
import nibabel as nib

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from lqtpy._discon_index import (TRACT_AFFINE, TRACT_ATLAS_ORIGIN_SHIFT,
                                 TRACT_SHAPE, build_index, save_index)


def _read_tract_info(source_dir: str):
    """Read tract_info.csv -> ordered list of (short_code, expected_count).

    Sorted by short_code so the index's tract-id assignment (and therefore
    every downstream array indexed by tract id) is in a stable, predictable
    order independent of the CSV's own row order or the filesystem's
    directory-listing order.
    """
    path = os.path.join(source_dir, "tract_info.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"tract_info.csv not found under '{source_dir}'")
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((row["tract_names"], int(row["tract_counts"])))
    rows.sort(key=lambda r: r[0])
    return rows


def _load_bundle_streamlines(source_dir: str, short_code: str):
    """Load one bundle's raw (un-shifted) points, one array per streamline.

    Uses plain ``nibabel.streamlines.load`` -- no dipy needed to build this
    index. Returns points as float64 in the trk's own (translation-less)
    world, i.e. NOT yet corrected for the ~120 mm trap; the shift is
    applied by the caller (see the module-level warning above).
    """
    trk_path = os.path.join(source_dir, f"{short_code}.trk.gz")
    if not os.path.exists(trk_path):
        raise FileNotFoundError(f"Bundle file not found: '{trk_path}'")
    trk = nib.streamlines.load(trk_path, lazy_load=False)
    return [np.asarray(s, dtype=np.float64) for s in trk.streamlines]


def build(source_dir: str, *, verbose: bool = True):
    """Build the index dict (see ``lqtpy._discon_index.build_index``) from
    ``source_dir``. Validates every bundle's streamline count against
    ``tract_info.csv`` and fails loudly (``ValueError``) on any mismatch."""
    tract_info = _read_tract_info(source_dir)

    tract_streamlines = []
    mismatches = []
    for short_code, expected_count in tract_info:
        raw_streamlines = _load_bundle_streamlines(source_dir, short_code)
        actual_count = len(raw_streamlines)
        if actual_count != expected_count:
            mismatches.append((short_code, expected_count, actual_count))
            continue
        # THE SHIFT: apply the validated ~120 mm origin correction here,
        # once, before any point ever reaches a voxel lookup. See the trap
        # warning at the top of this file and in lqtpy/_discon_index.py.
        shifted = [pts + TRACT_ATLAS_ORIGIN_SHIFT for pts in raw_streamlines]
        tract_streamlines.append((short_code, shifted))
        if verbose:
            print(f"  {short_code}: {actual_count} streamlines (matches tract_info.csv)")

    if mismatches:
        lines = "\n".join(
            f"  {code}: tract_info.csv says {exp}, bundle file has {act}"
            for code, exp, act in mismatches)
        raise ValueError(
            f"Streamline count mismatch for {len(mismatches)} bundle(s) -- "
            f"refusing to build a silently-wrong index:\n{lines}"
        )

    index = build_index(tract_streamlines, TRACT_SHAPE, TRACT_AFFINE)
    return index


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", required=True,
                        help="Directory with the 70 *.trk.gz bundles + tract_info.csv "
                             "(e.g. the HCP-842 atlas's All_Tracts directory).")
    parser.add_argument("--out", required=True,
                        help="Output .npz path for the built index.")
    parser.add_argument("--no-compress", action="store_true",
                        help="Save uncompressed (np.savez instead of np.savez_compressed). "
                             "Larger on disk, faster to load.")
    args = parser.parse_args(argv)

    t0 = time.time()
    print(f"Building disconnection index from '{args.source}' ...")
    index = build(args.source)
    build_seconds = time.time() - t0

    n_tracts = len(index["tract_codes"])
    print(f"\nBuilt index: {index['n_streamlines']} streamlines across {n_tracts} tracts "
         f"in {build_seconds:.1f}s.")
    print(f"Chosen layout: '{index['layout']}' "
         f"(full-grid would be {index['_full_bytes'] / 1e6:.2f} MB, "
         f"occupied-voxel would be {index['_occupied_bytes'] / 1e6:.2f} MB, in-memory arrays only)")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir, exist_ok=True)

    t1 = time.time()
    save_index(args.out, index, compress=not args.no_compress)
    save_seconds = time.time() - t1
    disk_bytes = os.path.getsize(args.out)
    print(f"Wrote '{args.out}': {disk_bytes / 1e6:.2f} MB on disk "
         f"({'compressed' if not args.no_compress else 'uncompressed'}) in {save_seconds:.1f}s.")
    print(f"Total build time: {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
