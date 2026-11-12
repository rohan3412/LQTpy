"""BIDS in, BIDS derivatives out (``lqtpy group DATASET --bids``, opt-in).

In: a BIDS dataset folder. Lesion masks are files named with
``label-lesion`` or ``desc-lesion`` (``*_label-lesion_roi.nii.gz`` in raw
data, ``*_desc-lesion_mask.nii.gz`` in derivatives), or with the suffix
``_lesion``, anywhere in the dataset (``derivatives/`` included, lqtpy's own
output excluded). They must already be in MNI space, as for any lqtpy run.
``participants.tsv`` supplies the metadata, matched on ``participant_id``.

Out: a derivative dataset, by default ``<dataset>/derivatives/lqtpy``:

- ``dataset_description.json`` (``DatasetType: derivative``)
- per mask: ``sub-<id>/[ses-<id>/]anat/<entities>_space-MNI152NLin6Asym_desc-<table>_stats.tsv``
  (every table of that subject, tab-separated) and ``..._desc-morphometry_stats.json``
- lqtpy's usual group report and per-subject reports (``subjects/``),
  listed in ``.bidsignore`` so the BIDS validator skips them.
"""
from __future__ import annotations

import csv
import json
import os
import re

__all__ = ["find_masks", "participants", "write_derivatives", "SPACE"]

#: The FSL MNI152 1 mm template lqtpy's atlases are on (BIDS template name).
SPACE = "MNI152NLin6Asym"
_MASK = re.compile(r"(^|_)(label|desc)-lesion(_|$)|_lesion$", re.I)
_NIFTI = (".nii.gz", ".nii")


def _entities(stem: str) -> "list[tuple[str, str]]":
    """``sub-01_ses-2_T1w_label-lesion_roi`` -> [("sub","01"), ("ses","2"), ...]
    (key-value parts only; the suffix and bare words are left out)."""
    return [tuple(p.split("-", 1)) for p in stem.split("_") if "-" in p]


def _stem(fname: str) -> str:
    for ext in _NIFTI:
        if fname.lower().endswith(ext):
            return fname[:-len(ext)]
    return fname


def find_masks(root: str, *, exclude: "str | None" = None) -> "list[tuple[str, str, dict]]":
    """``[(name, path, entities)]`` for the lesion masks in the dataset.
    ``name`` is ``sub-X`` (plus ``_ses-Y`` and further entities only where
    needed to tell one subject's masks apart)."""
    ex = os.path.normcase(os.path.abspath(exclude)) if exclude else None
    found = []
    for r, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".")
                         and (ex is None or os.path.normcase(os.path.abspath(os.path.join(r, d))) != ex))
        for f in sorted(files):
            if not f.lower().endswith(_NIFTI):
                continue
            stem = _stem(f)
            if not _MASK.search(stem) or not stem.startswith("sub-"):
                continue
            ents = dict(_entities(stem))
            found.append((os.path.join(r, f), ents))
    keep = [k for k in ("sub", "ses", "acq", "run", "rec", "space") ]
    names = []
    for depth in range(1, len(keep) + 1):
        names = ["_".join(f"{k}-{e[k]}" for k in keep[:depth] if k in e) for _, e in found]
        if len(set(names)) == len(names):
            break
    else:
        names = [_stem(os.path.basename(p)) for p, _ in found]       # last resort: the file name
    return [(n, p, e) for n, (p, e) in zip(names, found)]


def participants(root: str, masks) -> "dict[str, dict]":
    """Metadata for :func:`lqtpy.analyze_group`: ``participants.tsv`` rows,
    keyed by mask name (every mask of a participant gets that row)."""
    path = os.path.join(root, "participants.tsv")
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh, dialect="excel-tab"))
    by_id = {}
    for r in rows:
        pid = str(r.get("participant_id", "")).strip()
        if pid:
            by_id[pid if pid.startswith("sub-") else f"sub-{pid}"] = {
                k: ("" if v in (None, "n/a") else v) for k, v in r.items() if k != "participant_id"}
    return {name: by_id[f"sub-{e['sub']}"] for name, _, e in masks if f"sub-{e['sub']}" in by_id}


def _label(s: str) -> str:
    """A BIDS label: letters and digits only."""
    return re.sub(r"[^A-Za-z0-9]", "", s)


def write_derivatives(root: str, out_dir: str, masks, result: dict) -> "list[str]":
    """Lay the finished group run out as a BIDS derivative dataset; returns
    the BIDS files written."""
    from ._version import __version__
    desc = {"Name": "lqtpy lesion analysis", "BIDSVersion": "1.9.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "lqtpy", "Version": __version__,
                             "Description": "Lesion morphometry, atlas overlap and quality checks"}],
            "SourceDatasets": [{"URL": os.path.relpath(os.path.abspath(root),
                                                       os.path.abspath(out_dir)).replace(os.sep, "/")}]}
    with open(os.path.join(out_dir, "dataset_description.json"), "w", encoding="utf-8") as fh:
        json.dump(desc, fh, indent=2)
    with open(os.path.join(out_dir, ".bidsignore"), "w", encoding="utf-8") as fh:
        fh.write("subjects/\n*.html\n*.csv\n*.pdf\n*.xlsx\n.lqtpy_cache/\nanonymisation_key*\n"
                 "*_summary.json\n*_morphometry.json\n/*.nii.gz\n")
    written = []
    by_name = {s["name"]: s for s in result.get("subjects", [])}
    for name, _, ents in masks:
        if name not in by_name:
            continue
        src = os.path.join(out_dir, "subjects", name)
        if not os.path.isdir(src):
            continue
        folder = os.path.join(out_dir, f"sub-{ents['sub']}")
        if "ses" in ents:
            folder = os.path.join(folder, f"ses-{ents['ses']}")
        folder = os.path.join(folder, "anat")
        os.makedirs(folder, exist_ok=True)
        prefix = "_".join(f"{k}-{v}" for k, v in ents.items()
                          if k in ("sub", "ses", "acq", "run", "rec"))
        base = f"{prefix}_space-{SPACE}"
        for f in sorted(os.listdir(src)):
            if f.endswith(".csv") and f.startswith(name + "_"):
                what = _label(f[len(name) + 1:-4])
                dst = os.path.join(folder, f"{base}_desc-{what}_stats.tsv")
                with open(os.path.join(src, f), newline="", encoding="utf-8") as fi, \
                        open(dst, "w", newline="", encoding="utf-8") as fo:
                    w = csv.writer(fo, dialect="excel-tab", lineterminator="\n")
                    for row in csv.reader(fi):
                        w.writerow(["n/a" if c == "" else c for c in row])
                written.append(dst)
        morph = os.path.join(src, f"{name}_morphometry.json")
        if os.path.exists(morph):
            dst = os.path.join(folder, f"{base}_desc-morphometry_stats.json")
            with open(morph, encoding="utf-8") as fi:
                data = json.load(fi)
            data["Sources"] = [os.path.relpath(p, out_dir).replace(os.sep, "/")
                               for n, p, _ in masks if n == name]
            with open(dst, "w", encoding="utf-8") as fo:
                json.dump(data, fo, indent=2)
            written.append(dst)
    return written
