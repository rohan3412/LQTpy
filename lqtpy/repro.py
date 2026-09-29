"""Reproducibility: what exactly produced a report, and how to repeat it.

Every result's manifest carries a **run record**: the lqtpy, Python, numpy,
scipy and nibabel versions, every setting, and for each input its SHA-256
checksum (of the file's bytes, or of the mask for an in-memory image) and,
unless the run was anonymised, its path. Reports embed the record as JSON
(``<script type="application/json" id="lqtpy-run">``) and show it in their
"About this report" section.

``lqtpy rerun report.html`` (or :func:`rerun`) reads that record and runs
the same analysis again. Inputs are found by their recorded path; if a file
has moved (or the run was anonymised), pass ``search=`` folders and it is
found by checksum. A file whose checksum no longer matches is reported, and
refused unless ``allow_changed=True``.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import re

__all__ = ["checksum", "run_record", "read_run_record", "rerun", "RUN_SETTINGS"]

#: Settings recorded and replayed (keyword arguments of analyze_single/group).
RUN_SETTINGS = ("atlases", "threshold", "tracts", "qc", "describe", "profile", "extras",
                "disconnection", "expected_side", "subject_reports", "metadata",
                "subject_column", "group_by", "compare", "side_column", "anonymise")


def checksum(p) -> str:
    """SHA-256 of a prepared lesion's source file, or of its mask and affine
    for an in-memory image."""
    src = getattr(p, "source", None)
    if isinstance(src, (str, os.PathLike)) and os.path.isfile(os.fspath(src)):
        return file_sha256(os.fspath(src))
    import numpy as np
    h = hashlib.sha256()
    h.update(np.packbits(p.mask).tobytes())
    h.update(np.asarray(p.affine, float).tobytes())
    return "mask:" + h.hexdigest()


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def environment() -> dict:
    import numpy
    import scipy
    import nibabel
    from ._version import __version__
    return {"lqtpy": __version__, "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__,
            "nibabel": nibabel.__version__, "platform": platform.platform(terse=True)}


def _jsonable(v):
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return f"<{type(v).__name__}>"


def run_record(kind: str, inputs: "list[dict]", settings: dict) -> dict:
    """``inputs``: ``[{"name", "path" (or None), "sha256"}]``."""
    return {"kind": kind, "environment": environment(),
            "settings": {k: _jsonable(settings.get(k)) for k in RUN_SETTINGS if k in settings},
            "inputs": inputs}


def read_run_record(report_path: str) -> dict:
    with open(report_path, encoding="utf-8") as fh:
        text = fh.read()
    m = re.search(r'<script type="application/json" id="lqtpy-run">(.*?)</script>', text, re.S)
    if not m:
        raise ValueError(f"'{report_path}' has no LQTpy run record (made by lqtpy < 0.5?).")
    return json.loads(m.group(1).replace("<\\/", "</"))


def _find_by_checksum(sha: str, search) -> "str | None":
    from .formats import is_lesion_file
    for folder in search or []:
        for root, _, files in os.walk(folder):
            for f in files:
                if is_lesion_file(f):
                    p = os.path.join(root, f)
                    try:
                        if file_sha256(p) == sha:
                            return p
                    except OSError:
                        continue
    return None


def resolve_inputs(record: dict, *, search=None, allow_changed: bool = False) -> "list[tuple[str, str]]":
    """``[(name, path)]`` for a record's inputs; raises if any is missing or
    changed (unless ``allow_changed``)."""
    from .errors import InvalidLesionError
    out, problems = [], []
    for inp in record.get("inputs", []):
        sha = inp.get("sha256", "")
        path = inp.get("path")
        if path and os.path.isfile(path):
            now = file_sha256(path)
            if sha and not sha.startswith("mask:") and now != sha:
                found = _find_by_checksum(sha, search)
                if found:
                    path = found
                elif not allow_changed:
                    problems.append(f"{inp['name']}: '{path}' has changed since the report was made")
                    continue
            out.append((inp["name"], path))
            continue
        if sha.startswith("mask:"):
            problems.append(f"{inp['name']}: was an in-memory image; it can't be found again")
            continue
        found = _find_by_checksum(sha, search)
        if found:
            out.append((inp["name"], found))
        else:
            problems.append(f"{inp['name']}: not found" + (" (pass search folders)" if not search else ""))
    if problems:
        raise InvalidLesionError("Can't rerun: " + "; ".join(problems) + ".")
    return out


def rerun(report_path: str, *, out_dir: "str | None" = None, search=None,
          allow_changed: bool = False, plots: bool = True, verbose: bool = False) -> dict:
    """Repeat the analysis recorded in a report. Returns the new result."""
    from .analysis import analyze_group, analyze_single
    rec = read_run_record(report_path)
    s = dict(rec.get("settings", {}))
    inputs = resolve_inputs(rec, search=search, allow_changed=allow_changed)
    out_dir = out_dir or os.path.join(os.path.dirname(os.path.abspath(report_path)), "rerun")
    common = {k: s[k] for k in ("atlases", "threshold", "tracts", "qc", "describe", "profile",
                                "extras", "disconnection") if k in s}
    if rec["kind"] == "group":
        for k in ("subject_reports", "metadata", "subject_column", "group_by", "compare",
                  "side_column", "anonymise"):
            if s.get(k) not in (None, "<DataFrame>"):
                common[k] = s[k]
        if isinstance(common.get("compare"), list):
            common["compare"] = tuple(common["compare"])
        return analyze_group(inputs, out_dir=out_dir, plots=plots, verbose=verbose, **common)
    name, path = inputs[0]
    return analyze_single(path, out_dir=out_dir, plots=plots, verbose=verbose,
                          expected_side=s.get("expected_side"), **common)
