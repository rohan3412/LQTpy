"""Subject metadata for group runs: age, condition, symptom side, scores.

``analyze_group(..., metadata="subjects.csv")`` joins one row per subject
into the group tables and reports. The file is CSV (or TSV) with a header
row; one column names the subject. It is found by ``subject_column=``, or
else the first of ``subject``, ``participant_id``, ``id``, ``name``,
``file`` that exists, or else the first column. A subject matches when that
value equals the lesion's name (file name without ``.nii.gz`` etc.); file
names with an extension match too.

Metadata can also be passed as a list of dicts, a ``{name: {column:
value}}`` dict, or a pandas DataFrame.

Side columns
------------
A column giving the side of the lesion (``lesion_side``, ``expected_side``,
``side``) or of the symptoms (``symptom_side``, ``hemiparesis_side``,
``weakness_side``, ``affected_side``; the lesion is expected on the
*opposite* side, which holds for supratentorial lesions) is used for the
left/right flip check: a subject whose lesion is mostly on the other side
gets a ``side_mismatch`` quality flag.
"""
from __future__ import annotations

import csv
import os

__all__ = ["load_metadata", "expected_sides", "LESION_SIDE_COLUMNS", "SYMPTOM_SIDE_COLUMNS"]

_ID_COLUMNS = ("subject", "subject_id", "participant_id", "participant", "id", "name", "file",
               "filename")
LESION_SIDE_COLUMNS = ("lesion_side", "expected_side", "side", "lesion_hemisphere", "hemisphere")
SYMPTOM_SIDE_COLUMNS = ("symptom_side", "hemiparesis_side", "weakness_side", "affected_side",
                        "body_side")


def _norm(v) -> str:
    from .formats import strip_extension
    return strip_extension(os.path.basename(str(v).strip())).lower()


def _rows(metadata) -> "list[dict]":
    if metadata is None:
        return []
    if hasattr(metadata, "to_dict") and hasattr(metadata, "columns"):
        return [{str(k): v for k, v in r.items()} for r in metadata.to_dict("records")]
    if isinstance(metadata, dict):
        return [{"subject": k, **dict(v)} for k, v in metadata.items()]
    if isinstance(metadata, (list, tuple)):
        return [dict(r) for r in metadata]
    path = os.fspath(metadata)
    with open(path, newline="", encoding="utf-8-sig") as fh:
        text = fh.read()
    dialect = "excel-tab" if path.lower().endswith((".tsv", ".tab")) or (
        "\t" in text.splitlines()[0] and "," not in text.splitlines()[0]) else "excel"
    return [dict(r) for r in csv.DictReader(text.splitlines(), dialect=dialect)]


def load_metadata(metadata, names, *, subject_column: "str | None" = None) -> dict:
    """Match metadata rows to subject ``names``.

    Returns ``{"by_subject": {name: {column: value}}, "columns": [...],
    "subject_column": str, "unmatched_rows": [...], "missing_subjects":
    [...]}``. Values are kept as strings (numbers are converted where the
    whole column is numeric)."""
    rows = _rows(metadata)
    out = {"by_subject": {}, "columns": [], "subject_column": None, "unmatched_rows": [],
           "missing_subjects": list(names)}
    if not rows:
        return out
    cols = list(rows[0].keys())
    if subject_column is None:
        low = {c.lower().strip(): c for c in cols}
        subject_column = next((low[c] for c in _ID_COLUMNS if c in low), cols[0])
    elif subject_column not in cols:
        raise ValueError(f"metadata has no column {subject_column!r}; columns: {cols}")
    others = [c for c in cols if c != subject_column]
    # Numeric columns become floats.
    numeric = set()
    for c in others:
        vals = [r.get(c) for r in rows if str(r.get(c, "")).strip() not in ("", "nan", "NA")]
        try:
            [float(v) for v in vals]
            if vals:
                numeric.add(c)
        except (TypeError, ValueError):
            pass
    lookup = {_norm(n): n for n in names}
    matched = set()
    for r in rows:
        key = _norm(r.get(subject_column, ""))
        name = lookup.get(key)
        if name is None:
            out["unmatched_rows"].append(str(r.get(subject_column, "")))
            continue
        vals = {}
        for c in others:
            v = r.get(c)
            sv = "" if v is None else str(v).strip()
            if c in numeric and sv not in ("", "nan", "NA"):
                f = float(sv)
                vals[c] = int(f) if f.is_integer() else f
            else:
                vals[c] = sv
        out["by_subject"][name] = vals
        matched.add(name)
    out["columns"] = others
    out["subject_column"] = subject_column
    out["missing_subjects"] = [n for n in names if n not in matched]
    return out


def _side(v) -> "str | None":
    s = str(v or "").strip().lower()
    if s in ("l", "left", "links", "gauche", "sinistra", "izquierda"):
        return "left"
    if s in ("r", "right", "rechts", "droite", "destra", "derecha"):
        return "right"
    return None


def expected_sides(meta: dict, side_column: "str | None" = "auto") -> "tuple[dict, str | None, bool]":
    """``({name: "left"|"right"}, column_used, from_symptoms)`` from the
    metadata's side column. ``side_column="auto"`` finds one by name;
    ``None`` turns the check off."""
    if not side_column or not meta.get("by_subject"):
        return {}, None, False
    cols = meta["columns"]
    if side_column == "auto":
        low = {c.lower().strip().replace(" ", "_"): c for c in cols}
        col = next((low[c] for c in LESION_SIDE_COLUMNS if c in low), None)
        symptom = False
        if col is None:
            col = next((low[c] for c in SYMPTOM_SIDE_COLUMNS if c in low), None)
            symptom = col is not None
        if col is None:
            return {}, None, False
    else:
        if side_column not in cols:
            raise ValueError(f"metadata has no column {side_column!r}")
        col = side_column
        symptom = col.lower().strip().replace(" ", "_") in SYMPTOM_SIDE_COLUMNS
    out = {}
    for name, vals in meta["by_subject"].items():
        s = _side(vals.get(col))
        if s:
            out[name] = ({"left": "right", "right": "left"}[s] if symptom else s)
    return out, col, symptom
