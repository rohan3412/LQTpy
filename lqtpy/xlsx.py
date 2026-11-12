"""Excel workbook of a run's tables (``--xlsx``, :func:`write_workbook`).

One sheet per CSV table a run wrote, plus a ``data_dictionary`` sheet (what
each column means and its unit). Numbers are written as numbers. Needs
``openpyxl`` (``pip install openpyxl``); nothing else changes without it.
"""
from __future__ import annotations

import csv
import os

from .errors import MissingDependencyError

__all__ = ["write_workbook"]

_DICT = "data_dictionary.csv"
_MAX_SHEET = 31                     # Excel's limit on a sheet name
_BAD = set('[]:*?/\\')


def _value(s: str):
    if s == "":
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
        return f if f == f and f not in (float("inf"), float("-inf")) else s
    except ValueError:
        return s


def _sheet_names(files: "list[str]", prefix: str) -> "dict[str, str]":
    """Unique, Excel-legal sheet names from file names: the common run
    prefix (``lesion_``) and the extension are dropped."""
    names, used = {}, set()
    for f in files:
        stem = os.path.splitext(f)[0]
        if prefix and stem.startswith(prefix) and len(stem) > len(prefix):
            stem = stem[len(prefix):]
        stem = "".join("_" if c in _BAD else c for c in stem).strip("_") or "sheet"
        name, i = stem[:_MAX_SHEET], 2
        while name.lower() in used:
            tail = f"_{i}"
            name, i = stem[:_MAX_SHEET - len(tail)] + tail, i + 1
        used.add(name.lower())
        names[f] = name
    return names


def write_workbook(folder: str, path: "str | None" = None) -> str:
    """Bundle the CSV tables in ``folder`` (and its data dictionary) into one
    workbook and return its path (default ``<folder>/lqtpy_tables.xlsx``)."""
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font
    except ImportError as exc:
        raise MissingDependencyError("Excel output needs openpyxl: pip install openpyxl") from exc
    # The anonymisation key maps IDs back to names: it never goes into a shareable workbook.
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".csv")
                   and f != _DICT and "anonymisation_key" not in f.lower())
    if not files:
        raise FileNotFoundError(f"no CSV tables in {folder}")
    if os.path.exists(os.path.join(folder, _DICT)):
        files.append(_DICT)
    prefix = os.path.commonprefix([os.path.splitext(f)[0] for f in files if f != _DICT])
    prefix = prefix[:prefix.rfind("_") + 1] if "_" in prefix else ""
    names = _sheet_names(files, prefix)
    wb = Workbook()
    wb.remove(wb.active)
    bold = Font(bold=True)
    for f in files:
        ws = wb.create_sheet(names[f])
        with open(os.path.join(folder, f), newline="", encoding="utf-8-sig") as fh:
            for r, row in enumerate(csv.reader(fh), 1):
                ws.append([c if (r == 1 or f == _DICT) else _value(c) for c in row])
        for cell in ws[1]:
            cell.font = bold
        ws.freeze_panes = "A2"
        for col in ws.columns:
            width = max((len(str(c.value)) for c in col[:200] if c.value is not None), default=8)
            ws.column_dimensions[col[0].column_letter].width = min(max(width + 2, 8), 60)
    path = path or os.path.join(folder, "lqtpy_tables.xlsx")
    wb.save(path)
    return path
