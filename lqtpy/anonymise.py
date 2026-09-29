"""Anonymised output: stable subject IDs instead of file names.

Lesion file names often contain patient names. With ``anonymise=True``
every output (folder and file names, reports, CSVs, the run record) uses an
ID such as ``sub-0007`` instead, and no input path is written anywhere.

The mapping lives in a private **key file** (CSV: ``ID, Name, Source``),
by default ``<out_dir>/anonymisation_key.csv``, or wherever
``anonymise="path/to/key.csv"`` points. An existing key file is reused, so
a subject keeps its ID across runs; new subjects get the next free number.
IDs are sequential, not hashes of names, so they cannot be reversed
without the key file. Keep the key file somewhere private and never share
it with the reports.
"""
from __future__ import annotations

import csv
import os

__all__ = ["Anonymiser", "KEY_FILE"]

KEY_FILE = "anonymisation_key.csv"


class Anonymiser:
    """Name -> ID mapping backed by a key file."""

    def __init__(self, key_path: str, prefix: str = "sub-"):
        self.key_path = key_path
        self.prefix = prefix
        self.ids: "dict[str, str]" = {}
        self.sources: "dict[str, str]" = {}
        if os.path.exists(key_path):
            with open(key_path, newline="", encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    self.ids[r["Name"]] = r["ID"]
                    self.sources[r["Name"]] = r.get("Source", "")
        self._dirty = False

    @classmethod
    def from_option(cls, anonymise, out_dir: str) -> "Anonymiser | None":
        if not anonymise:
            return None
        path = anonymise if isinstance(anonymise, (str, os.PathLike)) else os.path.join(
            out_dir, KEY_FILE)
        return cls(os.fspath(path))

    def id_for(self, name: str, source=None) -> str:
        if name not in self.ids:
            used = {int(v[len(self.prefix):]) for v in self.ids.values()
                    if v.startswith(self.prefix) and v[len(self.prefix):].isdigit()}
            n = max(used, default=0) + 1
            self.ids[name] = f"{self.prefix}{n:04d}"
            self._dirty = True
        if source is not None and isinstance(source, (str, os.PathLike)):
            src = os.fspath(source)
            if self.sources.get(name) != src:
                self.sources[name] = src
                self._dirty = True
        return self.ids[name]

    def save(self) -> str:
        if self._dirty or not os.path.exists(self.key_path):
            os.makedirs(os.path.dirname(os.path.abspath(self.key_path)), exist_ok=True)
            with open(self.key_path, "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ID", "Name", "Source"])
                for name, i in sorted(self.ids.items(), key=lambda kv: kv[1]):
                    w.writerow([i, name, self.sources.get(name, "")])
            self._dirty = False
        return self.key_path
