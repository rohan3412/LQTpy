"""``lqtpy doctor``: check an installation and say how to fix each gap.

Each check returns ``(status, what, how_to_fix)``; status is ``ok`` (works),
``note`` (an optional part is missing: reports still work) or ``problem``
(something lqtpy needs is broken). Nothing is changed or downloaded.
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import tempfile

__all__ = ["run_checks"]

#: optional package -> (what it adds, pip extra)
_OPTIONAL = {
    "nilearn": ("the slice viewer's script", "report"),
    "ggseg": ("subcortical map geometry in reports", "report"),
    "pandas": ("DataFrame tables (lqtpy.tables)", "tables"),
    "matplotlib": ("the older figure helpers", "plots"),
    "pypdf": ("merged PDF bundles (--pdf-bundle)", "pdf"),
    "highdicom": ("DICOM-SEG input", "formats"),
    "openpyxl": ("Excel workbooks (--xlsx)", "xlsx"),
}


def _has(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _writable(folder: str) -> bool:
    try:
        os.makedirs(folder, exist_ok=True)
        with tempfile.TemporaryFile(dir=folder):
            pass
        return True
    except OSError:
        return False


def run_checks() -> "list[tuple[str, str, str]]":
    out: "list[tuple[str, str, str]]" = []
    from ._version import __version__

    out.append(("ok", f"lqtpy {__version__} on Python {sys.version.split()[0]}", ""))

    for mod in ("numpy", "scipy", "nibabel"):
        if _has(mod):
            out.append(("ok", f"{mod} installed", ""))
        else:
            out.append(("problem", f"{mod} is missing", f"pip install {mod}"))

    for mod, (what, extra) in _OPTIONAL.items():
        if _has(mod):
            out.append(("ok", f"{mod} installed ({what})", ""))
        else:
            out.append(("note", f"{mod} not installed: no {what}", f"pip install {mod}"
                        + (f"   (or: pip install lqtpy[{extra}])" if extra != "xlsx" else "")))

    from . import _diskcache
    d = _diskcache.cache_dir()
    if not d:
        out.append(("note", "disk cache is off (LQTPY_NO_DISK_CACHE): every start rebuilds the "
                            "atlas arrays, about 1 s slower", "unset LQTPY_NO_DISK_CACHE"))
    elif _writable(os.path.dirname(d)):
        old = _diskcache.old_versions()
        if old:
            mb = sum(s for _, s in old) / 2**20
            out.append(("note", f"{len(old)} older version cache(s) use {mb:.0f} MB",
                        "lqtpy cache --prune"))
        else:
            out.append(("ok", f"disk cache at {d}", ""))
    else:
        out.append(("note", f"cache folder {os.path.dirname(d)} is not writable: lqtpy works, "
                            "but rebuilds its arrays on every start",
                    "set LQTPY_CACHE_DIR to a folder you can write to"))

    from . import datasets
    if datasets.discon_index_available():
        out.append(("ok", "disconnection index found", ""))
    else:
        out.append(("note", "no disconnection index: reports leave streamline disconnection out "
                            "(tract overlap still works)",
                    f"build it with tools/build_discon_index.py into {datasets.get_data_home()}"))

    if datasets.suit_available():
        out.append(("ok", "SUIT cerebellar atlas downloaded", ""))
    else:
        out.append(("note", "SUIT cerebellar atlas not downloaded", "lqtpy fetch suit"))

    from .config import find_config, load_config
    path = find_config()
    if path is None:
        out.append(("ok", "no settings file (defaults are used)", ""))
    else:
        try:
            load_config(path)
            out.append(("ok", f"settings file {path} reads fine", ""))
        except (ValueError, OSError) as exc:
            out.append(("problem", f"settings file {path}: {exc}",
                        "fix or remove that file (see `lqtpy config`)"))

    for label, folder in (("current folder", os.getcwd()),):
        if _writable(folder):
            out.append(("ok", f"can write in the {label}", ""))
        else:
            out.append(("problem", f"cannot write in the {label} ({folder})",
                        "run lqtpy from a folder you can write to, or pass --out"))

    try:
        from .atlases import list_atlases
        out.append(("ok", f"{len(list_atlases())} atlases available", ""))
    except Exception as exc:                      # bundled data damaged or missing
        out.append(("problem", f"atlases cannot be listed: {exc}",
                    "reinstall: pip install --force-reinstall lqtpy"))
    return out
