"""``lqtpy install-shortcut``: a Windows Explorer "Send to" entry.

Right-click a lesion file, choose Send to > LQTpy report: lqtpy makes the
report in a ``lqtpy_reports`` folder beside the file and opens it. The entry
is one small ``.cmd`` file in the user's SendTo folder (no registry, no
administrator rights); ``--remove`` deletes it. It runs the Python that
installed it, so reinstall the entry after moving to another Python.
"""
from __future__ import annotations

import os
import sys

from .errors import LQTpyError

__all__ = ["sendto_path", "install", "remove"]

_NAME = "LQTpy report.cmd"


def sendto_path() -> str:
    if not sys.platform.startswith("win"):
        raise LQTpyError("install-shortcut is for Windows (Explorer's Send to menu).")
    base = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
    return os.path.join(base, "Microsoft", "Windows", "SendTo", _NAME)


def _script() -> str:
    py = sys.executable
    return (
        "@echo off\r\n"
        "rem Written by `lqtpy install-shortcut`; remove with `lqtpy install-shortcut --remove`.\r\n"
        "setlocal\r\n"
        ":next\r\n"
        "if \"%~1\"==\"\" goto done\r\n"
        "echo Making a report for %~nx1 ...\r\n"
        "set made=\r\n"
        f"for /f \"usebackq delims=\" %%R in (`\"{py}\" -m lqtpy single \"%~1\" --quiet "
        "-o \"%~dp1lqtpy_reports\\%~n1\"`) do set made=1& start \"\" \"%%R\"\r\n"
        "if not defined made pause\r\n"
        "shift\r\n"
        "goto next\r\n"
        ":done\r\n"
    )


def install() -> str:
    path = sendto_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="mbcs", errors="replace", newline="") as fh:
        fh.write(_script())
    return path


def remove() -> "str | None":
    """Delete the entry; returns its path, or None when there was none."""
    path = sendto_path()
    if os.path.exists(path):
        os.remove(path)
        return path
    return None
