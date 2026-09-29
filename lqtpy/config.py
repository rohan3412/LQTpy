"""``lqtpy.toml``: default settings for the command line.

Looked up in this order (the first file found wins; ``--config PATH``
overrides the search):

1. ``./lqtpy.toml`` (the current folder)
2. ``LQTPY_CONFIG`` (environment variable)
3. the user config folder: ``%APPDATA%\\lqtpy\\lqtpy.toml`` on Windows,
   ``~/.config/lqtpy/lqtpy.toml`` elsewhere

Example::

    [defaults]
    atlases = ["aal", "dk", "arterial"]
    profile = "stroke"          # stroke | small_vessel | tumour | research
    extras = "none"             # none | all | "shape,fractal"
    out_dir = "D:/lesion_reports"
    threshold = 0.5
    jobs = 4                    # group runs: worker processes
    pdf = false
    anonymise = false
    viewer = true

Command-line options always win over the file. The Python API ignores the
file (explicit arguments only), so scripts behave the same everywhere.
"""
from __future__ import annotations

import os
import sys

__all__ = ["load_config", "find_config", "KEYS"]

#: key -> type
KEYS = {"atlases": list, "profile": str, "extras": str, "out_dir": str, "threshold": float,
        "jobs": int, "pdf": bool, "anonymise": bool, "viewer": bool, "tracts": bool,
        "disconnection": bool, "subject_reports": bool}


def _user_path() -> str:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or os.path.expanduser("~\\AppData\\Roaming")
        return os.path.join(base, "lqtpy", "lqtpy.toml")
    return os.path.join(os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config"),
                        "lqtpy", "lqtpy.toml")


def find_config() -> "str | None":
    for p in (os.path.join(os.getcwd(), "lqtpy.toml"), os.environ.get("LQTPY_CONFIG"), _user_path()):
        if p and os.path.isfile(p):
            return p
    return None


def _parse_value(v: str):
    v = v.strip()
    if v.startswith("[") and v.endswith("]"):
        inner = v[1:-1].strip()
        return [] if not inner else [_parse_value(x) for x in _split_list(inner)]
    if v[:1] in ('"', "'") and v[-1:] == v[:1]:
        return v[1:-1]
    if v in ("true", "false"):
        return v == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _split_list(s: str):
    out, cur, q = [], "", None
    for ch in s:
        if q:
            cur += ch
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
            cur += ch
        elif ch == ",":
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return out


def _strip_comment(line: str) -> str:
    q = None
    for i, ch in enumerate(line):
        if q:
            if ch == q:
                q = None
        elif ch in "\"'":
            q = ch
        elif ch == "#":
            return line[:i]
    return line


def _mini_toml(text: str) -> dict:
    """The subset of TOML a settings file needs (tables, strings, numbers,
    booleans, one-line arrays, comments), for Python < 3.11 without tomli."""
    data: dict = {}
    table = data
    for raw in text.splitlines():
        line = _strip_comment(raw).strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            table = data.setdefault(line[1:-1].strip(), {})
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            table[k.strip()] = _parse_value(v)
    return data


def _read(path: str) -> dict:
    with open(path, "rb") as fh:
        raw = fh.read()
    try:
        import tomllib  # Python 3.11+
        return tomllib.loads(raw.decode("utf-8"))
    except ImportError:
        pass
    try:
        import tomli
        return tomli.loads(raw.decode("utf-8"))
    except ImportError:
        return _mini_toml(raw.decode("utf-8"))


def load_config(path: "str | None" = None) -> "tuple[dict, str | None]":
    """``(defaults, path)``: the ``[defaults]`` table of the config file
    (validated and typed), and where it came from (None: no file)."""
    path = path or find_config()
    if not path:
        return {}, None
    data = _read(path)
    d = data.get("defaults", data)
    out = {}
    for k, v in d.items():
        if k not in KEYS:
            raise ValueError(f"{path}: unknown setting '{k}' (known: {', '.join(KEYS)})")
        t = KEYS[k]
        if t is list and isinstance(v, str):
            v = [x.strip() for x in v.split(",") if x.strip()]
        elif t is float and isinstance(v, int):
            v = float(v)
        if not isinstance(v, t) or (t is int and isinstance(v, bool)):
            raise ValueError(f"{path}: '{k}' should be {t.__name__}, got {v!r}")
        out[k] = v
    return out, path
