"""On-disk cache for template-derived arrays (distance maps, border zones).

They depend only on the bundled atlases, so each is built once per lqtpy
version and reused by every later process: a cold report goes from ~15 s
to ~1 s. The cache lives in ``LQTPY_CACHE_DIR`` if set, else the platform's
user cache folder (``%LOCALAPPDATA%\\lqtpy\\cache``, ``~/Library/Caches/lqtpy``
or ``$XDG_CACHE_HOME/lqtpy``). ``LQTPY_NO_DISK_CACHE=1`` turns it off.
Any read or write problem falls back to building in memory, silently.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np

__all__ = ["cache_dir", "cached_arrays", "clear"]


def cache_dir() -> "str | None":
    if os.environ.get("LQTPY_NO_DISK_CACHE"):
        return None
    d = os.environ.get("LQTPY_CACHE_DIR")
    if not d:
        if sys.platform.startswith("win"):
            base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
            d = os.path.join(base, "lqtpy", "cache")
        elif sys.platform == "darwin":
            d = os.path.expanduser("~/Library/Caches/lqtpy")
        else:
            d = os.path.join(os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache"), "lqtpy")
    from ._version import __version__
    return os.path.join(d, __version__)


def cached_arrays(name: str, build) -> dict:
    """``build()`` returns a dict of arrays; it is stored as ``name.npz``
    and loaded instead on later calls. Arrays come back read-only."""
    d = cache_dir()
    path = os.path.join(d, f"{name}.npz") if d else None
    if path and os.path.exists(path):
        try:
            with np.load(path, allow_pickle=False) as z:
                out = {k: z[k] for k in z.files}
            for v in out.values():
                v.flags.writeable = False
            return out
        except Exception:
            pass
    out = build()
    if path:
        try:
            os.makedirs(d, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=d, suffix=".npz")
            os.close(fd)
            np.savez_compressed(tmp, **out)
            os.replace(tmp, path)
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
    for v in out.values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return out


def clear() -> None:
    """Remove this version's cached arrays."""
    import shutil
    d = cache_dir()
    if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
