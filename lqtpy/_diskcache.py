"""On-disk cache for arrays derived from the bundled data: atlas label
volumes, distance maps, border zones, the brain mask, the tract indices.

Each entry is built once per lqtpy version and written uncompressed, one
``.npy`` file per array in a folder of its own. Later processes
memory-map those files instead of reading and decompressing them, so
start-up costs almost nothing, and processes running side by side (worker
pools) share one copy of the data in memory. An entry left by an earlier
release (one compressed ``.npz`` file) is converted on first use, not rebuilt.

The cache lives in ``LQTPY_CACHE_DIR`` if set, else the platform's user
cache folder (``%LOCALAPPDATA%\\lqtpy\\cache``, ``~/Library/Caches/lqtpy``
or ``$XDG_CACHE_HOME/lqtpy``). ``LQTPY_NO_DISK_CACHE=1`` turns it off.
Any read or write problem falls back to building in memory, silently.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tempfile

import numpy as np

__all__ = ["cache_dir", "cached_arrays", "clear", "file_key", "once"]

#: Arrays smaller than this are read into memory rather than mapped.
_MAP_MIN_BYTES = 1 << 16


def once(build):
    """Decorator: ``build()`` runs once per process, the first time the
    function is called; every caller (on any thread) gets that same value.
    Concurrent first callers wait for the one build. An exception is not
    cached: the next call tries again."""
    import functools
    import threading
    lock = threading.Lock()
    box: list = []

    @functools.wraps(build)
    def get():
        if not box:
            with lock:
                if not box:
                    box.append(build())
        return box[0]
    return get


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


def file_key(path: str) -> str:
    """A short key for a source file's identity (path, size, time modified),
    for entries built from a file that can change between runs."""
    st = os.stat(path)
    ident = f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}"
    return hashlib.sha1(ident.encode("utf-8")).hexdigest()[:12]


def _frozen(arrays: dict) -> dict:
    for v in arrays.values():
        if isinstance(v, np.ndarray):
            v.flags.writeable = False
    return arrays


def _read(folder: str) -> "dict | None":
    try:
        names = [f for f in os.listdir(folder) if f.endswith(".npy")]
    except OSError:
        return None
    if not names:
        return None
    out = {}
    try:
        for f in names:
            path = os.path.join(folder, f)
            if os.path.getsize(path) < _MAP_MIN_BYTES:
                a = np.load(path, allow_pickle=False)
            else:
                a = np.load(path, mmap_mode="r", allow_pickle=False).view(np.ndarray)
            out[f[:-4]] = a
    except Exception:
        return None
    return _frozen(out)


def _read_legacy(path: str) -> "dict | None":
    """An entry written by lqtpy 0.7.0 or earlier: one compressed .npz."""
    if not os.path.exists(path):
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except Exception:
        return None


def _write(d: str, name: str, arrays: dict) -> None:
    folder = os.path.join(d, name)
    tmp = None
    try:
        os.makedirs(d, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=d, prefix=f".{name}-")
        for k, v in arrays.items():
            np.save(os.path.join(tmp, f"{k}.npy"), np.asarray(v), allow_pickle=False)
        os.replace(tmp, folder)             # appears complete, or not at all
        tmp = None
    except Exception:
        pass                                # e.g. another process wrote it first
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)


def cached_arrays(name: str, build) -> dict:
    """``build()`` returns a dict of arrays; it is stored as the entry
    ``name`` and read from there on later calls. Arrays come back read-only
    (large ones memory-mapped)."""
    d = cache_dir()
    if not d:
        return _frozen(build())
    folder = os.path.join(d, name)
    if os.path.isdir(folder):
        out = _read(folder)
        # A damaged entry is left alone (another process may have it open)
        # and the arrays are built in memory instead.
        return out if out is not None else _frozen(build())
    arrays = _read_legacy(os.path.join(d, f"{name}.npz"))
    if arrays is None:
        arrays = build()
    _write(d, name, arrays)
    return _read(folder) or _frozen(arrays)


def _tree_size(path: str) -> int:
    total = 0
    for r, _, fs in os.walk(path):
        for f in fs:
            try:
                total += os.path.getsize(os.path.join(r, f))
            except OSError:
                pass
    return total


def old_versions() -> "list[tuple[str, int]]":
    """``(folder, bytes)`` for the caches of other lqtpy versions that sit
    next to this version's cache. Empty when the cache is off."""
    d = cache_dir()
    if not d:
        return []
    parent = os.path.dirname(d)
    try:
        names = sorted(os.listdir(parent))
    except OSError:
        return []
    mine = os.path.basename(d)
    # Only folders named like a version ("0.6.1", "0.7.0rc1"): a cache folder
    # chosen with LQTPY_CACHE_DIR may hold other things.
    return [(p, _tree_size(p)) for n in names
            if n != mine and n[:1].isdigit() and os.path.isdir(p := os.path.join(parent, n))]


def prune() -> "list[tuple[str, int]]":
    """Delete the caches of other lqtpy versions (never this version's);
    returns what was removed as ``(folder, bytes)``. Never runs on its own."""
    gone = []
    for p, size in old_versions():
        shutil.rmtree(p, ignore_errors=True)
        if not os.path.exists(p):
            gone.append((p, size))
    return gone


def clear() -> None:
    """Remove this version's cached arrays. (On Windows, files that a
    running process has mapped stay until it exits.)"""
    d = cache_dir()
    if d and os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
