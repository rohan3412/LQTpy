"""Watch a folder: new or changed masks are analysed and the group report
is rebuilt (``lqtpy watch FOLDER -o OUT``).

Polls the folder every ``interval`` seconds (no extra package needed). A
file counts once it has stopped changing (its size and time are the same
on two polls), so a mask still being copied isn't read half-written. Each
rebuild is an incremental group run: subjects already analysed with the
same file and settings are reused, so adding one mask to a folder of 300
re-analyses one. Stop with Ctrl+C.
"""
from __future__ import annotations

import os
import time

__all__ = ["watch", "snapshot"]


def snapshot(folder: str) -> dict:
    from .formats import is_lesion_file
    out = {}
    for f in os.listdir(folder):
        if is_lesion_file(f):
            p = os.path.join(folder, f)
            try:
                st = os.stat(p)
                out[p] = (st.st_size, st.st_mtime)
            except OSError:
                pass
    return out


def watch(folder: str, out_dir: str, *, interval: float = 10.0, max_cycles: "int | None" = None,
          on_run=None, **group_kwargs) -> None:
    """Rebuild the group report whenever the folder's masks change.

    ``group_kwargs`` go to :func:`lqtpy.analyze_group` (``plots`` defaults
    to True, ``incremental`` is always on). ``max_cycles`` stops after that
    many polls (for tests); ``on_run(result)`` is called after each rebuild.
    """
    from . import _term
    from .analysis import analyze_group
    group_kwargs.setdefault("plots", True)
    group_kwargs["incremental"] = True
    last_built, previous, cycles = None, None, 0
    _term.header(f"watching {_term.path(folder)} (every {interval:g} s; Ctrl+C to stop)")
    try:
        while max_cycles is None or cycles < max_cycles:
            cycles += 1
            now = snapshot(folder)
            stable = previous is not None and now == previous
            if now and stable and now != last_built:
                added = len(set(now) - set(last_built or {}))
                changed = sum(1 for p in now if last_built and p in last_built and now[p] != last_built[p])
                removed = len(set(last_built or {}) - set(now))
                _term.field("Change", f"{added} new, {changed} changed, {removed} removed "
                                      f"({len(now)} masks)")
                res = analyze_group(folder, out_dir=out_dir, verbose=False, **group_kwargs)
                last_built = now
                _term.field("Report", _term.path(res.get("report_path", out_dir))
                            + _term.paint(f"  ({res['elapsed']})", "dim"))
                if on_run:
                    on_run(res)
            previous = now
            if max_cycles is None or cycles < max_cycles:
                time.sleep(interval)
    except KeyboardInterrupt:
        _term.field("Stopped", "")
