"""Worker processes for group runs and folder QC.

``n_jobs`` / ``jobs`` / ``-j`` take a number of workers, ``-1`` (one per
CPU) or ``"auto"``: one process for a few files, otherwise one worker per CPU
less one, at most :data:`AUTO_MAX`. One process stays the default.

If the worker pool can't start, or breaks (a locked-down machine, a
frozen app, a worker killed for memory), the files not finished yet are run
in this process instead, with a one-line warning: results are the same,
only slower.
"""
from __future__ import annotations

import os
import sys

__all__ = ["resolve_jobs", "parse_jobs", "run_tasks", "AUTO_MAX", "AUTO_MIN_TASKS"]

#: "auto" never starts more workers than this (each holds a lesion's arrays).
AUTO_MAX = 8
#: "auto" stays in one process for fewer files than this (starting workers costs ~1 s each).
AUTO_MIN_TASKS = 4


def parse_jobs(value):
    """Command-line / settings value -> ``int`` or ``"auto"``."""
    if isinstance(value, str):
        v = value.strip().lower()
        if v == "auto":
            return "auto"
        try:
            return int(v)
        except ValueError:
            raise ValueError(f"jobs should be a number or 'auto', got {value!r}") from None
    return value


def resolve_jobs(n_jobs, n_tasks: int) -> int:
    """How many worker processes to use (1: none, run here)."""
    n_jobs = parse_jobs(n_jobs)
    if n_tasks <= 1 or not n_jobs:
        return 1
    cpus = os.cpu_count() or 2
    if n_jobs == "auto":
        if n_tasks < AUTO_MIN_TASKS:
            return 1
        return max(1, min(cpus - 1, AUTO_MAX, n_tasks))
    if n_jobs < 1:
        return min(cpus, n_tasks)
    return min(int(n_jobs), n_tasks)


def _warn(exc, left: int) -> None:
    print(f"lqtpy: worker processes unavailable ({type(exc).__name__}); running the "
          f"remaining {left} in one process.", file=sys.stderr)


def run_tasks(fn, items, n_jobs, on_done=None, *, ordered: bool = False) -> list:
    """``[fn(x) for x in items]``, in worker processes when ``n_jobs`` says
    so. ``on_done(i, result)`` is called as each finishes (in completion
    order, or in input order with ``ordered``). Falls back to this process
    for whatever is left if the pool can't start or breaks."""
    items = list(items)
    results: list = [None] * len(items)
    done = [False] * len(items)

    def finish(i, r):
        results[i], done[i] = r, True
        if on_done is not None:
            on_done(i, r)

    workers = resolve_jobs(n_jobs, len(items))
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from concurrent.futures.process import BrokenProcessPool
        ex = None
        try:
            # Starting the pool (workers spawn on submit): any failure here is
            # the environment, not a task.
            ex = ProcessPoolExecutor(max_workers=workers)
            futs = {ex.submit(fn, x): i for i, x in enumerate(items)}
        except (BrokenProcessPool, OSError, ImportError, RuntimeError, NotImplementedError) as exc:
            _warn(exc, len(items))
        else:
            try:
                # A task's own exception propagates as before; only a broken
                # pool (a worker died) falls back.
                if ordered:
                    for fut, i in sorted(futs.items(), key=lambda kv: kv[1]):
                        finish(i, fut.result())
                else:
                    for fut in as_completed(futs):
                        finish(futs[fut], fut.result())
            except BrokenProcessPool as exc:
                _warn(exc, done.count(False))
        finally:
            if ex is not None:
                ex.shutdown(wait=True, cancel_futures=True)
    for i, x in enumerate(items):
        if not done[i]:
            finish(i, fn(x))
    return results
