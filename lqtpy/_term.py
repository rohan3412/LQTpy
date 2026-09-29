"""Terminal output for the CLI and ``verbose=True`` runs.

Plain, aligned text; colour only for the quality word, and only when the
output is a terminal that supports it (``NO_COLOR`` or ``LQTPY_NO_COLOR``
turn it off, ``FORCE_COLOR`` on). Paths are printed in the platform's own
form, once.
"""
from __future__ import annotations

import os
import sys

from . import _format as fmt

_COLOURS = {"ok": "32", "info": "90", "warn": "33", "error": "31", "dim": "90", "bold": "1"}
_enabled = None


def _vt_windows() -> bool:
    try:
        import ctypes
        k = ctypes.windll.kernel32
        h = k.GetStdHandle(-11)
        mode = ctypes.c_uint32()
        if not k.GetConsoleMode(h, ctypes.byref(mode)):
            return False
        return bool(k.SetConsoleMode(h, mode.value | 0x0004))
    except Exception:
        return False


def colour_on() -> bool:
    global _enabled
    if _enabled is None:
        if os.environ.get("NO_COLOR") or os.environ.get("LQTPY_NO_COLOR"):
            _enabled = False
        elif os.environ.get("FORCE_COLOR"):
            _enabled = True
        else:
            tty = hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
            _enabled = bool(tty and os.environ.get("TERM") != "dumb"
                            and (os.name != "nt" or _vt_windows()))
    return _enabled


def paint(text: str, style: str) -> str:
    if not colour_on() or style not in _COLOURS:
        return text
    return f"\033[{_COLOURS[style]}m{text}\033[0m"


def quality(severity) -> str:
    sev = severity or "ok"
    return paint(fmt.quality_label(sev), sev)


def path(p) -> str:
    return os.path.normpath(os.fspath(p))


def _out(line: str = "") -> None:
    print(line, flush=True)


def header(what: str) -> None:
    from ._version import __version__
    _out(paint(f"lqtpy {__version__}", "bold") + paint(f" · {what}", "dim"))


def field(label: str, value: str) -> None:
    _out(f"  {paint(label.ljust(10), 'dim')}{value}")


# --------------------------------------------------------------------------- #
def start_single(name: str) -> None:
    header(f"analysing {name}")


def _side(m) -> str:
    pl, pr, pm = m["pct_left"], m["pct_right"], m.get("pct_midline", 0)
    s = f"{pl:.0f}% left · {pr:.0f}% right"
    return s + (f" · {pm:.0f}% midline" if pm >= 1 else "")


def done_single(res: dict) -> None:
    m = res["morphometry"]
    facts = (res.get("description") or {}).get("facts") or {}
    parts = facts.get("n_clusters", m["n_components"])
    field("Size", f"{fmt.volume(m['volume_mm3'])} · "
                  + ("one connected lesion" if parts == 1 else f"{parts} separate parts"))
    field("Side", _side(m))
    terrs = facts.get("territories") or []
    if terrs:
        field("Territory", f"{fmt.territory_short(terrs[0]['hemisphere'], terrs[0]['territory'])} "
                           f"({terrs[0]['pct_of_lesion']:.0f}%)")
    flags = [f for f in res.get("qc") or [] if f["severity"] != "info"]
    worst = "error" if any(f["severity"] == "error" for f in flags) else (
        "warn" if flags else "ok")
    field("Quality", quality(worst) + (paint(f" · {len(flags)} flag" + ("s" if len(flags) != 1 else ""), "dim")
                                        if flags else ""))
    for f in flags[:4]:
        _out(f"            {paint('-', 'dim')} {f['message']}")
    if res.get("report_path"):
        field("Report", path(res["report_path"]))
    if res.get("pdf_path"):
        field("PDF", path(res["pdf_path"]))
    if res.get("anonymisation_key"):
        field("Key file", path(res["anonymisation_key"]) + paint("  (private: maps IDs to names)", "dim"))
    _out(paint(f"  done in {res.get('elapsed', '?')}", "dim"))


def start_group(n: int, n_jobs=1) -> None:
    w = "" if not n_jobs or n_jobs == 1 else (
        f", {os.cpu_count()} workers" if n_jobs < 1 else f", {n_jobs} workers")
    header(f"group of {n} subject{'s' if n != 1 else ''}{w}")


def subject_line(i: int, n: int, e: dict, cached=False) -> None:
    width = len(str(n))
    name = e["name"] if len(e["name"]) <= 28 else e["name"][:25] + "..."
    worst = e.get("qc_worst")
    vol = fmt.volume(e["morphometry"]["volume_mm3"])
    terr = e.get("territory") or "—"
    tail = paint("  (reused)", "dim") if cached else ""
    _out(f"  {paint(f'[{i:>{width}}/{n}]', 'dim')} {name:<28} {vol:>11}  {terr:<14} "
         f"{quality(worst)}{tail}")


def done_group(res: dict) -> None:
    n = res["n_subjects"]
    q = res.get("n_subjects_with_warnings", 0)
    extra = []
    if res.get("duplicates"):
        k = len(res["duplicates"])
        extra.append(f"{k} duplicate pair{'s' if k != 1 else ''}")
    if res.get("outliers"):
        extra.append(f"{len(res['outliers'])} outlier{'s' if len(res['outliers']) != 1 else ''}")
    _out()
    field("Subjects", f"{n} · median {fmt.volume(res['median_volume_mm3'])}")
    field("Quality", (quality("warn") + f" · {q} of {n}") if q else quality("ok"))
    if extra:
        field("Group", " · ".join(extra))
    if res.get("comparison"):
        c = res["comparison"]
        field("Compared", f"{c['column']}: {c['a']} (n={c['n_a']}) vs {c['b']} (n={c['n_b']})")
    if res.get("report_path"):
        field("Report", path(res["report_path"]))
    if res.get("pdf_path"):
        field("PDF", path(res["pdf_path"]))
    if res.get("bundle_path"):
        field("Bundle", path(res["bundle_path"]))
    if res.get("anonymisation_key"):
        field("Key file", path(res["anonymisation_key"]) + paint("  (private: maps IDs to names)", "dim"))
    _out(paint(f"  done in {res.get('elapsed', '?')}", "dim"))
