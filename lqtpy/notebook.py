"""Notebook cards (opt-in): a prepared lesion, a description and a QC result
show as small cards in Jupyter instead of plain ``repr`` text.

    import lqtpy
    lqtpy.notebook.enable()          # the one line; disable() switches it off

Nothing changes until ``enable()`` is called, and nothing here imports
IPython unless a notebook session is running. Cards inherit the notebook's
colours (they use ``currentColor``), so they read in light and dark themes.
"""
from __future__ import annotations

import html as _html

__all__ = ["enable", "disable", "card_html"]

_BOX = ("font:13px/1.45 system-ui,sans-serif;border:1px solid rgba(128,128,128,.4);"
        "border-radius:8px;padding:10px 14px;margin:4px 0;max-width:46em")
_TAG = {"ok": "OK", "info": "Note", "warn": "Check", "error": "Problem"}
_MARK = {"ok": "✓", "info": "•", "warn": "▲", "error": "✖"}
_FORMATTERS: list = []


def _e(x) -> str:
    return _html.escape(str(x))


def _card(title: str, rows, body: str = "") -> str:
    r = "".join(f"<tr><td style='opacity:.65;padding:1px 14px 1px 0'>{_e(k)}</td>"
                f"<td style='padding:1px 0'>{v}</td></tr>" for k, v in rows)
    return (f"<div style=\"{_BOX}\"><div style='font-weight:600;font-size:14px;margin-bottom:4px'>"
            f"{_e(title)}</div>" + (f"<table style='border-collapse:collapse'>{r}</table>" if r else "")
            + body + "</div>")


def _prepared_card(p) -> str:
    import numpy as np
    zooms = np.linalg.norm(p.affine[:3, :3], axis=0)
    cc = p.n_voxels * float(abs(np.linalg.det(p.affine[:3, :3]))) / 1000.0
    times = " × "
    grid = times.join(str(s) for s in p.shape)
    vox = times.join(f"{z:.2g}" for z in zooms)
    rows = [("Voxels", f"{p.n_voxels:,}"), ("Volume", f"{cc:,.2f} cc"),
            ("Grid", f"{grid} at {vox} mm"), ("Threshold", f"above {p.threshold:g}")]
    if p.n_voxels:
        c = p.voxel_coords().mean(axis=0)
        w = p.affine[:3, :3] @ c + p.affine[:3, 3]
        rows.insert(2, ("Centre", f"{w[0]:.0f}, {w[1]:.0f}, {w[2]:.0f} mm (world)"))
    return _card(f"Lesion {p.name}", rows)


def _description_card(d) -> str:
    body = d.html(heading_tag="h4") if hasattr(d, "html") else f"<p>{_e(d.text)}</p>"
    return _card(f"Description ({d.profile})", [],
                 "<div style='margin-top:2px'>" + body + "</div>")


def _qc_card(r) -> str:
    rows = [("Quality", f"{_MARK.get(r.worst, '')} <b>{_e(r.verdict)}</b> · score {r.score}")]
    body = ""
    if r.findings:
        items = []
        for f in r.findings:
            tail = f" <span style='opacity:.65'>[{_e(f.where)}]</span>" if f.where else ""
            sug = f"<div style='opacity:.65;margin-left:1.3em'>→ {_e(f.suggestion)}</div>" if f.suggestion else ""
            items.append(f"<li style='list-style:none;margin:3px 0'>{_MARK.get(f.severity, '')} "
                         f"<b>{_e(f.label)}</b> {_e(f.message)}{tail}{sug}</li>")
        body = "<ul style='margin:6px 0 0;padding:0'>" + "".join(items) + "</ul>"
    else:
        rows.append(("Findings", "none"))
    return _card("Mask quality" + (f": {_e(r.source)}" if r.source else ""), rows, body)


def card_html(obj) -> "str | None":
    """The card for a PreparedLesion, Description or QCResult (``None`` for
    anything else)."""
    from .describe import Description
    from .prepared import PreparedLesion
    from .qc_model import QCResult
    if isinstance(obj, PreparedLesion):
        return _prepared_card(obj)
    if isinstance(obj, Description):
        return _description_card(obj)
    if isinstance(obj, QCResult):
        return _qc_card(obj)
    return None


def enable() -> bool:
    """Show lesions, descriptions and QC results as cards in this notebook.
    Returns False (and does nothing) outside IPython."""
    try:
        from IPython import get_ipython
    except ImportError:
        return False
    ip = get_ipython()
    if ip is None:
        return False
    from .describe import Description
    from .prepared import PreparedLesion
    from .qc_model import QCResult
    fmt = ip.display_formatter.formatters["text/html"]
    for cls in (PreparedLesion, Description, QCResult):
        fmt.for_type(cls, card_html)
        _FORMATTERS.append((fmt, cls))
    return True


def disable() -> None:
    """Back to the plain text display."""
    while _FORMATTERS:
        fmt, cls = _FORMATTERS.pop()
        try:
            fmt.pop(cls, None)
        except Exception:
            pass
