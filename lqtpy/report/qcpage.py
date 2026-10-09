"""One page with every mask's quality flags (``lqtpy qc FOLDER``)."""
from __future__ import annotations

import datetime

from .. import _format as fmt
from ._standalone import _esc, _flag_label, _flag_where, _page

__all__ = ["render_qc_folder"]


def render_qc_folder(rows: "list[dict]", folder: str, cleaned: "list[dict] | None" = None) -> str:
    """Rows from :func:`lqtpy.qc.check_folder`; ``cleaned`` from
    :func:`lqtpy.qc.clean_folder` adds a column saying what was written."""
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    n = len(rows)
    counts = {k: sum(1 for r in rows if r["worst"] == k) for k in ("ok", "info", "warn", "error")}
    by_flag: "dict[str, int]" = {}
    for r in rows:
        for f in r["flags"]:
            if f["severity"] != "info":
                by_flag[f["id"]] = by_flag.get(f["id"], 0) + 1
    done = {c["file"]: c for c in cleaned or []}
    kpis = [("Masks", str(n), "lesion files checked"),
            ("OK", str(counts["ok"] + counts["info"]), "no problems"),
            ("Check", str(counts["warn"]), "worth a look"),
            ("Problem", str(counts["error"]), "major problems, wrong space or unreadable")]
    kpi_html = "".join(f'<div class="kpi"><div class="k">{_esc(k)}</div><div class="v">{_esc(v)}</div>'
                       f'<div class="s">{_esc(s)}</div></div>' for k, v, s in kpis)
    flag_rows = "".join(f'<tr><td>{_esc(k.replace("_", " "))}</td><td class="n">{v}</td></tr>'
                        for k, v in sorted(by_flag.items(), key=lambda kv: -kv[1]))
    chips = "".join(f'<button type="button" class="fchip" data-f="qc" data-v="{q}" aria-pressed="false">{q}</button>'
                    for q in ("OK", "Note", "Check", "Problem")
                    if any(fmt.quality_label(r["worst"]) == q for r in rows))
    body = []
    for r in rows:
        msgs = "<br>".join(
            f'<b>{_esc(_flag_label(f))}:</b> {_esc(f["message"])}'
            + (f' <span class="muted">({_esc(_flag_where(f))})</span>' if _flag_where(f) else "")
            for f in r["flags"] if f["severity"] != "info")
        if r["error"]:
            msgs = _esc(r["error"])
        c = done.get(r["file"])
        cl = f'<td>{_esc(c["status"])}</td>' if cleaned is not None else ""
        vol = f'{fmt.volume_num(r["volume_mm3"])}' if r["volume_mm3"] is not None else "—"
        body.append(f'<tr data-subject="{_esc(r["name"])}" data-f-qc="{fmt.quality_label(r["worst"])}">'
                    f'<td>{_esc(r["name"])}</td><td class="n" data-sort="{r["volume_mm3"] or 0}">{vol}</td>'
                    f'<td data-sort="{ {"ok": 0, "info": 0, "warn": 1, "error": 2}[r["worst"]]}">'
                    f'<span class="qc-badge {r["worst"]}">{fmt.quality_label(r["worst"])}</span></td>'
                    f'<td data-sort="{r.get("score") if r.get("score") is not None else -1}">'
                    f'{_esc(r.get("verdict") or "—")}'
                    + (f' <span class="muted">{r["score"]}</span>' if r.get("score") is not None else "")
                    + '</td>'
                    f'<td>{msgs or "<span class=muted>—</span>"}</td>{cl}</tr>')
    clh = "<th>Cleaned copy</th>" if cleaned is not None else ""
    table = (f'<div class="tablewrap{" scroll" if n > 12 else ""}"><table data-table="qcfolder"><thead><tr>'
             f'<th>Mask</th><th class="n">cm³</th><th>Quality</th><th>Verdict · score</th><th>Flags</th>{clh}</tr></thead>'
             f'<tbody>{"".join(body)}</tbody></table></div>')
    dash = (f'<div class="dash"><div class="hero"><div><h1>Mask quality</h1><div class="meta">{_esc(folder)} · '
            f'{_esc(ts)}</div></div></div><div class="kpis">{kpi_html}</div>'
            '<section class="card"><h2>Masks</h2><div class="filters"><input type="search" id="subj-search" '
            f'placeholder="Search masks" aria-label="Search masks"><span class="fgroup">{chips}</span>'
            '<span class="shown" id="subj-shown" aria-live="polite"></span></div>'
            f'{table}</section>'
            + (f'<section class="card"><h2>Flags</h2><table data-table="qcflags"><thead><tr><th>Flag</th>'
               f'<th class="n">Masks</th></tr></thead><tbody>{flag_rows}</tbody></table></section>' if flag_rows else "")
            + '<p class="note">Clean copies with <code>lqtpy qc FOLDER --clean-to NEW_FOLDER</code>: '
              'image-edge artefacts and specks are removed into new files; the originals are never changed.</p>'
            "</div>")
    clin = (f'<article class="doc"><h1>Mask quality</h1><div class="sub">{_esc(folder)} · {_esc(ts)} · '
            f'{n} masks, {counts["warn"]} to check, {counts["error"]} with problems</div>'
            + table.replace('class="tablewrap scroll"', 'class="tablewrap"') + "</article>")
    # The group filter script expects its subject list; give it an empty one.
    tail = '<script type="application/json" id="lqtpy-subjects">{"subjects": [], "labels": {}}</script>'
    return _page("LQTpy · mask quality", dash, clin, subtitle="Mask quality", tail=tail).replace(
        "table[data-table='subjects']", "table[data-table='qcfolder']")
