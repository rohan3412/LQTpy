"""Build a review page for labelling mask quality by eye, then score lqtpy
against those labels.

    python tools/qc_review.py build FOLDER --survey survey.json --out review/ [--n 200]
                                    [--baseline old_survey.json]
    python tools/qc_review.py pack review/ [--out review_phone.html]
    python tools/qc_review.py score labels.json --survey survey.json

``build`` picks ``--n`` masks from a survey made with ``tools/qc_survey.py``,
stratified so every finding type and grade is represented (plus clean
masks; with ``--baseline``, also about a tenth whose Check/Problem went away
since that older survey, to confirm the drop was right), renders a glass brain and three slices through the first problem
location of each, and writes ``index.html`` plus one picture per mask.
Masks are shown by the survey's short SHA-1 id only; file names (which may
hold patient names) are never written. lqtpy's own findings are hidden
behind a button so the labels stay blind.

Each mask is labelled "Looks fine", "Not fine" (something is wrong, no
details), or with problem types and grades. Labels are kept in the
browser (localStorage) and saved with the
"Save labels" button as ``labels.json``.

``pack`` turns a built review folder into one self-contained HTML file for
a phone: one mask per screen, large buttons, labels saved as you go, and a
Save that shares, downloads or copies ``labels.json``. Labels have the same
format (and the same browser storage key) as the folder version.

``score`` compares the labels with the survey: per type, precision and
recall of Check/Problem findings, and grade agreement, separately for the
tuning half and the validation half (split by id, so the split never
changes).
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import math
import os
import random
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))

#: Problem types offered on the page (id, plain name).
TYPES = [
    ("sparse_slices", "Gaps between drawn slices"),
    ("flat", "Drawn on one slice only"),
    ("hollow", "Outline only, not filled"),
    ("holes", "Holes inside"),
    ("strokes", "Stray strokes / sheets"),
    ("box", "Rectangular block"),
    ("outside_brain", "Outside the brain"),
    ("misregistered", "Shifted / misregistered"),
    ("edge_artifact", "Fragments at the image edge"),
    ("specks", "Stray specks"),
    ("fragmentation", "Shattered into many pieces"),
    ("values", "Odd label values"),
    ("inverted", "Inverted"),
    ("plausibility", "Implausible lesion"),
    ("not_mni", "Not in MNI space"),
    ("other", "Something else"),
]
GRADES = ("minor", "moderate", "major")


def half(fid: str) -> str:
    """'tune' or 'validate', fixed by the id."""
    return "tune" if int(fid, 16) % 2 == 0 else "validate"


def _problems(row) -> "list[dict]":
    return [f for f in row.get("findings", []) if f["sev"] in ("warn", "error")]


def select(rows, n: int, seed: int = 7, dropped: "set | None" = None) -> "list[dict]":
    """Stratified pick: about a fifth clean masks, the rest spread over
    (finding, grade) strata in proportion to the square root of their size,
    at least 3 from each stratum that has them. ``dropped``: ids of masks
    that lost a Check/Problem since a baseline survey; about a tenth of the
    pick comes from them."""
    rng = random.Random(seed)
    ok = [r for r in rows if "error" not in r]
    first = []
    if dropped:
        pool = [r for r in ok if r["id"] in dropped]
        first = rng.sample(pool, min(len(pool), n // 10))
        taken = {r["id"] for r in first}
        ok = [r for r in ok if r["id"] not in taken]
        n -= len(first)
    clean = [r for r in ok if not _problems(r)]
    strata = collections.defaultdict(list)
    for r in ok:
        p = _problems(r)
        if p:
            f = p[0]
            strata[(f["id"], f.get("grade") or f["sev"])].append(r)
    n_clean = min(len(clean), max(1, n // 5))
    rest = n - n_clean
    weights = {k: math.sqrt(len(v)) for k, v in strata.items()}
    total = sum(weights.values()) or 1
    quota = {k: min(len(strata[k]), max(3, round(rest * w / total))) for k, w in weights.items()}
    while sum(quota.values()) > rest:            # trim the biggest quotas first
        k = max(quota, key=lambda k: (quota[k], weights[k]))
        if quota[k] <= 3:
            break
        quota[k] -= 1
    picked = []
    for k, q in sorted(quota.items()):
        picked += rng.sample(strata[k], q)
    picked += rng.sample(clean, min(len(clean), n - len(picked)))
    picked = picked[:n] + first
    rng.shuffle(picked)
    return picked


def _render(args) -> "tuple[str, str | None]":
    path, fid, cut, out_dir = args
    warnings.simplefilter("ignore")
    from lqtpy import figures
    try:
        a = figures.glass_brain_svg(path, scale=1.6)
        b = figures.slices_svg(path, cut, scale=1.6)
    except Exception as exc:                     # noqa: BLE001 - one bad file
        return fid, f"{type(exc).__name__}"
    style = "<style>text{font:12px sans-serif;fill:#555}</style>"
    for suffix, svg in (("g", a), ("s", b)):
        svg = svg.replace("<svg ", '<svg xmlns="http://www.w3.org/2000/svg" '
                          'style="background:#fff" ', 1).replace(">", ">" + style, 1)
        with open(os.path.join(out_dir, "views", f"{fid}{suffix}.svg"), "w", encoding="utf-8") as fh:
            fh.write(svg)
    return fid, None


def build(a) -> int:
    from lqtpy.formats import is_lesion_file
    from qc_survey import file_id
    with open(a.survey, encoding="utf-8") as fh:
        rows = json.load(fh)
    by_id = {}
    for name in os.listdir(a.folder):
        if is_lesion_file(name):
            by_id[file_id(name)] = os.path.join(a.folder, name)
    rows = [r for r in rows if r["id"] in by_id]
    dropped = None
    if a.baseline:
        with open(a.baseline, encoding="utf-8") as fh:
            old = {r["id"]: r for r in json.load(fh) if "error" not in r}
        dropped = set()
        for r in rows:
            o = old.get(r["id"])
            if o is None or "error" in r:
                continue
            before = {f["id"] for f in o["findings"] if f["sev"] in ("warn", "error")}
            now = {f["id"] for f in _problems(r)}
            if before - now:
                dropped.add(r["id"])
    picked = select(rows, a.n, dropped=dropped)
    os.makedirs(os.path.join(a.out, "views"), exist_ok=True)
    jobs = []
    for r in picked:
        p = _problems(r)
        loc = p[0]["locations"][0] if p and p[0].get("locations") else None
        jobs.append((by_id[r["id"]], r["id"], loc["mm"] if loc else None, a.out))
    failed = {}
    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        for fid, err in ex.map(_render, jobs, chunksize=2):
            if err:
                failed[fid] = err
    items = []
    for r in picked:
        if r["id"] in failed:
            continue
        items.append({
            "id": r["id"], "half": half(r["id"]), "verdict": r.get("verdict"),
            "vol_cm3": round((r.get("vol") or 0) / 1000, 2),
            "findings": [{"id": f["id"], "sev": f["sev"], "grade": f.get("grade"),
                          "affected_pct": f.get("affected_pct"),
                          "where": _where(f)} for f in r.get("findings", [])]})
    with open(os.path.join(a.out, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_page(items))
    print(f"{len(items)} masks written to {os.path.join(a.out, 'index.html')}"
          + (f" ({len(failed)} could not be drawn)" if failed else ""))
    return 0


def _where(f) -> str:
    locs = f.get("locations") or []
    if not locs:
        return ""
    loc = locs[0]
    if loc.get("kind") == "slice":
        return f"{loc['plane']} slice {loc['index']}"
    return "at " + ", ".join(f"{v:.0f}" for v in loc["mm"]) + " mm"


def _page(items) -> str:
    types = json.dumps(TYPES)
    data = json.dumps(items)
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mask Quality Review</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#b5481c;--sel:#fbe3d6;--ok:#2f7d4a}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecece8;--mut:#a3a39c;--card:#22221f;--line:#3a3a35;--acc:#f08a55;--sel:#4a2a1a;--ok:#6fcf8f}}
:root[data-theme=dark]{--bg:#171716;--fg:#ecece8;--mut:#a3a39c;--card:#22221f;--line:#3a3a35;--acc:#f08a55;--sel:#4a2a1a;--ok:#6fcf8f}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0;padding:0 16px 80px}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:10px 0;display:flex;gap:12px;align-items:center;flex-wrap:wrap;z-index:2}
h1{font-size:18px;margin:0 12px 0 0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:4px 10px;cursor:pointer}
button.on{background:var(--sel);border-color:var(--acc)}
button.okb.on{border-color:var(--ok);color:var(--ok)}
button.badb.on{border-color:var(--acc);color:var(--acc)}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;margin:16px auto;max-width:1100px;padding:12px 14px}
.card.done{border-left:4px solid var(--ok)}
.top{display:flex;justify-content:space-between;gap:8px;flex-wrap:wrap;color:var(--mut)}
.card img{display:block;width:100%;height:auto;border-radius:6px;margin-top:6px}
.types{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:4px 16px;margin-top:8px}
.t{display:flex;align-items:center;gap:4px}
.t span{flex:1;font-size:14px}
.t button{padding:2px 6px;font-size:13px}
.lq{margin-top:8px;color:var(--mut);font-size:14px}
textarea{width:100%;box-sizing:border-box;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;margin-top:8px;font:inherit}
</style></head><body>
<header><h1>Mask quality review</h1><span id="count"></span>
<button id="save">Save labels</button><label><button id="loadb">Load labels</button><input id="load" type="file" accept=".json" hidden></label>
<label><input type="checkbox" id="todo"> only unlabelled</label></header>
<p style="max-width:1100px;margin:12px auto;color:var(--mut)">For each mask, press <b>Looks fine</b> or <b>Not fine</b>; if you like, also pick the problems you see and how bad they are
(minor: results barely change; moderate: worth fixing; major: the numbers would be wrong). Views are shown through the first place lqtpy
found a problem (or the lesion's centre). Labels stay in this browser until you press Save labels.</p>
<main id="list"></main>
<script>
const TYPES=""" + types + """, ITEMS=""" + data + """, KEY="lqtpy-qc-review-v1";
let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
const store=()=>{try{localStorage.setItem(KEY,JSON.stringify(L))}catch(e){};count()};
const done=id=>{const l=L[id];return !!l&&(l.ok||l.bad||Object.keys(l.problems||{}).length>0)};
const count=()=>{const n=ITEMS.filter(i=>done(i.id)).length;
 document.getElementById("count").textContent=n+" / "+ITEMS.length+" labelled"};
function card(it){const c=document.createElement("section");c.className="card";c.id="m"+it.id;
 const lab=()=>L[it.id]||(L[it.id]={ok:false,problems:{},note:""});
 const mark=()=>{const l=L[it.id];c.classList.toggle("done",done(it.id));
  c.querySelector(".okb").classList.toggle("on",!!(l&&l.ok));
  c.querySelector(".badb").classList.toggle("on",!!(l&&l.bad));
  c.querySelectorAll(".t button").forEach(b=>b.classList.toggle("on",!!(l&&l.problems[b.dataset.t]===b.dataset.g)))};
 c.innerHTML=`<div class="top"><b>${it.id}</b><span>${it.vol_cm3} cm³ · ${it.half}</span></div>
  <img loading="lazy" src="views/${it.id}g.svg" alt="Glass brain of mask ${it.id}"><img loading="lazy" src="views/${it.id}s.svg" alt="Slices through mask ${it.id}">
  <div><button class="okb">Looks fine</button> <button class="badb">Not fine</button></div>
  <div class="types">${TYPES.map(([t,n])=>`<div class="t"><span>${n}</span>${["minor","moderate","major"].map(g=>`<button data-t="${t}" data-g="${g}">${g}</button>`).join("")}</div>`).join("")}</div>
  <textarea rows="1" placeholder="note (optional)"></textarea>
  <div class="lq"><button class="rev">Show lqtpy's findings</button> <span class="lqf" hidden></span></div>`;
 c.querySelector(".okb").onclick=()=>{const l=lab();l.ok=!l.ok;if(l.ok){l.problems={};l.bad=false}store();mark()};
 c.querySelector(".badb").onclick=()=>{const l=lab();l.bad=!l.bad;if(l.bad)l.ok=false;store();mark()};
 c.querySelectorAll(".t button").forEach(b=>b.onclick=()=>{const l=lab();l.ok=false;
  if(l.problems[b.dataset.t]===b.dataset.g)delete l.problems[b.dataset.t];else l.problems[b.dataset.t]=b.dataset.g;store();mark()});
 const ta=c.querySelector("textarea");ta.value=(L[it.id]||{}).note||"";ta.oninput=()=>{lab().note=ta.value;store()};
 c.querySelector(".rev").onclick=()=>{const s=c.querySelector(".lqf");s.hidden=!s.hidden;
  s.textContent=`${it.verdict}: `+(it.findings.map(f=>`${f.id} (${f.grade||f.sev}${f.affected_pct!=null?", "+f.affected_pct+"%":""}${f.where?", "+f.where:""})`).join("; ")||"no findings")};
 mark();return c}
const list=document.getElementById("list");ITEMS.forEach(i=>list.appendChild(card(i)));count();
document.getElementById("todo").onchange=e=>ITEMS.forEach(i=>{const l=L[i.id];
 document.getElementById("m"+i.id).hidden=e.target.checked&&done(i.id)});
document.getElementById("save").onclick=()=>{const b=new Blob([JSON.stringify({schema:"lqtpy.qc-labels",version:1,labels:L},null,1)],{type:"application/json"});
 const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download="labels.json";a.click()};
document.getElementById("loadb").onclick=()=>document.getElementById("load").click();
document.getElementById("load").onchange=async e=>{const f=e.target.files[0];if(!f)return;
 const d=JSON.parse(await f.text());Object.assign(L,d.labels||d);store();location.reload()};
</script></body></html>"""


def pack(a) -> int:
    """One self-contained, phone-friendly file from a built review folder."""
    with open(os.path.join(a.review, "index.html"), encoding="utf-8") as fh:
        page = fh.read()
    items = json.loads(page.split("ITEMS=", 1)[1].split(', KEY="', 1)[0])
    views = {}
    for it in items:
        pair = []
        for suffix in ("g", "s"):
            with open(os.path.join(a.review, "views", f"{it['id']}{suffix}.svg"), encoding="utf-8") as fh:
                pair.append(fh.read())
        views[it["id"]] = pair
    out = a.out or os.path.join(a.review, "review_phone.html")
    blob = json.dumps(views).replace("</", "<\\/")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(_phone_page(items, blob))
    print(f"{len(items)} masks packed into {out} ({os.path.getsize(out) / 1e6:.0f} MB)")
    return 0


def _phone_page(items, views_json: str) -> str:
    return """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mask Quality Review</title>
<style>
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#deded8;--acc:#b5481c;--sel:#fbe3d6;--ok:#2f7d4a;--okbg:#dcefe2}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecece8;--mut:#a3a39c;--card:#22221f;--line:#3a3a35;--acc:#f08a55;--sel:#4a2a1a;--ok:#6fcf8f;--okbg:#1d3325}}
:root[data-theme=dark]{--bg:#171716;--fg:#ecece8;--mut:#a3a39c;--card:#22221f;--line:#3a3a35;--acc:#f08a55;--sel:#4a2a1a;--ok:#6fcf8f;--okbg:#1d3325}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:16px/1.4 system-ui,sans-serif;margin:0;padding:0 16px 96px;-webkit-text-size-adjust:100%}
header{display:flex;justify-content:space-between;align-items:center;gap:8px;padding:10px 0;flex-wrap:wrap}
h1{font-size:17px;margin:0}
button{font:inherit;border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:10px;padding:8px 12px;cursor:pointer;touch-action:manipulation}
button.on{background:var(--sel);border-color:var(--acc);font-weight:600}
.meta{display:flex;justify-content:space-between;color:var(--mut);font-size:14px;margin:4px 0}
.pics svg{display:block;width:100%;max-width:100%;height:auto;border-radius:8px;margin-bottom:6px}
.big{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:6px 0 10px}
.okb,.badb{width:100%;padding:14px 6px;font-size:18px}
.okb.on{background:var(--okbg);border-color:var(--ok);color:var(--ok)}
.badb.on{background:var(--sel);border-color:var(--acc);color:var(--acc)}
.t{display:grid;grid-template-columns:minmax(0,1fr) repeat(3,auto);gap:4px;align-items:center;padding:4px 0;border-bottom:1px solid var(--line)}
.t span{font-size:15px}
.t button{padding:8px 6px;font-size:13px;min-width:0}
html,body{max-width:100%;overflow-x:hidden}
textarea{width:100%;background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:8px;margin-top:10px;font:inherit;padding:8px}
.lq{margin-top:10px;color:var(--mut);font-size:14px}
nav{position:fixed;left:0;right:0;bottom:0;background:var(--bg);border-top:1px solid var(--line);display:flex;gap:8px;padding:10px 16px calc(10px + env(safe-area-inset-bottom))}
nav button{flex:1;padding:12px 4px}
#msg{color:var(--ok);font-size:14px;min-height:1.2em}
dialog{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:12px;width:min(92vw,480px)}
dialog button{width:100%;margin:4px 0}
</style></head><body>
<header><h1>Mask quality review</h1><span id="count"></span><button id="menu">Save / load</button></header>
<div id="msg" aria-live="polite"></div>
<main id="card"></main>
<nav><button id="prev">&#9664; Back</button><button id="todo">Next unlabelled</button><button id="next">Next &#9654;</button></nav>
<dialog id="dlg"><p style="margin-top:0">Labels are kept on this phone as you go. Save them to send back.</p>
<button id="share">Share labels.json</button><button id="dl">Download labels.json</button><button id="copy">Copy labels as text</button>
<button id="loadb">Load labels from a file</button><input id="load" type="file" accept=".json,application/json" hidden>
<textarea id="raw" rows="3" readonly hidden></textarea><button id="close">Close</button></dialog>
<script id="views" type="application/json">""" + views_json + """</script>
<script>
const TYPES=""" + json.dumps(TYPES) + """, ITEMS=""" + json.dumps(items) + """, KEY="lqtpy-qc-review-v1", POS="lqtpy-qc-review-pos";
const VIEWS=JSON.parse(document.getElementById("views").textContent);
let L={};try{L=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){}
let i=0;try{i=Math.min(ITEMS.length-1,parseInt(localStorage.getItem(POS)||"0",10)||0)}catch(e){}
const done=id=>{const l=L[id];return !!l&&(l.ok||l.bad||Object.keys(l.problems||{}).length>0)};
const store=()=>{try{localStorage.setItem(KEY,JSON.stringify(L));localStorage.setItem(POS,String(i))}catch(e){}count()};
const count=()=>{document.getElementById("count").textContent=ITEMS.filter(x=>done(x.id)).length+" / "+ITEMS.length+" labelled"};
const flash=t=>{const m=document.getElementById("msg");m.textContent=t;setTimeout(()=>{if(m.textContent===t)m.textContent=""},2500)};
function show(){const it=ITEMS[i],c=document.getElementById("card");
 const lab=()=>L[it.id]||(L[it.id]={ok:false,problems:{},note:""});
 c.innerHTML=`<div class="meta"><b>${i+1}. ${it.id}</b><span>${it.vol_cm3} cm³</span></div>
  <div class="pics">${VIEWS[it.id][0]}${VIEWS[it.id][1]}</div>
  <div class="big"><button class="okb">Looks fine</button><button class="badb">Not fine</button></div>
  ${TYPES.map(([t,n])=>`<div class="t"><span>${n}</span>${["minor","moderate","major"].map(g=>`<button data-t="${t}" data-g="${g}">${g}</button>`).join("")}</div>`).join("")}
  <textarea rows="2" placeholder="note (optional)"></textarea>
  <div class="lq"><button class="rev">Show lqtpy's findings</button><p class="lqf" hidden></p></div>`;
 const mark=()=>{const l=L[it.id];c.querySelector(".okb").classList.toggle("on",!!(l&&l.ok));
  c.querySelector(".badb").classList.toggle("on",!!(l&&l.bad));
  c.querySelectorAll(".t button").forEach(b=>b.classList.toggle("on",!!(l&&l.problems[b.dataset.t]===b.dataset.g)))};
 c.querySelector(".okb").onclick=()=>{const l=lab();l.ok=!l.ok;if(l.ok){l.problems={};l.bad=false}store();mark();if(l.ok)setTimeout(()=>go(1),250)};
 c.querySelector(".badb").onclick=()=>{const l=lab();l.bad=!l.bad;if(l.bad)l.ok=false;store();mark();if(l.bad)setTimeout(()=>go(1),250)};
 c.querySelectorAll(".t button").forEach(b=>b.onclick=()=>{const l=lab();l.ok=false;
  if(l.problems[b.dataset.t]===b.dataset.g)delete l.problems[b.dataset.t];else l.problems[b.dataset.t]=b.dataset.g;store();mark()});
 const ta=c.querySelector("textarea");ta.value=(L[it.id]||{}).note||"";ta.oninput=()=>{lab().note=ta.value;store()};
 c.querySelector(".rev").onclick=()=>{const s=c.querySelector(".lqf");s.hidden=!s.hidden;
  s.textContent=`${it.verdict}: `+(it.findings.map(f=>`${f.id} (${f.grade||f.sev}${f.affected_pct!=null?", "+f.affected_pct+"%":""}${f.where?", "+f.where:""})`).join("; ")||"no findings")};
 mark();store();window.scrollTo(0,0)}
function go(d){i=Math.max(0,Math.min(ITEMS.length-1,i+d));show()}
document.getElementById("prev").onclick=()=>go(-1);
document.getElementById("next").onclick=()=>go(1);
document.getElementById("todo").onclick=()=>{for(let k=1;k<=ITEMS.length;k++){const j=(i+k)%ITEMS.length;if(!done(ITEMS[j].id)){i=j;show();return}}flash("All masks are labelled. Save and send labels.json.")};
const payload=()=>JSON.stringify({schema:"lqtpy.qc-labels",version:1,labels:L},null,1);
const dlg=document.getElementById("dlg");
document.getElementById("menu").onclick=()=>dlg.showModal();
document.getElementById("close").onclick=()=>dlg.close();
document.getElementById("dl").onclick=()=>{const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([payload()],{type:"application/json"}));a.download="labels.json";document.body.appendChild(a);a.click();a.remove()};
document.getElementById("share").onclick=async()=>{try{const f=new File([payload()],"labels.json",{type:"application/json"});
 if(navigator.canShare&&navigator.canShare({files:[f]})){await navigator.share({files:[f],title:"labels.json"})}else{flash("Sharing isn't available here: use Download or Copy.")}}catch(e){if(e.name!=="AbortError")flash("Sharing failed: use Download or Copy.")}};
document.getElementById("copy").onclick=async()=>{const r=document.getElementById("raw");r.value=payload();r.hidden=false;
 try{await navigator.clipboard.writeText(r.value);flash("Copied. Paste it into a message or a note.")}catch(e){r.select();flash("Select the text below and copy it.")}};
document.getElementById("loadb").onclick=()=>document.getElementById("load").click();
document.getElementById("load").onchange=async e=>{const f=e.target.files[0];if(!f)return;
 try{const d=JSON.parse(await f.text());Object.assign(L,d.labels||d);store();show();dlg.close();flash("Labels loaded.")}catch(err){flash("That file isn't a labels file.")}};
show();
</script></body></html>"""


def score(a) -> int:
    with open(a.labels, encoding="utf-8") as fh:
        labels = json.load(fh)
    labels = labels.get("labels", labels)
    with open(a.survey, encoding="utf-8") as fh:
        rows = {r["id"]: r for r in json.load(fh)}
    rank = {None: 0, "minor": 1, "moderate": 2, "major": 3}
    for part in ("tune", "validate", "all"):
        stats = collections.defaultdict(lambda: collections.Counter())
        grades = collections.Counter()
        masks = collections.Counter()
        n = 0
        for fid, lab in labels.items():
            if fid not in rows or (part != "all" and half(fid) != part):
                continue
            if not lab.get("ok") and not lab.get("problems") and not lab.get("bad"):
                continue                          # unlabelled
            n += 1
            truth = dict(lab.get("problems") or {})
            found = {f["id"]: f.get("grade") for f in _problems(rows[fid])}
            bad = bool(truth) or bool(lab.get("bad"))
            masks[("tp" if found else "fn") if bad else ("fp" if found else "tn")] += 1
            if lab.get("bad") and not truth:
                continue                          # "Not fine" without details: masks only
            for t in set(truth) | set(found):
                if t in truth and t in found:
                    stats[t]["tp"] += 1
                    d = abs(rank[truth[t]] - rank[found[t]])
                    grades["exact" if d == 0 else "one off" if d == 1 else "two off"] += 1
                elif t in found:
                    stats[t]["fp"] += 1
                else:
                    stats[t]["fn"] += 1
        print(f"\n{part}: {n} labelled masks")
        m = masks
        if n:
            pr = m["tp"] / (m["tp"] + m["fp"]) if m["tp"] + m["fp"] else float("nan")
            rc = m["tp"] / (m["tp"] + m["fn"]) if m["tp"] + m["fn"] else float("nan")
            print(f"  any problem: {m['tp']} found, {m['fn']} missed, {m['fp']} false alarms, "
                  f"{m['tn']} agreed clean (precision {pr:.2f}, recall {rc:.2f})")
        print(f"  {'type':16s} {'found':>5s} {'missed':>6s} {'false':>5s} {'precision':>9s} {'recall':>6s}")
        for t, s in sorted(stats.items()):
            pr = s["tp"] / (s["tp"] + s["fp"]) if s["tp"] + s["fp"] else float("nan")
            rc = s["tp"] / (s["tp"] + s["fn"]) if s["tp"] + s["fn"] else float("nan")
            print(f"  {t:16s} {s['tp']:5d} {s['fn']:6d} {s['fp']:5d} {pr:9.2f} {rc:6.2f}")
        if grades:
            tot = sum(grades.values())
            print("  grade agreement: " + ", ".join(f"{k} {v / tot:.0%}" for k, v in grades.items()))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("folder")
    b.add_argument("--survey", required=True)
    b.add_argument("--out", default="qc_review")
    b.add_argument("--n", type=int, default=200)
    b.add_argument("--baseline", help="an older survey: include masks whose flags went away")
    b.add_argument("--jobs", type=int, default=max(2, (os.cpu_count() or 4) - 2))
    k = sub.add_parser("pack")
    k.add_argument("review", help="a folder written by build")
    k.add_argument("--out", help="default: <review>/review_phone.html")
    s = sub.add_parser("score")
    s.add_argument("labels")
    s.add_argument("--survey", required=True)
    a = ap.parse_args(argv)
    return {"build": build, "pack": pack, "score": score}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
