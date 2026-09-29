"""Build the documentation site and the example gallery.

    python tools/build_docs.py              # docs/site/ + docs/gallery/
    python tools/build_docs.py --no-gallery # just the site (fast)

The site is plain HTML generated from the package's own docstrings (no
Sphinx/MkDocs needed): an overview from README.md, the command-line
reference from the argparse parser, and an API reference for every public
function and module. The gallery holds three reports made from synthetic
lesions (a stroke, small-vessel disease, a tumour) -- no patient data.
"""
from __future__ import annotations

import argparse
import html
import inspect
import io
import os
import re
import sys
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
SITE = os.path.join(ROOT, "docs", "site")
GALLERY = os.path.join(ROOT, "docs", "gallery")

#: (module, [public names]) documented on the API page, in reading order.
API = [
    ("lqtpy", ["analyze_single", "analyze_group", "describe", "prepare", "lesion_stats",
               "atlas_overlap", "atlas_coverage", "lobe_overlap", "network_rollup_for",
               "tract_proxy_overlap", "tract_disconnection", "list_atlases", "load_atlas",
               "register_atlas", "preload"]),
    ("lqtpy.qc", ["check", "clusters", "repair_gaps", "clean", "check_folder", "clean_folder"]),
    ("lqtpy.tissue", ["composition", "locate"]),
    ("lqtpy.watershed", ["border_zones"]),
    ("lqtpy.structures", ["distances", "cross_section_load"]),
    ("lqtpy.mirror", ["mirror", "homologue"]),
    ("lqtpy.native", ["warp_to_mni", "native_qc"]),
    ("lqtpy.formats", ["load_image", "read_nrrd", "write_nrrd", "read_dicom_seg"]),
    ("lqtpy.metadata", ["load_metadata", "expected_sides"]),
    ("lqtpy.cohort", ["find_duplicates", "find_outliers", "compare"]),
    ("lqtpy.repro", ["rerun", "read_run_record"]),
    ("lqtpy.report", ["render_single", "render_group", "to_pdf"]),
    ("lqtpy.figures", ["glass_brain_svg", "slices_svg", "flat_map_svg", "viewer_data"]),
    ("lqtpy.registry", ["MeasureSpec", "Field", "Descriptor", "list_measures", "register_measure"]),
    ("lqtpy.datasets", ["fetch_suit", "get_data_home"]),
    ("lqtpy.scaffold", ["new_plugin"]),
    ("lqtpy.web", ["serve"]),
    ("lqtpy.watch", ["watch"]),
    ("lqtpy.config", ["load_config"]),
]

CSS = """
:root{color-scheme:light dark;--bg:#f4f5f7;--panel:#fff;--ink:#121820;--ink2:#4a5361;--ink3:#5f6877;
--line:#e3e6eb;--acc:#d9582a;--code:#f1f3f6;--link:#1f6fd1}
@media (prefers-color-scheme:dark){:root{--bg:#0d1015;--panel:#151a21;--ink:#e9edf2;--ink2:#b3bcc8;
--ink3:#8f98a6;--line:rgba(255,255,255,.1);--acc:#f07a45;--code:#1b212a;--link:#7ab3ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.6 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}
a{color:var(--link)}header{border-bottom:1px solid var(--line);padding:12px 24px;display:flex;gap:20px;
align-items:center;flex-wrap:wrap}header b{font-size:17px;letter-spacing:.02em}header nav{display:flex;gap:16px}
main{max-width:980px;margin:0 auto;padding:28px 24px 80px}
h1{font-size:30px;margin:.2em 0 .4em;letter-spacing:-.01em}h2{font-size:21px;margin:1.6em 0 .5em;
border-bottom:1px solid var(--line);padding-bottom:6px}h3{font-size:16px;margin:1.4em 0 .3em}
code,pre{font-family:ui-monospace,"SF Mono",Consolas,monospace;font-size:13.5px}
pre{background:var(--code);border:1px solid var(--line);border-radius:10px;padding:12px 14px;overflow-x:auto}
p code,li code,td code{background:var(--code);border-radius:5px;padding:1px 5px}
.fn{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:6px 18px 10px;margin:14px 0}
.fn h3{font-family:ui-monospace,Consolas,monospace;font-size:14.5px;word-break:break-word}
.fn .doc{white-space:pre-wrap;color:var(--ink2);font-size:14px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px 18px}
.card h3{margin-top:0}.muted{color:var(--ink3)}table{border-collapse:collapse;width:100%}
td,th{border-bottom:1px solid var(--line);padding:6px 8px;text-align:left;vertical-align:top}
"""


def _page(title, body, depth=0):
    up = "../" * depth
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{html.escape(title)} · LQTpy</title>'
            f'<style>{CSS}</style></head><body><header><b>LQTpy</b><nav>'
            f'<a href="{up}index.html">Overview</a><a href="{up}cli.html">Command line</a>'
            f'<a href="{up}api.html">Python API</a><a href="{up}gallery.html">Gallery</a></nav></header>'
            f"<main>{body}</main></body></html>")


def _md(text: str) -> str:
    """Just enough Markdown for README.md: headings, code blocks, lists,
    tables, inline code, links, bold."""
    out, in_code, in_list, table = [], False, False, []

    def inline(s):
        s = html.escape(s)
        s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
        s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", s)
        return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)

    def flush_table():
        if table:
            rows = [r for r in table if not re.match(r"^\|[\s:|-]+\|$", r)]
            cells = [[c.strip() for c in r.strip("|").split("|")] for r in rows]
            out.append("<table>" + "".join(
                "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{inline(c)}</{'th' if i == 0 else 'td'}>"
                                 for c in row) + "</tr>" for i, row in enumerate(cells)) + "</table>")
            table.clear()

    for line in text.splitlines():
        if line.startswith("```"):
            flush_table()
            out.append("</code></pre>" if in_code else "<pre><code>")
            in_code = not in_code
            continue
        if in_code:
            out.append(html.escape(line) + "\n")
            continue
        if line.startswith("|"):
            table.append(line)
            continue
        flush_table()
        if in_list and not line.startswith(("- ", "  ")):
            out.append("</ul>")
            in_list = False
        m = re.match(r"^(#{1,4}) (.*)", line)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
        elif line.startswith("- "):
            if not in_list:
                out.append("<ul>")
                in_list = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip().startswith("<"):
            continue  # raw HTML in the README (banners)
        elif line.strip():
            out.append(f"<p>{inline(line)}</p>")
    flush_table()
    if in_list:
        out.append("</ul>")
    return "".join(out)


def build_site() -> None:
    os.makedirs(SITE, exist_ok=True)
    with open(os.path.join(ROOT, "README.md"), encoding="utf-8") as fh:
        readme = fh.read()
    with open(os.path.join(SITE, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(_page("Overview", _md(readme)))

    from lqtpy.cli import build_parser
    parser = build_parser()
    buf = [f"<h1>Command line</h1><pre>{html.escape(parser.format_help())}</pre>"]
    sub = parser._subparsers._group_actions[0]
    for name, sp in sub.choices.items():
        buf.append(f'<h2 id="{name}">lqtpy {name}</h2><pre>{html.escape(sp.format_help())}</pre>')
    import lqtpy.config
    buf.append(f"<h2>Settings file</h2><pre>{html.escape(inspect.getdoc(lqtpy.config))}</pre>")
    with open(os.path.join(SITE, "cli.html"), "w", encoding="utf-8") as fh:
        fh.write(_page("Command line", "".join(buf)))

    import importlib
    parts = ["<h1>Python API</h1><p>Every public function, with its docstring. "
             "Modules: " + ", ".join(f'<a href="#{m}">{m}</a>' for m, _ in API) + ".</p>"]
    for mod_name, names in API:
        mod = importlib.import_module(mod_name)
        doc = inspect.getdoc(mod) or ""
        parts.append(f'<h2 id="{mod_name}">{mod_name}</h2><p class="muted" style="white-space:pre-wrap">'
                     f"{html.escape(doc.split(chr(10) + chr(10))[0])}</p>")
        for n in names:
            obj = getattr(mod, n, None)
            if obj is None:
                continue
            try:
                sig = str(inspect.signature(obj))
            except (TypeError, ValueError):
                sig = ""
            parts.append(f'<div class="fn" id="{mod_name}.{n}"><h3>{n}{html.escape(sig)}</h3>'
                         f'<div class="doc">{html.escape(inspect.getdoc(obj) or "")}</div></div>')
    with open(os.path.join(SITE, "api.html"), "w", encoding="utf-8") as fh:
        fh.write(_page("Python API", "".join(parts)))

    cards = "".join(
        f'<div class="card"><h3><a href="../gallery/{k}.html">{t}</a></h3><p class="muted">{d}</p></div>'
        for k, t, d in GALLERY_ITEMS)
    with open(os.path.join(SITE, "gallery.html"), "w", encoding="utf-8") as fh:
        fh.write(_page("Gallery", "<h1>Gallery</h1><p>Reports made from synthetic lesions (no patient "
                                  "data). Each is one self-contained file: switch between the dashboard "
                                  "and the clinical report, light and dark, or export a PDF.</p>"
                                  f'<div class="cards">{cards}</div>'))


GALLERY_ITEMS = [
    ("stroke", "Stroke (MCA territory)", "One cortico-subcortical lesion in the left middle cerebral "
     "artery territory: lobes, regions, functional labels, nearby tracts."),
    ("small_vessel", "Small-vessel disease", "Many small lesions: counts by location "
     "(periventricular, deep white matter, infratentorial) instead of a paragraph each."),
    ("tumour", "Tumour", "One large mass: where it is centred, what it extends into, midline "
     "crossing, shape extras."),
]


def build_gallery() -> None:
    warnings.simplefilter("ignore")
    import lqtpy
    import synthetic as syn
    from lqtpy.report import render_single
    os.makedirs(GALLERY, exist_ok=True)
    lesions = {
        "stroke": (syn.blobs([((-40, 10, 10), 14), ((-35, -10, 15), 10)]), "stroke", None),
        "small_vessel": (syn.blobs([((-24, 8, 22), 3), ((26, -6, 24), 3), ((-28, -30, 26), 4),
                                    ((18, 28, 12), 3), ((0, -62, -36), 4), ((-20, -12, 30), 2.5),
                                    ((22, 14, 28), 2.5)]), "small_vessel", None),
        "tumour": (syn.ellipsoid((-22, 28, 18), (24, 20, 18)), "tumour", "all"),
    }
    for key, (mask, prof, extras) in lesions.items():
        res = lqtpy.analyze_single(syn.img(mask), write=False, plots=True, profile=prof,
                                   extras=extras)
        res["name"] = {"stroke": "Example: stroke", "small_vessel": "Example: small-vessel disease",
                       "tumour": "Example: tumour"}[key]
        res["source_path"] = None
        with open(os.path.join(GALLERY, f"{key}.html"), "w", encoding="utf-8") as fh:
            fh.write(render_single(res))
        print(f"gallery  {key}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-gallery", action="store_true")
    a = ap.parse_args()
    build_site()
    print(f"site     {os.path.relpath(SITE, ROOT)}")
    if not a.no_gallery:
        build_gallery()
