"""High-level orchestration: single-lesion and group analysis.

``analyze_single`` and ``analyze_group`` run the whole pipeline on a
:class:`~lqtpy.prepared.PreparedLesion` -- loaded and binarized once --
and return plain records plus a run ``manifest`` that says exactly which
modules ran, which were skipped or unavailable, and why. Reports render
their scope block from that manifest.

Pipeline per lesion
-------------------
core    morphometry, QC flags, per-atlas overlap + coverage (+ network /
        lobe / homolog rollups), white-matter tract proxy, streamline
        disconnection (when an index is installed), plain-language
        description.
extras  registered extra measures (e.g. ``shape``; plugins such as fractal
        dimension), requested with ``extras="all"`` or a list of names.

``plots=False`` (the default) never imports matplotlib/nilearn/pandas.
"""
from __future__ import annotations

import csv
import datetime
import json
import os
import time

import numpy as np
import nibabel as nib

from . import _format as _fmt
from . import registry
from . import tracts as _tracts
from .atlases import DEFAULT_ATLASES, load_atlas
from .errors import DataUnavailableError, InvalidLesionError, MissingDependencyError
from .manifest import FAILED, RAN, SKIPPED, UNAVAILABLE, Manifest
from .morphometry import compute_morphometry
from .overlap import (_COLUMNS as _DAMAGE_COLUMNS, _HOMOLOG_COLUMNS,
                      _LOBE_COLUMNS, _NETWORK_COLUMNS, atlas_has_lobes, coverage,
                      homolog_symmetry, lobe_rollup, network_rollup, region_overlap_from_mask)
from .prepared import prepare
from .tracts import _COLUMNS as _TRACT_COLUMNS

_GROUP_DAMAGE_COLUMNS = ["RegionName", "Group", "NSubjects", "NSubjectsTotal",
                         "CohortMeanPercentDamage", "MeanPercentDamageAffected",
                         "MaxPercentDamage"]
_GROUP_NETWORK_COLUMNS = ["Network", "NSubjects", "NSubjectsTotal",
                          "CohortMeanPercentDamage", "MeanPercentDamageAffected",
                          "MaxPercentDamage"]
_GROUP_LOBE_COLUMNS = ["Lobe", "NSubjects", "NSubjectsTotal",
                       "CohortMeanPercentDamage", "MeanPercentDamageAffected",
                       "MaxPercentDamage"]
_GROUP_TRACT_COLUMNS = ["Tract", "NSubjects", "NSubjectsTotal",
                        "CohortMeanPercentOverlap", "MeanPercentOverlapAffected",
                        "MaxPercentOverlap"]
_DISCON_COLUMNS = ["tract", "name", "n_streamlines", "n_disconnected", "percent_disconnected"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _stem(path) -> str:
    """Output-file stem for a lesion source; non-path sources get "lesion"."""
    if not isinstance(path, (str, os.PathLike)):
        return "lesion"
    b = os.path.basename(os.fspath(path))
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def _atlas_keys(atlases) -> "list[str]":
    """None -> the defaults; a bare string -> a one-item list (``"aal"`` must
    not be iterated per character); duplicates dropped, order kept."""
    if not atlases:
        return list(DEFAULT_ATLASES)
    if isinstance(atlases, str):
        atlases = [atlases]
    return list(dict.fromkeys(a.lower() for a in atlases))


def _fmt_elapsed(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{sec / 60:.1f} min"


def _write_csv(path, records, columns):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependencyError(
            "tables",
            "This requires pandas. Install the 'tables' extra for a "
            "frame='pandas' conversion, or the 'report'/'plots' extras to "
            "generate reports/plots: pip install 'lqtpy[tables]'") from exc
    return pd


def _frame(records, columns):
    """A DataFrame for the plotting helpers (and ``frame="pandas"``)."""
    if records is None:
        return None
    return _pandas().DataFrame(records, columns=columns)


def _ggseg_dict(atlas, records, value_col="PercentDamage") -> dict:
    if not atlas.ggseg or not records:
        return {}
    return {atlas.ggseg_map[r["RegionName"]]: float(r[value_col])
            for r in records if r["RegionName"] in atlas.ggseg_map}


_ASEG_BASE = {"Hippocampus": "Hippocampus", "Amygdala": "Amygdala",
              "Caudate": "Caudate", "Putamen": "Putamen",
              "Pallidum": "Pallidum", "Thalamus": "Thalamus"}


def _aal_to_aseg(records, value_col="PercentDamage") -> dict:
    """Map AAL subcortical/cerebellar damage onto ggseg aseg regions."""
    if not records:
        return {}
    out, cereb = {}, []
    for r in records:
        name, val = r["RegionName"], r[value_col]
        base = name.rsplit("_", 1)[0]
        hemi = "Left" if name.endswith("_L") else "Right" if name.endswith("_R") else None
        if base in _ASEG_BASE and hemi:
            out[f"{hemi}-{_ASEG_BASE[base]}"] = float(val)
        elif name.startswith(("Cerebelum", "Vermis")):
            cereb.append(float(val))
    if cereb:
        out["Cerebellum-Cortex"] = max(cereb)
    return out


def _flat_maps(atlas, records, value_col="PercentDamage"):
    """``[(caption, {"light": uri, "dark": uri})]`` ggseg-style flat maps for
    an atlas, one render per report theme."""
    from . import plotting
    from .ggseg_flat import dk_flat_png, schaefer_flat_png
    maps = []
    for theme in ("light", "dark"):
        if atlas.key == "dk":
            uri = dk_flat_png(_ggseg_dict(atlas, records, value_col), theme=theme)
            cap = "Cortical damage (DKT), percent of each region damaged"
        elif atlas.key == "schaefer":
            data = {r["RegionName"]: float(r[value_col]) for r in records} if records else {}
            uri = schaefer_flat_png(data, theme=theme)
            cap = "Cortical damage (Schaefer-100), percent of each parcel damaged"
        elif atlas.key == "aal":
            uri = plotting.ggseg_aseg_png(_aal_to_aseg(records, value_col), theme=theme)
            cap = "Subcortical and cerebellar damage (AAL, drawn on aseg geometry)"
        else:
            return []
        if not uri:
            return []
        if not maps:
            maps.append((cap, {}))
        maps[0][1][theme] = uri
    return maps


def _require_plot_stack() -> None:
    """Fail early, with the extra to install, when plots are requested
    without matplotlib/nilearn/pandas."""
    import importlib
    missing = []
    for mod in ("matplotlib", "nilearn", "pandas"):
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise MissingDependencyError(
            "plots", f"plots=True needs {', '.join(missing)}. Install the 'plots' extra: "
                     "pip install 'lqtpy[plots]'")


def _extra_names(extras) -> "list[str]":
    """Resolve the ``extras`` argument to measure names."""
    if not extras or extras == "none":
        return []
    available = [m.name for m in registry.list_measures("extra", available_only=False)]
    if extras == "all":
        return available
    if isinstance(extras, str):
        extras = [e.strip() for e in extras.split(",") if e.strip()]
    unknown = [e for e in extras if e not in available]
    if unknown:
        raise ValueError(f"unknown extra measure(s) {unknown}; available: {available}")
    return list(extras)


_PART_COLUMNS = ["Part", "Voxels", "VolumeMM3", "Location", "TissueClass", "PctCortex", "PctWhiteMatter",
                 "PctDeepGrey", "PctCerebellum", "PctBrainstem", "PctVentricles", "PctCSF",
                 "PctOutside", "DistVentricleMM", "DistCortexMM", "MaxAxialMM"]
_TISSUE_KEYS = [("PctCortex", "cortex"), ("PctWhiteMatter", "white_matter"),
                ("PctDeepGrey", "deep_grey"), ("PctCerebellum", "cerebellum"),
                ("PctBrainstem", "brainstem"), ("PctVentricles", "ventricles"),
                ("PctCSF", "csf"), ("PctOutside", "outside")]
_TISSUE_COLUMNS = ["TissueClass", "PctCortex", "PctWhiteMatter", "PctDeepGrey", "PctInfratentorial",
                   "NPeriventricular", "NJuxtacortical", "NCortical", "NDeepWhiteMatter",
                   "NDeepGrey", "NInfratentorial", "NOutsideBrain"]


def _part_row(pt: dict) -> dict:
    row = {"Part": pt["id"], "Voxels": pt["voxels"], "VolumeMM3": pt["volume_mm3"],
           "Location": pt["class"], "TissueClass": pt.get("tissue_class") or "",
           "DistVentricleMM": pt["dist_ventricle_mm"],
           "DistCortexMM": pt["dist_cortex_mm"], "MaxAxialMM": pt["max_axial_mm"]}
    row.update({col: pt["composition"].get(key, 0.0) for col, key in _TISSUE_KEYS})
    return row


def _tissue_cols(t: dict) -> dict:
    comp = t.get("composition") or {}
    counts = t.get("counts") or {}
    return {"TissueClass": t.get("class") or "",
            "PctCortex": comp.get("cortex", 0.0), "PctWhiteMatter": comp.get("white_matter", 0.0),
            "PctDeepGrey": comp.get("deep_grey", 0.0),
            "PctInfratentorial": round(comp.get("cerebellum", 0.0) + comp.get("brainstem", 0.0), 1),
            "NPeriventricular": counts.get("periventricular", 0),
            "NJuxtacortical": counts.get("juxtacortical", 0), "NCortical": counts.get("cortical", 0),
            "NDeepWhiteMatter": counts.get("deep white matter", 0),
            "NDeepGrey": counts.get("deep grey matter", 0),
            "NInfratentorial": counts.get("infratentorial", 0),
            "NOutsideBrain": counts.get("outside the brain", 0)}


def _count_by(values) -> dict:
    out: dict = {}
    for v in values:
        if v:
            out[v] = out.get(v, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# --------------------------------------------------------------------------- #
# One lesion, no file output
# --------------------------------------------------------------------------- #
def _analyse(p, atlases, *, kind, tracts, plots, qc, describe, profile, extras,
             disconnection, expected_side, threshold) -> dict:
    from . import datasets as _datasets
    from . import qc as _qc

    man = Manifest(kind, settings={"threshold": threshold, "atlases": atlases,
                                   "profile": profile if describe else None,
                                   "extras": extras or "none"})
    t0 = time.perf_counter()
    morph = compute_morphometry(p)
    man.add("morphometry", "Core morphometry", RAN)

    flags = []
    if qc:
        flags = _qc.check(p, expected_side=expected_side)
        man.add("qc", "Mask quality checks", RAN)
    else:
        man.add("qc", "Mask quality checks", SKIPPED, reason="turned off (qc=False)")

    from . import tissue as _tissue
    loc = _tissue.locate(p)
    tissue_res = {"class": loc["class"], "composition": loc["composition"],
                  "counts": loc["counts"], "contact_mm": loc["contact_mm"],
                  "parts": loc["parts"]}
    man.add("tissue", "Tissue composition and location classes", RAN)

    glass_uri = slices_uri = glass_light = slices_light = None
    if plots:
        from . import plotting
        com = morph["center_of_mass_mm"]
        glass_uri = plotting.glass_brain_png(p.img)
        slices_uri = plotting.slices_png(p.img, cut_coords=com)
        # White-background versions: light mode, clinical report, PDF.
        glass_light = plotting.glass_brain_png(p.img, theme="light")
        slices_light = plotting.slices_png(p.img, cut_coords=com, theme="light")

    atlas_results = []
    for key in atlases:
        atlas = load_atlas(key)
        gm = p.on_grid(atlas.img)
        df = region_overlap_from_mask(gm.mask, atlas)
        entry = {"key": key, "name": atlas.name, "tier": atlas.tier, "df": df,
                 "coverage": coverage(p, atlas)}
        if key in ("schaefer", "arterial"):
            # Yeo-7 networks / major vascular territories (the atlas's Group).
            entry["network_df"] = network_rollup(df, atlas, lesion_total=gm.n_voxels)
        if key == "aal":
            entry["homolog_df"] = homolog_symmetry(df)
        if atlas_has_lobes(atlas):
            entry["lobe_df"] = lobe_rollup(df, atlas, lesion_total=gm.n_voxels)
        if plots:
            entry["flat_uris"] = _flat_maps(atlas, df)
        atlas_results.append(entry)
        man.add(f"atlas:{key}", atlas.name, RAN, tier=atlas.tier)

    tract_res = None
    if not tracts:
        man.add("tract_proxy", "White-matter tract overlap (proxy)", SKIPPED,
                reason="turned off (tracts=False)")
    elif not _tracts.tracts_available():
        man.add("tract_proxy", "White-matter tract overlap (proxy)", UNAVAILABLE,
                reason="tract proxy index missing from the installation")
    else:
        tdf = _tracts.tract_overlap(p)
        tract_res = {"df": tdf}
        man.add("tract_proxy", "White-matter tract overlap (proxy)", RAN)

    discon = None
    label = "Streamline disconnection"
    if disconnection is False:
        man.add("disconnection", label, SKIPPED, reason="turned off (disconnection=False)")
    elif not _datasets.discon_index_available():
        if disconnection is True:
            _datasets.get_discon_index_path()  # raises DataUnavailableError with instructions
        man.add("disconnection", label, UNAVAILABLE,
                reason="no disconnection index installed (tools/build_discon_index.py)")
    else:
        from .api import tract_disconnection
        try:
            discon = tract_disconnection(p)
            man.add("disconnection", label, RAN)
        except DataUnavailableError as exc:
            man.add("disconnection", label, FAILED, reason=str(exc).splitlines()[0])

    extra_names = _extra_names(extras)
    measures = registry.run_measures(p, extra_names, manifest=man) if extra_names else {}
    for spec in registry.list_measures("extra", available_only=False):
        if spec.name not in extra_names:
            man.add(f"measure:{spec.name}", spec.label, SKIPPED, tier="extra",
                    reason="not requested")

    description = None
    if describe:
        from .describe import describe as _describe
        d = _describe(p, profile=profile, qc_flags=flags,
                      shape="shape" in measures)
        description = {"text": d.text, "sections": d.sections, "marked": d.marked,
                       "profile": d.profile, "facts": d.facts}
        man.add("description", "Plain-language description", RAN, profile=profile)
    else:
        man.add("description", "Plain-language description", SKIPPED,
                reason="turned off (describe=False)")

    return {
        "name": p.name,
        "source_path": p.source if isinstance(p.source, (str, os.PathLike)) else None,
        "was_binary": p.was_binary,
        "morphometry": morph,
        "qc": [f.to_dict() for f in flags],
        "tissue": tissue_res,
        "description": description,
        "measures": {k: {"values": v["values"], "descriptors": [d.text for d in v["descriptors"]]}
                     for k, v in measures.items()},
        "glass_uri": glass_uri,
        "slices_uri": slices_uri,
        "glass_uri_light": glass_light,
        "slices_uri_light": slices_light,
        "atlases": atlas_results,
        "tracts": tract_res,
        "disconnection": discon,
        "manifest": man.to_dict(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "compute_seconds": round(time.perf_counter() - t0, 3),
    }


def _write_single(result, out_dir, name, *, plots, report_style=None, back_link=None,
                  pdf=False):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{name}_morphometry.json"), "w", encoding="utf-8") as fh:
        json.dump(result["morphometry"], fh, indent=2)
    summary = {"name": name, "qc": result["qc"], "manifest": result["manifest"],
               "measures": result["measures"],
               "description": (result["description"] or {}).get("text"),
               "tissue": {k: v for k, v in (result.get("tissue") or {}).items() if k != "parts"},
               "coverage": {a["key"]: a["coverage"] for a in result["atlases"]}}
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    if result.get("tissue") and result["tissue"]["parts"]:
        _write_csv(os.path.join(out_dir, f"{name}_parts.csv"),
                   [_part_row(pt) for pt in result["tissue"]["parts"]], _PART_COLUMNS)
    for a in result["atlases"]:
        _write_csv(os.path.join(out_dir, f"{name}_{a['key']}_damage.csv"), a["df"], _DAMAGE_COLUMNS)
        if a.get("lobe_df"):
            _write_csv(os.path.join(out_dir, f"{name}_{a['key']}_lobes.csv"), a["lobe_df"],
                       _LOBE_COLUMNS)
    if result["tracts"] is not None:
        _write_csv(os.path.join(out_dir, f"{name}_tracts.csv"), result["tracts"]["df"],
                   _TRACT_COLUMNS)
    if result["disconnection"] is not None:
        _write_csv(os.path.join(out_dir, f"{name}_disconnection.csv"), result["disconnection"],
                   _DISCON_COLUMNS)
    if result["description"]:
        with open(os.path.join(out_dir, f"{name}_description.txt"), "w", encoding="utf-8") as fh:
            fh.write(result["description"]["text"] + "\n")
    if plots:
        from .report import render_single
        path = os.path.join(out_dir, f"{name}_report.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_single(result, report_style, back_link=back_link))
        result["report_path"] = path
        if pdf:
            from .report import to_pdf
            result["pdf_path"] = to_pdf(path)


def _apply_frame_single(result):
    for a in result["atlases"]:
        a["df"] = _frame(a.get("df"), _DAMAGE_COLUMNS)
        for k, cols in (("network_df", _NETWORK_COLUMNS), ("homolog_df", _HOMOLOG_COLUMNS),
                        ("lobe_df", _LOBE_COLUMNS)):
            if k in a:
                a[k] = _frame(a.get(k), cols)
    if result.get("tracts") is not None:
        result["tracts"]["df"] = _frame(result["tracts"].get("df"), _TRACT_COLUMNS)


# --------------------------------------------------------------------------- #
# Single lesion
# --------------------------------------------------------------------------- #
def analyze_single(lesion_path, atlases=None, out_dir=None, threshold=0.5,
                   tracts=True, write=True, plots=False, verbose=False, frame=None, *,
                   qc=True, describe=True, profile="stroke", extras=None,
                   disconnection="auto", expected_side=None, report_style=None, pdf=False):
    """Analyse one lesion; return a result dict and (optionally) write CSV/JSON
    tables (``write=True``) and a self-contained HTML report
    (``write=True`` and ``plots=True``).

    ``lesion_path`` accepts anything :func:`lqtpy.prepare` does. Core
    modules always run unless turned off (``qc``, ``describe``, ``tracts``,
    ``disconnection`` -- the latter runs automatically when an index is
    installed). ``extras`` selects extra measures: ``None`` (core only),
    ``"all"``, or a list of names from :func:`lqtpy.registry.list_measures`.
    ``profile`` picks the description style (stroke / small_vessel / tumour
    / research). ``result["manifest"]`` records what ran and what didn't.

    ``pdf=True`` (with ``write`` and ``plots``) also saves the report as a
    PDF next to it, via :func:`lqtpy.report.to_pdf`.

    Tables are records by default; ``frame="pandas"`` returns DataFrames.
    """
    t0 = time.time()
    atlases = _atlas_keys(atlases)
    if plots:
        _require_plot_stack()
    p = prepare(lesion_path, threshold)
    if verbose:
        print(f"[lqtpy] analysing {p.name} ...")
    result = _analyse(p, atlases, kind="single", tracts=tracts, plots=plots, qc=qc,
                      describe=describe, profile=profile, extras=extras,
                      disconnection=disconnection, expected_side=expected_side,
                      threshold=threshold)
    result["elapsed"] = _fmt_elapsed(time.time() - t0)
    if write:
        out_dir = out_dir or os.path.join("lqtpy_output", p.name)
        _write_single(result, out_dir, p.name, plots=plots, report_style=report_style,
                      pdf=pdf)
        if verbose:
            suffix = f" -> {result['report_path']}" if "report_path" in result else ""
            print(f"[lqtpy] done in {result['elapsed']}{suffix}")
    if frame == "pandas":
        _apply_frame_single(result)
    return result


# --------------------------------------------------------------------------- #
# Group
# --------------------------------------------------------------------------- #
def _normalise_inputs(inputs):
    """Accept a directory, a single lesion file, a list of paths, or a list of
    (name, lesion). A 2-tuple is only treated as ``(name, lesion)`` when its
    first element is a string, so a bare ``(array, affine)`` tuple passed as
    a list item is one lesion source. Duplicate names get a numeric suffix."""
    if isinstance(inputs, (str, os.PathLike)):
        path = os.fspath(inputs)
        if os.path.isdir(path):
            files = sorted(os.path.join(path, f) for f in os.listdir(path)
                           if f.endswith((".nii", ".nii.gz")))
            inputs = files
        else:
            # A single lesion file: a one-subject group. Iterating the string
            # would treat each character as a path.
            inputs = [path]
    out, seen = [], {}
    for item in inputs:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            name, src = item[0], item[1]
        else:
            name, src = _stem(item), item
        k = seen.get(name, 0)
        seen[name] = k + 1
        out.append((f"{name}_{k + 1}" if k else name, src))
    return out


def _aggregate(frames, key, value, n_total, extra_first=None):
    """Per-key group stats over subjects: NSubjects (affected), NSubjectsTotal,
    cohort mean (unaffected subjects count as 0), mean over affected
    subjects, and max."""
    grouped: "dict[str, dict]" = {}
    for f in frames:
        for r in f or []:
            g = grouped.get(r[key])
            if g is None:
                g = grouped[r[key]] = {"sum": 0.0, "max": float("-inf"), "n": 0}
                if extra_first:
                    g[extra_first] = r.get(extra_first)
            g["sum"] += r[value]
            g["max"] = max(g["max"], r[value])
            g["n"] += 1
    rows = []
    for k in sorted(grouped):
        g = grouped[k]
        row = {key: k}
        if extra_first:
            row[extra_first] = g[extra_first]
        suffix = "Overlap" if "Overlap" in value else "Damage"
        row.update({
            "NSubjects": g["n"],
            "NSubjectsTotal": n_total,
            f"CohortMeanPercent{suffix}": round(g["sum"] / n_total, 2) if n_total else 0.0,
            f"MeanPercent{suffix}Affected": round(g["sum"] / g["n"], 2),
            f"MaxPercent{suffix}": round(g["max"], 2),
        })
        rows.append(row)
    suffix = "Overlap" if "Overlap" in value else "Damage"
    rows.sort(key=lambda r: (-r["NSubjects"], -r[f"CohortMeanPercent{suffix}"]))
    return rows


def _worst(flags):
    order = {"error": 0, "warn": 1, "info": 2}
    return min((f["severity"] for f in flags), key=lambda s: order[s], default=None)


def analyze_group(inputs, atlases=None, out_dir=None, threshold=0.5,
                  tracts=True, write=True, plots=False, verbose=False, frame=None, *,
                  qc=True, describe=True, profile="stroke", extras=None,
                  disconnection="auto", subject_reports=True, report_style=None, pdf=False):
    """Analyse a group of lesions; return a result dict and (optionally) write
    group CSVs, a group HTML report (``plots=True``) and -- with
    ``subject_reports=True`` -- one full report per subject, linked from the
    group report.

    ``inputs``: a directory, a single lesion file, or a list of lesion
    sources or ``(name, source)`` pairs. Group tables report, per region,
    how many subjects are affected, the **cohort** mean (unaffected
    subjects count as 0), the mean over affected subjects, and the max.
    """
    t0 = time.time()
    atlases = _atlas_keys(atlases)
    if plots:
        _require_plot_stack()
    pairs = _normalise_inputs(inputs)
    if len(pairs) < 1:
        raise InvalidLesionError("No lesion files found for group analysis.")
    n = len(pairs)
    if verbose:
        print(f"[lqtpy] group analysis: {n} subjects")

    ref_atlas = load_atlas(atlases[0])
    ref_img = ref_atlas.img
    freq = np.zeros(ref_img.shape[:3], dtype=np.int32)
    out_dir = out_dir or os.path.join("lqtpy_output", "group")
    per_report = bool(write and plots and subject_reports)

    subjects = []
    per_atlas = {k: [] for k in atlases}
    per_lobe = {k: [] for k in atlases}
    per_net = {k: [] for k in atlases}
    tract_frames = []
    qc_counts: "dict[str, int]" = {}
    tissue_counts: "dict[str, int]" = {}
    subject_manifest = None
    for i, (name, src) in enumerate(pairs, start=1):
        p = prepare(src, threshold, name=name)
        ts = time.time()
        res = _analyse(p, atlases, kind="group-member", tracts=tracts, plots=per_report,
                       qc=qc, describe=describe, profile=profile, extras=extras,
                       disconnection=disconnection, expected_side=None, threshold=threshold)
        res["elapsed"] = _fmt_elapsed(time.time() - ts)
        subject_manifest = res["manifest"]
        freq += p.on_grid(ref_img).mask
        for a in res["atlases"]:
            per_atlas[a["key"]].append(a["df"])
            if a.get("lobe_df"):
                per_lobe[a["key"]].append(a["lobe_df"])
            if a.get("network_df") is not None:
                per_net[a["key"]].append(a["network_df"])
        if res["tracts"] is not None:
            tract_frames.append(res["tracts"]["df"])
        for f in res["qc"]:
            if f["severity"] != "info":
                qc_counts[f["id"]] = qc_counts.get(f["id"], 0) + 1
        ref_df = res["atlases"][0]["df"]
        summary_para = ""
        if res["description"]:
            summary_para = dict(res["description"]["sections"]).get("Summary", "")
        facts = (res["description"] or {}).get("facts") or {}
        terrs = facts.get("territories") or []
        main_terr = (_fmt.territory_short(terrs[0]["hemisphere"], terrs[0]["territory"])
                     if terrs else None)
        tis = res.get("tissue") or {}
        tissue_counts = {k: tissue_counts.get(k, 0) + v for k, v in (tis.get("counts") or {}).items()}
        entry = {"name": name, "morphometry": res["morphometry"],
                 "tissue": {k: v for k, v in tis.items() if k != "parts"},
                 "top_region": ref_df[0]["RegionName"] if ref_df else "-",
                 "territory": main_terr,
                 "qc": res["qc"], "qc_worst": _worst(res["qc"]),
                 "summary": summary_para}
        if write:
            sub_dir = os.path.join(out_dir, "subjects", name)
            _write_single(res, sub_dir, name, plots=per_report, report_style=report_style,
                          back_link=os.path.join("..", "..", "group_report.html"))
            if "report_path" in res:
                entry["report_path"] = os.path.relpath(res["report_path"], out_dir)
        subjects.append(entry)
        if verbose:
            print(f"[lqtpy]   [{i}/{n}] {name}: {_fmt.volume(res['morphometry']['volume_mm3'])}")

    freq_img = nib.Nifti1Image(freq, ref_img.affine)
    overlap_uri = overlap_light = None
    if plots:
        from . import plotting
        overlap_uri = plotting.group_overlap_png(freq_img, n)
        overlap_light = plotting.group_overlap_png(freq_img, n, theme="light")

    atlas_results = []
    for key in atlases:
        atlas = load_atlas(key)
        agg = _aggregate(per_atlas[key], "RegionName", "PercentDamage", n, extra_first="Group")
        entry = {"key": key, "name": atlas.name, "tier": atlas.tier, "df": agg}
        if per_lobe[key]:
            entry["lobe_df"] = _aggregate(per_lobe[key], "Lobe", "PercentDamage", n)
        if per_net[key]:
            entry["network_df"] = _aggregate(per_net[key], "Network", "PercentDamage", n)
        if plots:
            entry["flat_uris"] = _flat_maps(atlas, agg, "CohortMeanPercentDamage")
        atlas_results.append(entry)

    tract_agg = _aggregate(tract_frames, "Tract", "PercentOverlap", n) if tract_frames else None

    man = Manifest("group", settings={"threshold": threshold, "atlases": atlases,
                                      "profile": profile if describe else None,
                                      "extras": extras or "none", "n_subjects": n})
    for m in (subject_manifest or {}).get("modules", []):
        man.add(m["id"], m["label"], m["status"], tier=m["tier"], reason=m.get("reason"))
    man.add("subject_reports", "One report per subject",
            RAN if per_report else SKIPPED,
            reason=None if per_report else ("needs write=True and plots=True" if not (write and plots)
                                            else "turned off (subject_reports=False)"))

    vols = sorted(s["morphometry"]["volume_cc"] for s in subjects)
    lis = [s["morphometry"]["laterality_index"] for s in subjects]
    result = {
        "n_subjects": n,
        "subjects": subjects,
        "median_volume_cc": round(float(np.median(vols)), 2),
        "median_laterality_index": round(float(np.median(lis)), 3),
        "n_bilateral": sum(1 for s in subjects if s["morphometry"].get("bilateral")),
        "max_overlap": int(freq.max()),
        "qc_counts": qc_counts,
        "tissue_classes": _count_by(s["tissue"].get("class") for s in subjects),
        "tissue_part_counts": tissue_counts,
        "n_subjects_with_warnings": sum(1 for s in subjects if s["qc_worst"] in ("warn", "error")),
        "overlap_uri": overlap_uri,
        "overlap_uri_light": overlap_light,
        "atlases": atlas_results,
        "tracts": None if tract_agg is None else {"df": tract_agg},
        "manifest": man.to_dict(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed": _fmt_elapsed(time.time() - t0),
    }

    if write:
        os.makedirs(out_dir, exist_ok=True)
        nib.save(freq_img, os.path.join(out_dir, "lesion_overlap_map.nii.gz"))
        for a in atlas_results:
            _write_csv(os.path.join(out_dir, f"group_{a['key']}_damage.csv"), a["df"],
                       _GROUP_DAMAGE_COLUMNS)
            if a.get("lobe_df"):
                _write_csv(os.path.join(out_dir, f"group_{a['key']}_lobes.csv"), a["lobe_df"],
                           _GROUP_LOBE_COLUMNS)
            if a.get("network_df"):
                _write_csv(os.path.join(out_dir, f"group_{a['key']}_networks.csv"),
                           a["network_df"], _GROUP_NETWORK_COLUMNS)
        if tract_agg:
            _write_csv(os.path.join(out_dir, "group_tracts.csv"), tract_agg, _GROUP_TRACT_COLUMNS)
        morph_keys = list(subjects[0]["morphometry"].keys())
        rows = [{"Subject": s["name"], "QC": _fmt.quality_label(s["qc_worst"]),
                 "QCFlags": ";".join(f["id"] for f in s["qc"] if f["severity"] != "info"),
                 **_tissue_cols(s["tissue"]), **s["morphometry"]} for s in subjects]
        _write_csv(os.path.join(out_dir, "group_subject_summary.csv"), rows,
                   ["Subject", "QC", "QCFlags"] + _TISSUE_COLUMNS + morph_keys)
        if plots:
            from .report import render_group
            path = os.path.join(out_dir, "group_report.html")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(render_group(result, report_style))
            result["report_path"] = path
            if pdf:
                from .report import to_pdf
                result["pdf_path"] = to_pdf(path)
        if verbose:
            suffix = f" -> {result['report_path']}" if "report_path" in result else ""
            print(f"[lqtpy] group done in {result['elapsed']}{suffix}")

    if frame == "pandas":
        for a in result["atlases"]:
            a["df"] = _frame(a["df"], _GROUP_DAMAGE_COLUMNS)
            if "lobe_df" in a:
                a["lobe_df"] = _frame(a["lobe_df"], _GROUP_LOBE_COLUMNS)
            if "network_df" in a:
                a["network_df"] = _frame(a["network_df"], _GROUP_NETWORK_COLUMNS)
        if result["tracts"] is not None:
            result["tracts"]["df"] = _frame(result["tracts"]["df"], _GROUP_TRACT_COLUMNS)
    return result
