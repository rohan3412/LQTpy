"""High-level orchestration: single-lesion and group analysis.

``analyze_single`` and ``analyze_group`` run the whole pipeline on a
:class:`~lqtpy.prepared.PreparedLesion` -- loaded and binarized once --
and return plain records plus a run ``manifest`` that says exactly which
modules ran, which were skipped or unavailable, and why. Reports render
their scope block from that manifest, and embed its run record so
``lqtpy rerun`` can repeat the analysis.

Pipeline per lesion
-------------------
core    morphometry, QC flags (native-space drawing checks when a native
        mask is given), per-atlas overlap + coverage (+ network / lobe /
        homolog rollups), tissue and location classes, arterial border
        zones, distances to key structures, the mirror-image homologue,
        white-matter tract proxy + cross-section load, streamline
        disconnection (when an index is installed), plain-language
        description.
extras  registered extra measures (e.g. ``shape``; plugins such as fractal
        dimension), requested with ``extras="all"`` or a list of names.

Group runs add subject metadata, a left/right flip check, duplicate and
outlier detection, a two-group comparison, anonymised output, parallel
workers and incremental re-runs (see :func:`analyze_group`).

``plots=True`` draws figures with numpy only (:mod:`lqtpy.figures`);
matplotlib, nilearn and pandas are never needed for a report.
"""
from __future__ import annotations

import csv
import datetime
import hashlib
import json
import os
import time
import warnings

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

__all__ = ["analyze_single", "analyze_group"]

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
_KEY_COLUMNS = ["Structure", "Hemisphere", "DistanceMM", "Reaches"]
_CROSS_COLUMNS = ["Tract", "Hemi", "PercentOverlap", "MaxCrossSectionLoad", "BottleneckLoad",
                  "BottleneckAreaMM2"]
_COMPARE_COLUMNS = ["Name", "NAffectedA", "NA", "PctAffectedA", "MeanA", "NAffectedB", "NB",
                    "PctAffectedB", "MeanB", "DiffMean", "PFisher", "QFisher", "PMannWhitney",
                    "QMannWhitney"]

#: Legacy positional order after (lesion, atlases, out_dir), before 0.5.
_LEGACY_POSITIONAL = ("threshold", "tracts", "write", "plots", "verbose", "frame")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _stem(path) -> str:
    """Output-file stem for a lesion source; non-path sources get "lesion"."""
    if not isinstance(path, (str, os.PathLike)):
        return "lesion"
    from .formats import strip_extension
    return strip_extension(os.path.basename(os.fspath(path)))


def _atlas_keys(atlases) -> "list[str]":
    """None -> the defaults; a bare string -> a one-item list (``"aal"`` must
    not be iterated per character; ``"aal,dk"`` is split); duplicates
    dropped, order kept."""
    if not atlases:
        return list(DEFAULT_ATLASES)
    if isinstance(atlases, str):
        atlases = [a.strip() for a in atlases.split(",") if a.strip()]
    return list(dict.fromkeys(a.lower() for a in atlases))


def _legacy(fn_name, legacy, kw) -> dict:
    """Map pre-0.5 positional flags (threshold, tracts, write, plots,
    verbose, frame) to keywords, with a DeprecationWarning."""
    if not legacy:
        return kw
    if len(legacy) > len(_LEGACY_POSITIONAL):
        raise TypeError(f"{fn_name}() takes at most {3 + len(_LEGACY_POSITIONAL)} positional "
                        "arguments")
    warnings.warn(f"Passing {', '.join(_LEGACY_POSITIONAL[:len(legacy)])} to {fn_name}() by "
                  "position is deprecated; pass them by keyword.", DeprecationWarning,
                  stacklevel=3)
    for k, v in zip(_LEGACY_POSITIONAL, legacy):
        kw[k] = v
    return kw


def _fmt_elapsed(sec: float) -> str:
    return f"{sec:.1f} s" if sec < 60 else f"{sec / 60:.1f} min"


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
            "frame='pandas' needs pandas: pip install 'lqtpy[tables]'") from exc
    return pd


def _frame(records, columns):
    """A DataFrame for ``frame="pandas"``."""
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


def _aal_to_aseg(records, value_col="PercentDamage") -> "tuple[dict, dict]":
    """Map AAL subcortical/cerebellar damage onto ggseg aseg regions:
    ``(values, labels)`` where labels are ``{aseg key: (name, highlight)}``."""
    if not records:
        return {}, {}
    out, labels, cereb = {}, {}, []
    for r in records:
        name, val = r["RegionName"], r[value_col]
        base = name.rsplit("_", 1)[0]
        hemi = "Left" if name.endswith("_L") else "Right" if name.endswith("_R") else None
        if base in _ASEG_BASE and hemi:
            key = f"{hemi}-{_ASEG_BASE[base]}"
            out[key] = float(val)
            labels[key] = (f"{hemi.lower()} {base.lower()} (AAL {name})", f"aal|{name}")
        elif name.startswith(("Cerebelum", "Vermis")):
            cereb.append(float(val))
    if cereb:
        out["Cerebellum-Cortex"] = max(cereb)
        labels["Cerebellum-Cortex"] = ("cerebellum (most damaged AAL cerebellar region)", None)
    return out, labels


def _display_names(atlas) -> dict:
    return {m.RegionName: m.display() for m in atlas.labels.values()}


def _flat_maps(atlas, records, value_col="PercentDamage", uid="fm"):
    """``[(caption, svg)]``: vector region maps for an atlas (themeable, with
    ``data-hl`` keys linking regions to table rows)."""
    from . import figures
    label = "Percent damaged" if value_col == "PercentDamage" else "Mean % damaged"
    disp = _display_names(atlas)
    if atlas.key == "dk":
        names = {g: (disp.get(r, r), f"dk|{r}") for r, g in atlas.ggseg_map.items()}
        svg = figures.flat_map_svg("dk", _ggseg_dict(atlas, records, value_col), labels=names,
                                   label=label, uid=uid)
        cap = "Cortical damage (DKT), percent of each region damaged"
    elif atlas.key == "schaefer":
        data = {r["RegionName"]: float(r[value_col]) for r in records} if records else {}
        names = {k: (v, f"schaefer|{k}") for k, v in disp.items()}
        svg = figures.flat_map_svg("schaefer100", data, labels=names, label=label, uid=uid)
        cap = "Cortical damage (Schaefer-100), percent of each parcel damaged"
    elif atlas.key == "aal":
        vals, labels = _aal_to_aseg(records, value_col)
        svg = figures.aseg_svg(vals, labels=labels, label=label, uid=uid)
        cap = "Subcortical and cerebellar damage (AAL, drawn on aseg geometry)"
    else:
        return []
    return [(cap, svg)] if svg else []


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


def _jsonable(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


# --------------------------------------------------------------------------- #
# One lesion, no file output
# --------------------------------------------------------------------------- #
def _analyse(p, atlases, *, kind, tracts, plots, qc, describe, profile, extras,
             disconnection, expected_side, threshold, native=None, viewer=True,
             thumbnail=False, anonymous=False) -> dict:
    from . import datasets as _datasets
    from . import qc as _qc
    from . import structures as _st
    from .mirror import homologue as _hom
    from .repro import checksum
    from .watershed import border_zones as _bz

    man = Manifest(kind, settings={"threshold": threshold, "atlases": atlases,
                                   "profile": profile if describe else None,
                                   "extras": extras or "none"})
    t0 = time.perf_counter()
    morph = compute_morphometry(p)
    man.add("morphometry", "Core morphometry", RAN)

    flags = []
    if qc:
        flags = list(_qc.check(p, expected_side=expected_side))
        if native is not None:
            from .native import DRAWING_CHECKS, native_qc
            flags = [f for f in flags if f.id not in DRAWING_CHECKS] + native_qc(
                native, threshold=threshold)
            flags.sort(key=lambda f: {"error": 0, "warn": 1, "info": 2}[f.severity])
            man.add("qc", "Mask quality checks", RAN,
                    reason="drawing checks ran on the native-space mask")
        else:
            man.add("qc", "Mask quality checks", RAN)
    else:
        man.add("qc", "Mask quality checks", SKIPPED, reason="turned off (qc=False)")
    from .qc_model import QC_VERSION, verdict_and_score
    qc_verdict, qc_score, _ = verdict_and_score(flags) if qc else (None, None, None)

    from . import tissue as _tissue
    loc = _tissue.locate(p)
    tissue_res = {"class": loc["class"], "composition": loc["composition"],
                  "counts": loc["counts"], "contact_mm": loc["contact_mm"],
                  "parts": loc["parts"]}
    man.add("tissue", "Tissue composition and location classes", RAN)

    bz = _bz(p)
    man.add("border_zones", "Arterial border zones (watershed)", RAN)
    key_structs, cross = [], None
    try:
        key_structs = _st.distances(p)
        man.add("key_structures", "Distance to key structures", RAN)
    except DataUnavailableError as exc:
        man.add("key_structures", "Distance to key structures", UNAVAILABLE,
                reason=str(exc).splitlines()[0])
    hom = _hom(p)
    man.add("homologue", "Mirror image (homologous region)", RAN)

    figs = {}
    if plots:
        from . import figures
        figs["glass_svg"] = figures.glass_brain_svg(p)
        figs["slices_svg"] = figures.slices_svg(p, morph["center_of_mass_mm"])
        if viewer:
            figs["viewer"] = figures.viewer_data(p)
        if thumbnail:
            figs["thumb_uri"] = figures.thumbnail_uri(p)

    atlas_results = []
    for i, key in enumerate(atlases):
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
            entry["flat_svgs"] = _flat_maps(atlas, df, uid=f"fm{i}")
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
        tract_res = {"df": _tracts.tract_overlap(p)}
        man.add("tract_proxy", "White-matter tract overlap (proxy)", RAN)
        cross = _st.cross_section_load(p)
        man.add("cross_section", "Tract load at cross-sections", RAN)

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
        d = _describe(p, profile=profile, qc_flags=flags, shape="shape" in measures)
        description = {"text": d.text, "sections": d.sections, "marked": d.marked,
                       "profile": d.profile, "facts": d.facts}
        man.add("description", "Plain-language description", RAN, profile=profile)
    else:
        man.add("description", "Plain-language description", SKIPPED,
                reason="turned off (describe=False)")

    src = p.source if isinstance(p.source, (str, os.PathLike)) else None
    man_d = man.to_dict()
    man_d["input"] = {"name": p.name,
                      "path": None if anonymous or src is None else os.path.abspath(src),
                      "sha256": checksum(p)}
    out = {
        "name": p.name,
        "source_path": None if anonymous else src,
        "was_binary": p.was_binary,
        "morphometry": morph,
        "qc": [f.to_dict() for f in flags],
        "qc_verdict": qc_verdict,
        "qc_score": qc_score,
        "qc_version": QC_VERSION if qc else None,
        "tissue": tissue_res,
        "border_zones": bz,
        "key_structures": key_structs,
        "cross_section": cross,
        "homologue": hom,
        "description": description,
        "measures": {k: {"values": v["values"], "descriptors": [d.text for d in v["descriptors"]]}
                     for k, v in measures.items()},
        "atlases": atlas_results,
        "tracts": tract_res,
        "disconnection": discon,
        "manifest": man_d,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "compute_seconds": round(time.perf_counter() - t0, 3),
    }
    out.update(figs)
    return out


def _write_single(result, out_dir, name, *, plots, report_style=None, back_link=None,
                  pdf=False, run=None):
    from .dictionary import write_dictionary
    os.makedirs(out_dir, exist_ok=True)
    written = {}
    with open(os.path.join(out_dir, f"{name}_morphometry.json"), "w", encoding="utf-8") as fh:
        json.dump(result["morphometry"], fh, indent=2)
    summary = {"name": name, "qc": result["qc"], "manifest": result["manifest"],
               "measures": result["measures"],
               "description": (result["description"] or {}).get("text"),
               "tissue": {k: v for k, v in (result.get("tissue") or {}).items() if k != "parts"},
               "border_zones": result.get("border_zones"), "homologue": result.get("homologue"),
               "coverage": {a["key"]: a["coverage"] for a in result["atlases"]}}
    with open(os.path.join(out_dir, f"{name}_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=_jsonable)

    def table(fname, rows, cols):
        _write_csv(os.path.join(out_dir, fname), rows, cols)
        written[fname] = cols

    if result.get("tissue") and result["tissue"]["parts"]:
        table(f"{name}_parts.csv", [_part_row(pt) for pt in result["tissue"]["parts"]], _PART_COLUMNS)
    for a in result["atlases"]:
        table(f"{name}_{a['key']}_damage.csv", a["df"], _DAMAGE_COLUMNS)
        if a.get("lobe_df"):
            table(f"{name}_{a['key']}_lobes.csv", a["lobe_df"], _LOBE_COLUMNS)
    if result["tracts"] is not None:
        table(f"{name}_tracts.csv", result["tracts"]["df"], _TRACT_COLUMNS)
    if result.get("cross_section"):
        table(f"{name}_tract_cross_section.csv", result["cross_section"], _CROSS_COLUMNS)
    if result.get("key_structures"):
        table(f"{name}_key_structures.csv", result["key_structures"], _KEY_COLUMNS)
    if result["disconnection"] is not None:
        table(f"{name}_disconnection.csv", result["disconnection"], _DISCON_COLUMNS)
    downloads = {f: os.path.join(out_dir, f) for f in written}
    written[f"{name}_morphometry.json"] = list(result["morphometry"].keys())
    write_dictionary(out_dir, written)
    downloads["data_dictionary.csv"] = os.path.join(out_dir, "data_dictionary.csv")
    if result["description"]:
        with open(os.path.join(out_dir, f"{name}_description.txt"), "w", encoding="utf-8") as fh:
            fh.write(result["description"]["text"] + "\n")
    if plots:
        from .report import render_single
        path = os.path.join(out_dir, f"{name}_report.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_single(result, report_style, back_link=back_link,
                                   run=run or (result.get("manifest") or {}).get("run"),
                                   downloads=downloads))
        result["report_path"] = os.path.normpath(path)
        if pdf:
            from .report import to_pdf
            result["pdf_path"] = os.path.normpath(to_pdf(path))


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
def analyze_single(lesion=None, atlases=None, out_dir=None, *legacy, lesion_path=None,
                   threshold=0.5, tracts=True, write=True, plots=False, verbose=False,
                   frame=None, qc=True, describe=True, profile="stroke", extras=None,
                   disconnection="auto", expected_side=None, report_style=None, pdf=False,
                   native=None, transform=None, anonymise=False, viewer=True):
    """Analyse one lesion; return a result dict and (optionally) write CSV/JSON
    tables (``write=True``) and a self-contained HTML report
    (``write=True`` and ``plots=True``).

    ``lesion`` accepts anything :func:`lqtpy.prepare` does (``lesion_path=``
    is the old name). Core modules always run unless turned off (``qc``,
    ``describe``, ``tracts``, ``disconnection`` -- the latter runs
    automatically when an index is installed). ``extras`` selects extra
    measures: ``None`` (core only, the default everywhere), ``"all"``, or a
    list of names from :func:`lqtpy.registry.list_measures`. ``profile``
    picks the description style (stroke / small_vessel / tumour / research).
    ``result["manifest"]`` records what ran and what didn't, and the input's
    checksum (``manifest["run"]`` is the full run record).

    Native space (:mod:`lqtpy.native`): ``native=`` a native-space mask
    runs the drawing checks there; ``transform=`` treats ``lesion`` as a
    native mask, warps it to MNI with that transform and does the same.

    ``anonymise=True`` (or a key-file path) writes an ID instead of the
    file name everywhere (:mod:`lqtpy.anonymise`). ``pdf=True`` (with
    ``write`` and ``plots``) also saves the report as PDF. ``viewer=False``
    leaves out the scroll-through slice viewer (smaller file).

    Tables are records by default; ``frame="pandas"`` returns DataFrames.
    Arguments after ``out_dir`` are keyword-only (positional use is
    deprecated).
    """
    kw = _legacy("analyze_single", legacy, dict(threshold=threshold, tracts=tracts, write=write,
                                                plots=plots, verbose=verbose, frame=frame))
    threshold, tracts, write, plots, verbose, frame = (kw[k] for k in _LEGACY_POSITIONAL)
    if lesion is None:
        lesion = lesion_path
    if lesion is None:
        raise TypeError("analyze_single() needs a lesion")
    t0 = time.time()
    atlases = _atlas_keys(atlases)
    if transform is not None:
        from .native import warp_to_mni
        native = lesion
        p = prepare(warp_to_mni(lesion, transform, threshold=threshold), 0.5, name=_stem(lesion))
    else:
        p = prepare(lesion, threshold)
    anon = None
    if anonymise:
        from .anonymise import Anonymiser
        anon = Anonymiser.from_option(anonymise, out_dir or "lqtpy_output")
        # Renamed in place: the source file (and so its checksum) stays.
        p.name = anon.id_for(p.name, p.source)
    out_dir = out_dir or os.path.join("lqtpy_output", p.name)
    if verbose:
        from . import _term
        _term.start_single(p.name)
    result = _analyse(p, atlases, kind="single", tracts=tracts, plots=plots, qc=qc,
                      describe=describe, profile=profile, extras=extras,
                      disconnection=disconnection, expected_side=expected_side,
                      threshold=threshold, native=native, viewer=viewer, thumbnail=plots,
                      anonymous=anon is not None)
    result["elapsed"] = _fmt_elapsed(time.time() - t0)
    from .repro import run_record
    result["manifest"]["run"] = run_record(
        "single", [result["manifest"]["input"]],
        dict(atlases=atlases, threshold=threshold, tracts=tracts, qc=qc, describe=describe,
             profile=profile, extras=extras, disconnection=disconnection,
             expected_side=expected_side, anonymise=bool(anonymise)))
    if write:
        _write_single(result, out_dir, p.name, plots=plots, report_style=report_style, pdf=pdf)
        if anon is not None:
            result["anonymisation_key"] = anon.save()
    if verbose:
        from . import _term
        _term.done_single(result)
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
    from .formats import is_lesion_file
    if isinstance(inputs, (str, os.PathLike)):
        path = os.fspath(inputs)
        if os.path.isdir(path):
            inputs = sorted(os.path.join(path, f) for f in os.listdir(path) if is_lesion_file(f))
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
    suffix = "Overlap" if "Overlap" in value else "Damage"
    for k in sorted(grouped):
        g = grouped[k]
        row = {key: k}
        if extra_first:
            row[extra_first] = g[extra_first]
        row.update({
            "NSubjects": g["n"],
            "NSubjectsTotal": n_total,
            f"CohortMeanPercent{suffix}": round(g["sum"] / n_total, 2) if n_total else 0.0,
            f"MeanPercent{suffix}Affected": round(g["sum"] / g["n"], 2),
            f"MaxPercent{suffix}": round(g["max"], 2),
        })
        rows.append(row)
    rows.sort(key=lambda r: (-r["NSubjects"], -r[f"CohortMeanPercent{suffix}"]))
    return rows


def _worst(flags):
    order = {"error": 0, "warn": 1, "info": 2}
    return min((f["severity"] for f in flags), key=lambda s: order[s], default=None)


def _settings_key(opts: dict, expected_side=None) -> str:
    keep = {k: opts[k] for k in ("atlases", "threshold", "tracts", "qc", "describe", "profile",
                                 "extras", "disconnection", "plots", "per_report", "thumbnail",
                                 "viewer", "anonymous")}
    keep["expected_side"] = expected_side
    from ._version import __version__
    keep["version"] = __version__
    return hashlib.sha1(json.dumps(keep, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _subject_job(job: dict) -> dict:
    """Analyse one group member. Top-level (picklable) so it can run in a
    worker process. Returns what the group needs, never the heavy figures."""
    from .repro import checksum
    o = job["opts"]
    name, src = job["name"], job["src"]
    p = prepare(src, o["threshold"], name=name)
    sha = checksum(p)
    key = f"{sha}:{_settings_key(o, job.get('expected_side'))}"
    cache_file = None
    if job.get("cache_dir"):
        cache_file = os.path.join(job["cache_dir"], f"{name}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, encoding="utf-8") as fh:
                    cached = json.load(fh)
                if cached.get("key") == key:
                    out = cached["payload"]
                    out["idx"] = np.asarray(out["idx"], np.int64)
                    out["cached"] = True
                    return out
            except (OSError, ValueError, KeyError):
                pass
    ts = time.time()
    res = _analyse(p, o["atlases"], kind="group-member", tracts=o["tracts"], plots=o["per_report"],
                   qc=o["qc"], describe=o["describe"], profile=o["profile"], extras=o["extras"],
                   disconnection=o["disconnection"], expected_side=job.get("expected_side"),
                   threshold=o["threshold"], viewer=o["viewer"], thumbnail=o["thumbnail"],
                   anonymous=o["anonymous"])
    res["elapsed"] = _fmt_elapsed(time.time() - ts)
    idx = np.flatnonzero(p.on_grid((tuple(job["ref_shape"]),
                                    np.asarray(job["ref_affine"]))).mask.ravel())
    report_path = None
    if o["write"]:
        sub_dir = os.path.join(o["out_dir"], "subjects", name)
        _write_single(res, sub_dir, name, plots=o["per_report"], report_style=o["report_style"],
                      back_link="../../group_report.html", pdf=o["subject_pdf"])
        if "report_path" in res:
            report_path = os.path.relpath(res["report_path"], o["out_dir"]).replace("\\", "/")
    facts = (res["description"] or {}).get("facts") or {}
    terrs = facts.get("territories") or []
    desc = res["description"] or {}
    summary_para = dict(desc.get("sections") or []).get("Summary", "")
    marked = dict(desc.get("marked") or []).get("Summary", "")
    tis = res.get("tissue") or {}
    ref_df = res["atlases"][0]["df"] if res["atlases"] else []
    out = {
        "entry": {"name": name, "morphometry": res["morphometry"],
                  "tissue": {k: v for k, v in tis.items() if k != "parts"},
                  "top_region": ref_df[0]["RegionName"] if ref_df else "-",
                  "territory": (_fmt.territory_short(terrs[0]["hemisphere"], terrs[0]["territory"])
                                if terrs else None),
                  "border_zone": bool((res.get("border_zones") or {}).get("fits")),
                  "qc": res["qc"], "qc_worst": _worst(res["qc"]),
                  "qc_verdict": res.get("qc_verdict"), "qc_score": res.get("qc_score"),
                  "summary": summary_para, "summary_marked": marked or summary_para,
                  "report_path": report_path,
                  "pdf_path": res.get("pdf_path"),
                  "thumb": res.get("thumb_uri"), "sha256": sha,
                  "source": None if o["anonymous"] else (
                      os.path.abspath(src) if isinstance(src, (str, os.PathLike)) else None),
                  "elapsed": res["elapsed"],
                  "measures": res.get("measures") or {}},
        "atlases": {a["key"]: {"df": a["df"], "lobe_df": a.get("lobe_df"),
                               "network_df": a.get("network_df")} for a in res["atlases"]},
        "tracts": (res["tracts"] or {}).get("df"),
        "manifest": res["manifest"],
        "idx": idx,
    }
    if cache_file:
        os.makedirs(job["cache_dir"], exist_ok=True)
        payload = dict(out)
        payload["idx"] = idx.tolist()
        with open(cache_file, "w", encoding="utf-8") as fh:
            json.dump({"key": key, "payload": payload}, fh, default=_jsonable)
    out["cached"] = False
    return out


def _run_jobs(jobs, n_jobs, on_done):
    """Run subject jobs in order (``n_jobs=1``) or in a process pool;
    ``on_done(i, result)`` is called as each finishes."""
    results = [None] * len(jobs)
    if n_jobs and n_jobs != 1 and len(jobs) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        workers = (os.cpu_count() or 2) if n_jobs < 1 else n_jobs
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs))) as ex:
            futs = {ex.submit(_subject_job, j): i for i, j in enumerate(jobs)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                on_done(i, results[i])
    else:
        for i, j in enumerate(jobs):
            results[i] = _subject_job(j)
            on_done(i, results[i])
    return results


def analyze_group(inputs=None, atlases=None, out_dir=None, *legacy, threshold=0.5, tracts=True,
                  write=True, plots=False, verbose=False, frame=None, qc=True, describe=True,
                  profile="stroke", extras=None, disconnection="auto", subject_reports=True,
                  report_style=None, pdf=False, metadata=None, subject_column=None,
                  group_by=None, compare=None, side_column="auto", n_jobs=1, anonymise=False,
                  incremental=False, duplicates=True, outliers=True, viewer=True):
    """Analyse a group of lesions; return a result dict and (optionally) write
    group CSVs, a group HTML report (``plots=True``) and -- with
    ``subject_reports=True`` -- one full report per subject, linked from the
    group report.

    ``inputs``: a directory, a single lesion file, or a list of lesion
    sources or ``(name, source)`` pairs. Group tables report, per region,
    how many subjects are affected, the **cohort** mean (unaffected
    subjects count as 0), the mean over affected subjects, and the max.

    Cohort options:

    ``metadata``      CSV/TSV path, list of dicts, dict or DataFrame, one row
                      per subject (:mod:`lqtpy.metadata`); joined into the
                      subject table and CSV. ``subject_column`` names its
                      subject column.
    ``group_by``      a metadata column to summarise the cohort by.
    ``compare``       a metadata column with two values (or ``(column, a,
                      b)``): region-by-region comparison of the two groups
                      (:func:`lqtpy.cohort.compare`) with side-by-side
                      overlap maps.
    ``side_column``   metadata column giving the lesion side (or the symptom
                      side); mismatches get a ``side_mismatch`` flag.
                      ``"auto"`` finds one by name, ``None`` turns it off.
    ``duplicates``    flag identical / near-identical masks.
    ``outliers``      flag subjects far from the group (10+ subjects).
    ``n_jobs``        worker processes (``-1``: one per CPU). On Windows,
                      call from under ``if __name__ == "__main__":``.
    ``incremental``   reuse cached per-subject results from a previous run
                      into the same ``out_dir`` when the file and settings
                      are unchanged.
    ``anonymise``     IDs instead of names everywhere; ``True`` or a key-file
                      path (:mod:`lqtpy.anonymise`).
    ``pdf``           ``True``: the group report as PDF; ``"bundle"``: also
                      every subject's PDF and, with ``pypdf`` installed, one
                      merged PDF with bookmarks.

    Arguments after ``out_dir`` are keyword-only (positional use is
    deprecated).
    """
    from . import cohort
    from .metadata import expected_sides, load_metadata
    kw = _legacy("analyze_group", legacy, dict(threshold=threshold, tracts=tracts, write=write,
                                               plots=plots, verbose=verbose, frame=frame))
    threshold, tracts, write, plots, verbose, frame = (kw[k] for k in _LEGACY_POSITIONAL)
    t0 = time.time()
    atlases = _atlas_keys(atlases)
    pairs = _normalise_inputs(inputs)
    if len(pairs) < 1:
        raise InvalidLesionError("No lesion files found for group analysis.")
    n = len(pairs)
    out_dir = out_dir or os.path.join("lqtpy_output", "group")

    # Metadata is matched on the original names, before anonymising.
    meta = load_metadata(metadata, [nm for nm, _ in pairs], subject_column=subject_column)
    sides, side_col, side_from_symptoms = expected_sides(meta, side_column)
    anon = None
    if anonymise:
        from .anonymise import Anonymiser
        anon = Anonymiser.from_option(anonymise, out_dir)
        renamed = []
        for nm, src in pairs:
            new = anon.id_for(nm, src)
            if nm in meta["by_subject"]:
                meta["by_subject"][new] = meta["by_subject"].pop(nm)
            if nm in sides:
                sides[new] = sides.pop(nm)
            renamed.append((new, src))
        pairs = renamed
    cmp_col = compare[0] if isinstance(compare, (tuple, list)) else compare
    for col, what in ((group_by, "group_by"), (cmp_col, "compare")):
        if col and col not in meta["columns"]:
            raise ValueError(f"{what}={col!r} isn't a metadata column "
                             f"(columns: {meta['columns'] or 'no metadata given'})")
    cmp_a = cmp_b = None
    if compare:
        if isinstance(compare, (tuple, list)):
            cmp_a, cmp_b = str(compare[1]), str(compare[2])
        else:
            vals = sorted({str(v.get(cmp_col, "")) for v in meta["by_subject"].values()
                           if str(v.get(cmp_col, "")).strip() not in ("", "nan", "NA")})
            if len(vals) != 2:
                raise ValueError(f"compare={cmp_col!r} has {len(vals)} distinct values {vals[:6]}; "
                                 "pass (column, value_a, value_b)")
            cmp_a, cmp_b = vals

    ref_img = load_atlas("tissue").img
    per_report = bool(write and plots and subject_reports)
    opts = {"atlases": atlases, "threshold": threshold, "tracts": tracts, "qc": qc,
            "describe": describe, "profile": profile, "extras": extras,
            "disconnection": disconnection, "plots": plots,
            "per_report": per_report, "thumbnail": bool(plots), "viewer": viewer,
            "write": write, "out_dir": out_dir, "report_style": report_style,
            "subject_pdf": per_report and pdf == "bundle", "anonymous": anon is not None}
    cache_dir = os.path.join(out_dir, ".lqtpy_cache") if (incremental and write) else None
    jobs = [{"name": nm, "src": src, "opts": opts, "expected_side": sides.get(nm),
             "ref_shape": list(ref_img.shape[:3]), "ref_affine": np.asarray(ref_img.affine).tolist(),
             "cache_dir": cache_dir} for nm, src in pairs]
    if verbose:
        from . import _term
        _term.start_group(n, n_jobs)

    def on_done(i, r):
        if verbose:
            from . import _term
            _term.subject_line(i + 1, n, r["entry"], cached=r.get("cached"))

    results = _run_jobs(jobs, n_jobs, on_done)

    freq = np.zeros(int(np.prod(ref_img.shape[:3])), np.int32)
    freq_by: "dict[str, np.ndarray]" = {}
    subjects, sigs = [], []
    per_atlas = {k: [] for k in atlases}
    per_lobe = {k: [] for k in atlases}
    per_net = {k: [] for k in atlases}
    tract_frames = []
    qc_counts: "dict[str, int]" = {}
    tissue_counts: "dict[str, int]" = {}
    subject_manifest = None
    n_cached = 0
    for (nm, _), r in zip(pairs, results):
        e = r["entry"]
        e["meta"] = meta["by_subject"].get(nm, {})
        subject_manifest = r["manifest"]
        n_cached += bool(r.get("cached"))
        np.add.at(freq, r["idx"], 1)
        if cmp_col:
            g = str(e["meta"].get(cmp_col, ""))
            if g in (cmp_a, cmp_b):
                fb = freq_by.setdefault(g, np.zeros_like(freq))
                np.add.at(fb, r["idx"], 1)
        if duplicates:
            sigs.append(cohort.signature(nm, r["idx"], ref_img.shape[:3], np.asarray(ref_img.affine)))
        for k in atlases:
            a = r["atlases"].get(k) or {}
            per_atlas[k].append(a.get("df"))
            per_lobe[k].append(a.get("lobe_df"))
            per_net[k].append(a.get("network_df"))
        tract_frames.append(r["tracts"])
        for f in e["qc"]:
            if f["severity"] != "info":
                qc_counts[f["id"]] = qc_counts.get(f["id"], 0) + 1
        tis = e.get("tissue") or {}
        tissue_counts = {k: tissue_counts.get(k, 0) + v for k, v in (tis.get("counts") or {}).items()}
        e["group_flags"] = []
        subjects.append(e)

    # ---- cohort checks ---------------------------------------------------- #
    dup_pairs = cohort.find_duplicates(sigs) if duplicates else []
    by_name = {s["name"]: s for s in subjects}
    for d in dup_pairs:
        for x, y in ((d["a"], d["b"]), (d["b"], d["a"])):
            word = ("identical to" if d["kind"] == "identical"
                    else f"nearly identical (Dice {d['dice']:.2f}) to")
            by_name[x]["group_flags"].append({"id": "duplicate", "severity": "warn",
                                              "message": f"The mask is {word} {y}'s.",
                                              "evidence": d,
                                              "suggestion": "Check for a duplicated file."})
    outl = cohort.find_outliers(subjects) if outliers else {}
    for nm, items in outl.items():
        for it in items:
            by_name[nm]["group_flags"].append({"id": f"outlier_{it['feature']}", "severity": "warn",
                                               "message": it["message"], "evidence": it,
                                               "suggestion": ""})
    for s in subjects:
        s["qc_all"] = s["qc"] + s["group_flags"]
        s["qc_worst"] = _worst(s["qc_all"])
        for f in s["group_flags"]:
            qc_counts[f["id"]] = qc_counts.get(f["id"], 0) + 1

    # ---- figures ------------------------------------------------------------ #
    freq_img = nib.Nifti1Image(freq.reshape(ref_img.shape[:3]), ref_img.affine)
    overlap_svgs = None
    compare_svgs = {}
    if plots:
        from . import figures
        overlap_svgs = figures.overlap_svg(freq_img, n)
        for g, fb in freq_by.items():
            ng = sum(1 for s in subjects if str(s["meta"].get(cmp_col, "")) == g)
            compare_svgs[g] = figures.overlap_svg(
                nib.Nifti1Image(fb.reshape(ref_img.shape[:3]), ref_img.affine), ng, n_cuts=4)

    atlas_results = []
    for i, key in enumerate(atlases):
        atlas = load_atlas(key)
        agg = _aggregate(per_atlas[key], "RegionName", "PercentDamage", n, extra_first="Group")
        entry = {"key": key, "name": atlas.name, "tier": atlas.tier, "df": agg}
        if any(per_lobe[key]):
            entry["lobe_df"] = _aggregate(per_lobe[key], "Lobe", "PercentDamage", n)
        if any(x is not None for x in per_net[key]):
            entry["network_df"] = _aggregate(per_net[key], "Network", "PercentDamage", n)
        if plots:
            entry["flat_svgs"] = _flat_maps(atlas, agg, "CohortMeanPercentDamage", uid=f"gfm{i}")
        atlas_results.append(entry)

    tract_agg = (_aggregate(tract_frames, "Tract", "PercentOverlap", n)
                 if any(t is not None for t in tract_frames) else None)

    # Per-subject rollups: the report's filters recompute the cohort bars.
    for i, s in enumerate(subjects):
        vals = {}
        for key, src_list, col in (("dk", per_lobe, "Lobe"), ("arterial", per_net, "Network"),
                                   ("schaefer", per_net, "Network")):
            if key in per_atlas:
                vals[key] = {r[col]: r["PercentDamage"] for r in (src_list[key][i] or [])
                             if r["PercentDamage"] > 0}
        s["rollups"] = vals

    # ---- groups and comparison -------------------------------------------- #
    groups = None
    if group_by:
        groups = []
        for g in sorted({str(s["meta"].get(group_by, "")) for s in subjects}):
            mem = [s for s in subjects if str(s["meta"].get(group_by, "")) == g]
            terr = _count_by(s.get("territory") for s in mem)
            groups.append({"value": g or "(missing)", "n": len(mem),
                           "median_volume_mm3": float(np.median([s["morphometry"]["volume_mm3"]
                                                                 for s in mem])),
                           "n_left": sum(1 for s in mem if s["morphometry"]["pct_left"] >= 50),
                           "n_bilateral": sum(1 for s in mem if s["morphometry"].get("bilateral")),
                           "top_territory": next(iter(terr), None),
                           "n_flagged": sum(1 for s in mem if s["qc_worst"] in ("warn", "error"))})
    comparison = None
    if cmp_col:
        labels = [str(s["meta"].get(cmp_col, "")) for s in subjects]
        tables = []
        for key in atlases:
            nm_ = load_atlas(key).name
            rows = cohort.compare(per_atlas[key], labels, cmp_a, cmp_b, "RegionName", "PercentDamage")
            if rows:
                tables.append({"key": f"{key}_regions", "label": f"Regions: {nm_}", "rows": rows})
            if any(per_lobe[key]) and key == "dk":
                rows = cohort.compare(per_lobe[key], labels, cmp_a, cmp_b, "Lobe", "PercentDamage")
                if rows:
                    tables.append({"key": f"{key}_lobes", "label": "Lobes (DKT)", "rows": rows})
            if key in ("arterial", "schaefer") and any(x is not None for x in per_net[key]):
                rows = cohort.compare(per_net[key], labels, cmp_a, cmp_b, "Network", "PercentDamage")
                if rows:
                    tables.append({"key": f"{key}_networks",
                                   "label": "Vascular territories" if key == "arterial"
                                   else "Yeo-7 networks", "rows": rows})
        if any(t is not None for t in tract_frames):
            rows = cohort.compare(tract_frames, labels, cmp_a, cmp_b, "Tract", "PercentOverlap")
            if rows:
                tables.append({"key": "tracts", "label": "White-matter tracts (voxel overlap)",
                               "rows": rows})
        comparison = {"column": cmp_col, "a": cmp_a, "b": cmp_b,
                      "n_a": labels.count(cmp_a), "n_b": labels.count(cmp_b),
                      "tables": tables, "overlap_svgs": compare_svgs}

    # ---- manifest ------------------------------------------------------------ #
    man = Manifest("group", settings={"threshold": threshold, "atlases": atlases,
                                      "profile": profile if describe else None,
                                      "extras": extras or "none", "n_subjects": n})
    for m in (subject_manifest or {}).get("modules", []):
        man.add(m["id"], m["label"], m["status"], tier=m["tier"], reason=m.get("reason"))
    man.add("subject_reports", "One report per subject",
            RAN if per_report else SKIPPED,
            reason=None if per_report else ("needs write=True and plots=True" if not (write and plots)
                                            else "turned off (subject_reports=False)"))
    man.add("metadata", "Subject metadata", RAN if meta["columns"] else SKIPPED,
            reason=(f"{len(meta['by_subject'])} of {n} subjects matched"
                    + (f"; {len(meta['unmatched_rows'])} rows unmatched" if meta["unmatched_rows"] else "")
                    ) if meta["columns"] else "no metadata given")
    man.add("side_check", "Left/right flip check", RAN if sides else SKIPPED,
            reason=(f"from '{side_col}'" + (" (symptom side: lesion expected on the opposite side)"
                                            if side_from_symptoms else "")) if sides
            else "no side column in the metadata")
    man.add("duplicates", "Duplicate masks", RAN if duplicates else SKIPPED,
            reason=None if duplicates else "turned off")
    man.add("outliers", "Cohort outliers", RAN if (outliers and n >= 10) else SKIPPED,
            reason=None if (outliers and n >= 10) else ("needs 10+ subjects" if outliers else "turned off"))
    man.add("comparison", "Two-group comparison", RAN if comparison else SKIPPED,
            reason=(f"{cmp_col}: {cmp_a} vs {cmp_b}") if comparison else "no compare= column")
    if n_jobs and n_jobs != 1:
        man.add("parallel", "Parallel workers", RAN, reason=f"n_jobs={n_jobs}")
    if incremental:
        man.add("incremental", "Incremental run", RAN, reason=f"{n_cached} of {n} subjects reused")
    if anon is not None:
        man.add("anonymise", "Anonymised output", RAN, reason="names replaced by IDs")
    man_d = man.to_dict()
    from .repro import run_record
    man_d["run"] = run_record(
        "group", [{"name": s["name"], "path": s.get("source"), "sha256": s["sha256"]} for s in subjects],
        dict(atlases=atlases, threshold=threshold, tracts=tracts, qc=qc, describe=describe,
             profile=profile, extras=extras, disconnection=disconnection,
             subject_reports=subject_reports,
             metadata=(os.path.abspath(metadata) if isinstance(metadata, (str, os.PathLike))
                       else (None if metadata is None else f"<{type(metadata).__name__}>")),
             subject_column=subject_column, group_by=group_by,
             compare=list(compare) if isinstance(compare, (tuple, list)) else compare,
             side_column=side_column, anonymise=bool(anonymise)))

    vols = sorted(s["morphometry"]["volume_mm3"] for s in subjects)
    lis = [s["morphometry"]["laterality_index"] for s in subjects]
    result = {
        "n_subjects": n,
        "subjects": subjects,
        "median_volume_mm3": float(np.median(vols)),
        "median_volume_cc": round(float(np.median(vols)) / 1000.0, 2),
        "median_laterality_index": round(float(np.median(lis)), 3),
        "n_bilateral": sum(1 for s in subjects if s["morphometry"].get("bilateral")),
        "max_overlap": int(freq.max()),
        "qc_counts": qc_counts,
        "tissue_classes": _count_by(s["tissue"].get("class") for s in subjects),
        "tissue_part_counts": tissue_counts,
        "n_subjects_with_warnings": sum(1 for s in subjects if s["qc_worst"] in ("warn", "error")),
        "duplicates": dup_pairs,
        "outliers": outl,
        "metadata_columns": meta["columns"],
        "metadata_unmatched": meta["unmatched_rows"],
        "group_by": group_by,
        "groups": groups,
        "comparison": comparison,
        "overlap_svgs": overlap_svgs,
        "atlases": atlas_results,
        "tracts": None if tract_agg is None else {"df": tract_agg},
        "manifest": man_d,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed": _fmt_elapsed(time.time() - t0),
    }

    if write:
        _write_group(result, out_dir, plots=plots, pdf=pdf, report_style=report_style,
                     freq_img=freq_img, meta=meta, tract_agg=tract_agg)
        if anon is not None:
            result["anonymisation_key"] = anon.save()
    if verbose:
        from . import _term
        _term.done_group(result)

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


def _write_group(result, out_dir, *, plots, pdf, report_style, freq_img, meta, tract_agg):
    from .dictionary import write_dictionary
    os.makedirs(out_dir, exist_ok=True)
    written = {}

    def table(fname, rows, cols):
        _write_csv(os.path.join(out_dir, fname), rows, cols)
        written[fname] = cols

    subjects = result["subjects"]
    nib.save(freq_img, os.path.join(out_dir, "lesion_overlap_map.nii.gz"))
    for a in result["atlases"]:
        table(f"group_{a['key']}_damage.csv", a["df"], _GROUP_DAMAGE_COLUMNS)
        if a.get("lobe_df"):
            table(f"group_{a['key']}_lobes.csv", a["lobe_df"], _GROUP_LOBE_COLUMNS)
        if a.get("network_df"):
            table(f"group_{a['key']}_networks.csv", a["network_df"], _GROUP_NETWORK_COLUMNS)
    if tract_agg:
        table("group_tracts.csv", tract_agg, _GROUP_TRACT_COLUMNS)
    morph_keys = list(subjects[0]["morphometry"].keys())
    mcols = meta["columns"]
    rows = [{"Subject": s["name"], "QC": _fmt.quality_label(s["qc_worst"]),
             "QCFlags": ";".join(f["id"] for f in s["qc"] if f["severity"] != "info"),
             "GroupFlags": ";".join(f["id"] for f in s["group_flags"]),
             **{c: s["meta"].get(c, "") for c in mcols},
             **_tissue_cols(s["tissue"]), **s["morphometry"]} for s in subjects]
    table("group_subject_summary.csv", rows,
          ["Subject", "QC", "QCFlags", "GroupFlags"] + mcols + _TISSUE_COLUMNS + morph_keys)
    if result["duplicates"]:
        table("group_duplicates.csv", [{"SubjectA": d["a"], "SubjectB": d["b"], "Dice": d["dice"],
                                        "Kind": d["kind"]} for d in result["duplicates"]],
              ["SubjectA", "SubjectB", "Dice", "Kind"])
    cmp_ = result.get("comparison")
    if cmp_:
        for t in cmp_["tables"]:
            table(f"group_comparison_{t['key']}.csv", t["rows"], _COMPARE_COLUMNS)
    extra = {c: ("", "Metadata column (from the metadata file).") for c in mcols}
    extra.update({"SubjectA": ("", "First subject of the pair."),
                  "SubjectB": ("", "Second subject of the pair."),
                  "Dice": ("", "Dice overlap of the two masks (2 mm grid)."),
                  "Kind": ("", "identical or near-identical.")})
    write_dictionary(out_dir, written, extra=extra)
    downloads = {f: os.path.join(out_dir, f) for f in written}
    downloads["data_dictionary.csv"] = os.path.join(out_dir, "data_dictionary.csv")
    if plots:
        from .report import render_group
        path = os.path.join(out_dir, "group_report.html")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(render_group(result, report_style, run=result["manifest"].get("run"),
                                  downloads=downloads))
        result["report_path"] = os.path.normpath(path)
        if pdf:
            from .report import to_pdf
            result["pdf_path"] = os.path.normpath(to_pdf(path))
            if pdf == "bundle":
                result["bundle_path"] = _merge_pdfs(out_dir, result)


def _merge_pdfs(out_dir, result) -> "str | None":
    """One PDF with the group report then every subject, bookmarked
    (needs ``pypdf``; returns None without it)."""
    try:
        from pypdf import PdfWriter
    except ImportError:
        warnings.warn("pdf='bundle': install pypdf to merge the PDFs into one file; "
                      "the separate PDFs were written.")
        return None
    w = PdfWriter()
    parts = [("Group report", result.get("pdf_path"))]
    parts += [(s["name"], s.get("pdf_path")) for s in result["subjects"]]
    for title, path in parts:
        if path and os.path.exists(path):
            start = len(w.pages)
            w.append(path)
            w.add_outline_item(title, start)
    out = os.path.join(out_dir, "group_bundle.pdf")
    with open(out, "wb") as fh:
        w.write(fh)
    return os.path.normpath(out)
