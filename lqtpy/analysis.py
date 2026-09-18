"""High-level orchestration: single-lesion and group analysis."""
from __future__ import annotations

import csv
import json
import os
import time
import datetime

import numpy as np
import nibabel as nib

from .atlases import DEFAULT_ATLASES, load_atlas
from .errors import InvalidLesionError, MissingDependencyError
from .io import load_lesion, resample_mask_to
from .morphometry import compute_morphometry
from .overlap import (region_overlap, region_overlap_from_mask,
                      network_rollup, homolog_symmetry, lobe_rollup,
                      atlas_has_lobes,
                      _COLUMNS as _DAMAGE_COLUMNS,
                      _NETWORK_COLUMNS, _HOMOLOG_COLUMNS, _LOBE_COLUMNS)
from .tracts import _COLUMNS as _TRACT_COLUMNS
from . import tracts as _tracts

# `plotting` (matplotlib) and `report` are imported lazily inside the
# functions that use them, so `import lqtpy` doesn't pull matplotlib in.
# Region/tract tables are **records** (list[dict]) everywhere in this module
# -- pandas is never imported for the numeric path. The only place this
# module still touches pandas is building a throwaway DataFrame right before
# handing data to `plotting.py`'s bar-chart helpers or `report.py`'s HTML
# renderer, both of which predate this refactor and already require
# matplotlib/pandas whenever a report is generated (unchanged, out of scope
# here) -- see `_frame`, `_for_report_single`, `_for_report_group` below.

_GROUP_DAMAGE_COLUMNS = ["RegionName", "Group", "NSubjects",
                         "MeanPercentDamage", "MaxPercentDamage"]
_GROUP_NETWORK_COLUMNS = ["Network", "NSubjects",
                          "MeanPercentDamage", "MaxPercentDamage"]
_GROUP_TRACT_COLUMNS = ["Tract", "NSubjects",
                        "MeanPercentOverlap", "MaxPercentOverlap"]


def _stem(path) -> str:
    """Derive an output-file stem from a lesion source. Non-path sources
    (a nibabel image, or an (array, affine) tuple -- see as_lesion_image)
    have no filename to draw from, so they fall back to a fixed name."""
    if not isinstance(path, (str, os.PathLike)):
        return "lesion"
    b = os.path.basename(os.fspath(path))
    for ext in (".nii.gz", ".nii"):
        if b.endswith(ext):
            return b[: -len(ext)]
    return os.path.splitext(b)[0]


def _fmt_elapsed(sec: float) -> str:
    return f"{sec:.1f}s" if sec < 60 else f"{sec/60:.1f} min"


def _write_csv(path, records, columns):
    """Write a list of dict records to CSV via the stdlib ``csv`` module."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)


def _pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependencyError(
            "tables",
            "This requires pandas. Install the 'tables' extra for a "
            "frame='pandas' conversion, or the 'report'/'ggseg'/'plots' "
            "extras to generate reports/plots: pip install 'lqtpy[tables]'"
        ) from exc
    return pd


def _frame(records, columns):
    """Build a pandas DataFrame from records, purely for feeding into the
    (unmodified) plotting.py/report.py helpers, which still expect
    DataFrame-style column access. Returns None for None input."""
    if records is None:
        return None
    return _pandas().DataFrame(records, columns=columns)


def _ggseg_dict(atlas, records, value_col="PercentDamage") -> dict:
    """Map an atlas damage table to {ggseg_region: value} for dk_flat_png."""
    if not atlas.ggseg or not records:
        return {}
    return {
        atlas.ggseg_map[r["RegionName"]]: float(r[value_col])
        for r in records if r["RegionName"] in atlas.ggseg_map
    }


# AAL subcortical/cerebellar regions -> ggseg aseg region keys.
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
    """Return a list of (caption, data_uri) ggseg-style flat maps for an atlas."""
    from . import plotting
    from .ggseg_flat import dk_flat_png, schaefer_flat_png
    maps = []
    if atlas.key == "dk":
        maps.append(("", dk_flat_png(_ggseg_dict(atlas, records, value_col))))
    elif atlas.key == "schaefer":
        data = ({r["RegionName"]: float(r[value_col]) for r in records}
                if records else {})
        maps.append(("", schaefer_flat_png(data)))
    elif atlas.key == "aal":
        maps.append(("Subcortical / cerebellar overlap (aseg surrogate geometry)",
                     plotting.ggseg_aseg_png(_aal_to_aseg(records, value_col))))
    return [(cap, uri) for cap, uri in maps if uri]


def _add_extras(entry, key, atlas, records, plots=False):
    """Attach per-atlas extras: Yeo-7 network rollup (Schaefer), L/R homolog
    symmetry table (AAL), lobe rollup (any atlas with a lobe mapping --
    currently aal/dk). The rollup/symmetry tables are plain records, always
    computed; the network bar chart is matplotlib-backed and only built
    when ``plots`` is true."""
    if key == "schaefer":
        net = network_rollup(records, atlas)
        entry["network_df"] = net
        if plots:
            from . import plotting
            entry["network_bar_uri"] = plotting.network_bar_png(_frame(net, _NETWORK_COLUMNS))
    elif key == "aal":
        entry["homolog_df"] = homolog_symmetry(records)
    if atlas_has_lobes(atlas):
        entry["lobe_df"] = lobe_rollup(records, atlas)


def _group_network(schaefer_frames, atlas):
    """Aggregate per-subject Schaefer overlap into per-network group stats.

    Records with keys Network, NSubjects, MeanPercentDamage,
    MaxPercentDamage, sorted by MeanPercentDamage descending.
    """
    nets = [network_rollup(f, atlas) for f in schaefer_frames if f]
    nets = [n for n in nets if n]
    if not nets:
        return []

    grouped: dict[str, dict] = {}
    for n in nets:
        for r in n:
            g = grouped.setdefault(r["Network"], {"sum": 0.0, "max": float("-inf"), "n": 0})
            g["sum"] += r["PercentDamage"]
            g["max"] = max(g["max"], r["PercentDamage"])
            g["n"] += 1

    rows = []
    for net in sorted(grouped):  # mirrors pandas groupby's default key-sorted order
        g = grouped[net]
        rows.append({
            "Network": net,
            "NSubjects": g["n"],
            "MeanPercentDamage": round(g["sum"] / g["n"], 2),
            "MaxPercentDamage": round(g["max"], 2),
        })
    rows.sort(key=lambda r: r["MeanPercentDamage"], reverse=True)
    return rows


def _group_tracts(tract_frames):
    """Aggregate per-subject tract overlap into per-tract group stats.

    Records with keys Tract, NSubjects, MeanPercentOverlap,
    MaxPercentOverlap, sorted by (NSubjects, MeanPercentOverlap) descending.
    """
    frames = [f for f in tract_frames if f]
    if not frames:
        return []

    grouped: dict[str, dict] = {}
    for f in frames:
        for r in f:
            g = grouped.setdefault(r["Tract"], {"sum": 0.0, "max": float("-inf"), "n": 0})
            g["sum"] += r["PercentOverlap"]
            g["max"] = max(g["max"], r["PercentOverlap"])
            g["n"] += 1

    rows = []
    for tract in sorted(grouped):
        g = grouped[tract]
        rows.append({
            "Tract": tract,
            "NSubjects": g["n"],
            "MeanPercentOverlap": round(g["sum"] / g["n"], 2),
            "MaxPercentOverlap": round(g["max"], 2),
        })
    rows.sort(key=lambda r: (-r["NSubjects"], -r["MeanPercentOverlap"]))
    return rows


def _aggregate_group_damage(frames):
    """Aggregate per-subject atlas-overlap records into per-region group
    stats. Records with keys RegionName, Group, NSubjects,
    MeanPercentDamage, MaxPercentDamage, sorted by (NSubjects,
    MeanPercentDamage) descending. ``Group`` is the first-seen value for
    that region across subjects (mirrors pandas' ``agg(Group=("Group",
    "first"))``)."""
    frames = [f for f in frames if f]
    if not frames:
        return []

    grouped: dict[str, dict] = {}
    for f in frames:
        for r in f:
            name = r["RegionName"]
            g = grouped.get(name)
            if g is None:
                g = grouped[name] = {"Group": r["Group"], "sum": 0.0,
                                     "max": float("-inf"), "n": 0}
            g["sum"] += r["PercentDamage"]
            g["max"] = max(g["max"], r["PercentDamage"])
            g["n"] += 1

    rows = []
    for name in sorted(grouped):
        g = grouped[name]
        rows.append({
            "RegionName": name,
            "Group": g["Group"],
            "NSubjects": g["n"],
            "MeanPercentDamage": round(g["sum"] / g["n"], 2),
            "MaxPercentDamage": round(g["max"], 2),
        })
    rows.sort(key=lambda r: (-r["NSubjects"], -r["MeanPercentDamage"]))
    return rows


# --------------------------------------------------------------------------- #
# Report-time (DataFrame) view -- report.py/plotting.py predate this refactor
# and still consume DataFrame-style tables; both are already only reachable
# once matplotlib (and, for the dk/schaefer flat maps, pandas via
# ggseg_flat.py) are installed, so building a DataFrame here doesn't add a
# new dependency to the write=True path. Neither function mutates its input.
# --------------------------------------------------------------------------- #
def _for_report_single(result):
    rr = dict(result)
    rr["atlases"] = []
    for a in result["atlases"]:
        e = dict(a)
        e["df"] = _frame(e.get("df"), _DAMAGE_COLUMNS)
        if "network_df" in e:
            e["network_df"] = _frame(e.get("network_df"), _NETWORK_COLUMNS)
        if "homolog_df" in e:
            e["homolog_df"] = _frame(e.get("homolog_df"), _HOMOLOG_COLUMNS)
        if "lobe_df" in e:
            e["lobe_df"] = _frame(e.get("lobe_df"), _LOBE_COLUMNS)
        rr["atlases"].append(e)
    if result.get("tracts") is not None:
        t = dict(result["tracts"])
        t["df"] = _frame(t.get("df"), _TRACT_COLUMNS)
        rr["tracts"] = t
    return rr


def _for_report_group(result):
    rr = dict(result)
    rr["atlases"] = []
    for a in result["atlases"]:
        e = dict(a)
        e["df"] = _frame(e.get("df"), _GROUP_DAMAGE_COLUMNS)
        if "network_df" in e:
            e["network_df"] = _frame(e.get("network_df"), _GROUP_NETWORK_COLUMNS)
        rr["atlases"].append(e)
    if result.get("tracts") is not None:
        t = dict(result["tracts"])
        t["df"] = _frame(t.get("df"), _GROUP_TRACT_COLUMNS)
        rr["tracts"] = t
    return rr


# --------------------------------------------------------------------------- #
# frame="pandas" opt-in conversion of a returned result -- the only place a
# caller of analyze_single/analyze_group ever gets pandas objects back.
# --------------------------------------------------------------------------- #
def _apply_frame_single(result):
    for a in result["atlases"]:
        a["df"] = _frame(a.get("df"), _DAMAGE_COLUMNS)
        if "network_df" in a:
            a["network_df"] = _frame(a.get("network_df"), _NETWORK_COLUMNS)
        if "homolog_df" in a:
            a["homolog_df"] = _frame(a.get("homolog_df"), _HOMOLOG_COLUMNS)
        if "lobe_df" in a:
            a["lobe_df"] = _frame(a.get("lobe_df"), _LOBE_COLUMNS)
    if result.get("tracts") is not None:
        result["tracts"]["df"] = _frame(result["tracts"].get("df"), _TRACT_COLUMNS)


def _apply_frame_group(result):
    for a in result["atlases"]:
        a["df"] = _frame(a.get("df"), _GROUP_DAMAGE_COLUMNS)
        if "network_df" in a:
            a["network_df"] = _frame(a.get("network_df"), _GROUP_NETWORK_COLUMNS)
    if result.get("tracts") is not None:
        result["tracts"]["df"] = _frame(result["tracts"].get("df"), _GROUP_TRACT_COLUMNS)


# --------------------------------------------------------------------------- #
# Single lesion
# --------------------------------------------------------------------------- #
def analyze_single(lesion_path, atlases=None, out_dir=None, threshold=0.5,
                   tracts=True, write=True, plots=False, verbose=False, frame=None):
    """Analyse one lesion; return a result dict and (optionally) write
    per-atlas CSVs (``write=True``) and a self-contained HTML report with
    images (``write=True`` and ``plots=True``).

    ``lesion_path`` accepts anything :func:`lqtpy.io.as_lesion_image` does: a
    path (str/os.PathLike), a nibabel image, or an ``(array, affine)`` tuple.

    ``plots=False`` (the default) never imports matplotlib/nilearn/pandas:
    every atlas/tract table is still computed and (with ``write=True``)
    written to CSV, but no glass-brain/slice/bar/flat-map image and no HTML
    report are produced. Pass ``plots=True`` for the original behaviour.

    Atlas/tract tables in the returned result are **records** (``list[dict]``)
    by default. Pass ``frame="pandas"`` to get ``pandas.DataFrame`` objects
    instead (requires the ``tables`` extra).
    """
    t0 = time.time()
    atlases = atlases or DEFAULT_ATLASES
    name = _stem(lesion_path)
    if verbose:
        print(f"[lqtpy] analysing {name} ...")

    img, was_binary = load_lesion(lesion_path, threshold)
    morph = compute_morphometry(img)
    cut = morph["center_of_mass_mm"]

    glass_uri = slices_uri = None
    if plots:
        from . import plotting
        glass_uri = plotting.glass_brain_png(img, title=f"{name}")
        slices_uri = plotting.slices_png(img, cut_coords=cut, title="Lesion overlay")

    atlas_results, csv_tables = [], {}
    for key in atlases:
        atlas = load_atlas(key)
        df = region_overlap(img, atlas)
        csv_tables[key] = df
        entry = {"key": key, "name": atlas.name, "df": df}
        if plots:
            from . import plotting
            entry["bar_uri"] = plotting.damage_bar_png(
                _frame(df, _DAMAGE_COLUMNS), title=f"{atlas.name}: top damaged")
            entry["flat_uris"] = _flat_maps(atlas, df)
        _add_extras(entry, key, atlas, df, plots=plots)
        atlas_results.append(entry)
        if verbose:
            print(f"[lqtpy]   {key}: {len(df)} regions overlap")

    tract_df = _tracts.tract_overlap(img) if (tracts and _tracts.tracts_available()) else None
    tract_uri = None
    if plots and tract_df is not None:
        from . import plotting
        tract_uri = plotting.tract_bar_png(_frame(tract_df, _TRACT_COLUMNS))

    elapsed = time.time() - t0
    result = {
        "name": name,
        "source_path": lesion_path,
        "was_binary": was_binary,
        "morphometry": morph,
        "glass_uri": glass_uri,
        "slices_uri": slices_uri,
        "atlases": atlas_results,
        "tracts": None if tract_df is None else {"df": tract_df, "bar_uri": tract_uri},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed": _fmt_elapsed(elapsed),
    }

    if write:
        out_dir = out_dir or os.path.join("lqtpy_output", name)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{name}_morphometry.json"), "w") as fh:
            json.dump(morph, fh, indent=2)
        for key, records in csv_tables.items():
            _write_csv(os.path.join(out_dir, f"{name}_{key}_damage.csv"), records, _DAMAGE_COLUMNS)
        if tract_df is not None:
            _write_csv(os.path.join(out_dir, f"{name}_tracts.csv"), tract_df, _TRACT_COLUMNS)
        if plots:
            from .report import render_single
            report_path = os.path.join(out_dir, f"{name}_report.html")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(render_single(_for_report_single(result)))
            result["report_path"] = report_path
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
    """Accept a directory, a list of paths, or a list of (name, lesion).

    Each list item is either a ``(name, lesion)`` pair -- where ``lesion``
    is anything :func:`lqtpy.io.as_lesion_image` accepts -- or a bare lesion
    source (path/nibabel image/(array, affine) tuple), auto-named via
    ``_stem``. A 2-tuple is only treated as ``(name, lesion)`` when its first
    element is a string, so a bare ``(array, affine)`` tuple passed directly
    as a list item is treated as one lesion source, not a (name, lesion) pair.
    """
    if isinstance(inputs, str) and os.path.isdir(inputs):
        files = sorted(
            os.path.join(inputs, f) for f in os.listdir(inputs)
            if f.endswith((".nii", ".nii.gz"))
        )
        return [(_stem(f), f) for f in files]
    out = []
    for item in inputs:
        if isinstance(item, (tuple, list)) and len(item) == 2 and isinstance(item[0], str):
            out.append((item[0], item[1]))
        else:
            out.append((_stem(item), item))
    return out


def analyze_group(inputs, atlases=None, out_dir=None, threshold=0.5,
                  tracts=True, write=True, plots=False, verbose=False, frame=None):
    """Analyse a group of lesions; return a result dict and (optionally) write
    group CSVs (``write=True``) and a group HTML report with images
    (``write=True`` and ``plots=True``).

    Each item in ``inputs`` accepts anything :func:`lqtpy.io.as_lesion_image`
    does (see :func:`_normalise_inputs`).

    ``plots=False`` (the default) never imports matplotlib/nilearn/pandas.

    Atlas/tract tables in the returned result are **records** (``list[dict]``)
    by default. Pass ``frame="pandas"`` to get ``pandas.DataFrame`` objects
    instead (requires the ``tables`` extra).
    """
    t0 = time.time()
    atlases = atlases or DEFAULT_ATLASES
    pairs = _normalise_inputs(inputs)
    if len(pairs) < 1:
        raise InvalidLesionError("No lesion files found for group analysis.")
    if verbose:
        print(f"[lqtpy] group analysis: {len(pairs)} subjects")

    ref_key = atlases[0]
    ref_atlas = load_atlas(ref_key)
    ref_img = ref_atlas.img
    freq = np.zeros(ref_img.shape[:3], dtype=np.int32)

    subjects = []
    per_atlas_frames = {k: [] for k in atlases}
    tract_frames = []
    do_tracts = tracts and _tracts.tracts_available()

    for name, path in pairs:
        img, _ = load_lesion(path, threshold)
        morph = compute_morphometry(img)

        # Resample to the reference (first atlas) grid once; reuse for both the
        # frequency map and that atlas's region overlap.
        ref_mask = resample_mask_to(img, ref_img)
        freq += ref_mask.astype(np.int32)

        top_region = "-"
        for key in atlases:
            atlas = load_atlas(key)
            df = (region_overlap_from_mask(ref_mask, atlas) if key == ref_key
                  else region_overlap(img, atlas))
            per_atlas_frames[key].append(df)
            if key == ref_key and df:
                top_region = df[0]["RegionName"]
        if do_tracts:
            tract_frames.append(_tracts.tract_overlap(img))
        subjects.append({"name": name, "morphometry": morph, "top_region": top_region})
        if verbose:
            print(f"[lqtpy]   {name}: {morph['volume_cc']} cc")

    freq_img = nib.Nifti1Image(freq, ref_img.affine)
    overlap_uri = None
    if plots:
        from . import plotting
        overlap_uri = plotting.group_overlap_png(freq_img, len(pairs))

    # Aggregate each atlas across subjects.
    atlas_results, csv_tables = [], {}
    for key in atlases:
        agg = _aggregate_group_damage(per_atlas_frames[key])
        csv_tables[key] = agg
        atlas = load_atlas(key)
        entry = {"key": key, "name": atlas.name, "df": agg}
        if plots:
            from . import plotting
            entry["bar_uri"] = plotting.group_damage_bar_png(
                _frame(agg, _GROUP_DAMAGE_COLUMNS), title=f"{atlas.name}: group")
            entry["flat_uris"] = _flat_maps(atlas, agg, "MeanPercentDamage")
        if key == "schaefer":
            net_agg = _group_network(per_atlas_frames[key], atlas)
            entry["network_df"] = net_agg
            if plots:
                from . import plotting
                entry["network_bar_uri"] = plotting.group_network_bar_png(
                    _frame(net_agg, _GROUP_NETWORK_COLUMNS))
        atlas_results.append(entry)

    # Group white-matter tract aggregation.
    tract_agg = _group_tracts(tract_frames) if do_tracts else None
    tract_uri = None
    if plots and tract_agg is not None:
        from . import plotting
        tract_uri = plotting.group_tract_bar_png(_frame(tract_agg, _GROUP_TRACT_COLUMNS))

    vols = sorted(s["morphometry"]["volume_cc"] for s in subjects)
    lis = [s["morphometry"]["laterality_index"] for s in subjects]
    n_bilateral = sum(1 for s in subjects if s["morphometry"].get("bilateral"))
    elapsed = time.time() - t0
    result = {
        "n_subjects": len(pairs),
        "subjects": subjects,
        "median_volume_cc": round(float(np.median(vols)), 2),
        "median_laterality_index": round(float(np.median(lis)), 3),
        "n_bilateral": n_bilateral,
        "max_overlap": int(freq.max()),
        "overlap_uri": overlap_uri,
        "atlases": atlas_results,
        "tracts": None if tract_agg is None else {"df": tract_agg, "bar_uri": tract_uri},
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "elapsed": _fmt_elapsed(elapsed),
    }

    if write:
        out_dir = out_dir or os.path.join("lqtpy_output", "group")
        os.makedirs(out_dir, exist_ok=True)
        nib.save(freq_img, os.path.join(out_dir, "lesion_overlap_map.nii.gz"))
        for key, records in csv_tables.items():
            _write_csv(os.path.join(out_dir, f"group_{key}_damage.csv"), records, _GROUP_DAMAGE_COLUMNS)
        if tract_agg is not None:
            _write_csv(os.path.join(out_dir, "group_tracts.csv"), tract_agg, _GROUP_TRACT_COLUMNS)
        subj_rows = [{"Subject": s["name"], **s["morphometry"]} for s in subjects]
        subj_columns = ["Subject"] + list(subjects[0]["morphometry"].keys())
        _write_csv(os.path.join(out_dir, "group_subject_summary.csv"), subj_rows, subj_columns)
        if plots:
            from .report import render_group
            report_path = os.path.join(out_dir, "group_report.html")
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(render_group(_for_report_group(result)))
            result["report_path"] = report_path
        if verbose:
            suffix = f" -> {result['report_path']}" if "report_path" in result else ""
            print(f"[lqtpy] group done in {result['elapsed']}{suffix}")

    if frame == "pandas":
        _apply_frame_group(result)

    return result
