"""Atlas region overlap ("parcel damage"), network rollup, L/R symmetry.

Every function here returns **records** (``list[dict]``) by default -- the
internal representation used throughout ``lqtpy``. Pass ``frame="pandas"`` to
get a ``pandas.DataFrame`` instead (imported lazily; raises a clear
``ImportError`` naming the ``tables`` extra if pandas isn't installed).
"""
from __future__ import annotations

import numpy as np

from .atlases import Atlas
from .errors import MissingDependencyError, NoLobeMappingError
from .io import resample_mask_to

_COLUMNS = ["LabelID", "RegionName", "Group", "RegionVoxels",
            "LesionVoxels", "PercentDamage", "PercentOfLesion"]

_NETWORK_COLUMNS = ["Network", "NetworkVoxels", "LesionVoxels",
                    "PercentDamage", "PercentOfLesion"]

_LOBE_COLUMNS = ["Lobe", "LobeVoxels", "LesionVoxels",
                 "PercentDamage", "PercentOfLesion"]

_HOMOLOG_COLUMNS = ["Region", "LeftPercent", "RightPercent", "AbsAsymmetry"]


def _as_frame(records, columns):
    """Lazily convert records to a pandas DataFrame with a fixed column order."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise MissingDependencyError(
            "tables",
            "frame='pandas' requires pandas. Install the 'tables' extra: "
            "pip install 'lqtpy[tables]'"
        ) from exc
    return pd.DataFrame(records, columns=columns)


def _finish(records, columns, frame):
    if frame == "pandas":
        return _as_frame(records, columns)
    return records


def region_overlap_from_mask(mask, atlas: Atlas, frame: str | None = None):
    """Per-region damage from a boolean lesion mask already on the atlas grid.

    Uses the atlas's cached label array and per-label voxel counts, so the
    subject-invariant region volumes are computed once (not per call).

    Returns a list of dicts (records) with keys LabelID, RegionName, Group,
    RegionVoxels, LesionVoxels, PercentDamage, PercentOfLesion, sorted by
    PercentDamage descending. Pass ``frame="pandas"`` for a DataFrame.
    """
    labels = atlas.label_array()
    region_voxels = atlas.region_voxels()

    lesion_total = int(mask.sum())
    hit = labels[mask]                       # region id at each lesion voxel
    lesion_by_region = np.bincount(hit[hit > 0])

    rows = []
    for lid, meta in atlas.labels.items():
        region_vox = int(region_voxels[lid]) if lid < len(region_voxels) else 0
        les_vox = int(lesion_by_region[lid]) if lid < len(lesion_by_region) else 0
        if les_vox == 0 or region_vox == 0:
            continue
        rows.append({
            "LabelID": int(lid),
            "RegionName": meta.RegionName,
            "Group": meta.Group,
            "RegionVoxels": region_vox,
            "LesionVoxels": les_vox,
            "PercentDamage": round(100.0 * les_vox / region_vox, 2),
            "PercentOfLesion": round(100.0 * les_vox / lesion_total, 2) if lesion_total else 0.0,
        })

    rows.sort(key=lambda r: r["PercentDamage"], reverse=True)
    return _finish(rows, _COLUMNS, frame)


def region_overlap(lesion_img, atlas: Atlas, frame: str | None = None):
    """Resample ``lesion_img`` onto the atlas grid, then compute region overlap."""
    mask = resample_mask_to(lesion_img, atlas.img)
    return region_overlap_from_mask(mask, atlas, frame=frame)


def network_rollup(overlap_records, atlas: Atlas, frame: str | None = None):
    """Aggregate per-region damage to per-network (Yeo-7) totals.

    Denominators use the atlas's full per-network voxel totals (including
    undamaged regions) so PercentDamage is correct. Intended for the Schaefer
    atlas, whose ``Group`` column is the Yeo network token.

    ``overlap_records`` is the records (or DataFrame) returned by
    ``region_overlap``/``region_overlap_from_mask``. Returns records with
    keys Network, NetworkVoxels, LesionVoxels, PercentDamage, PercentOfLesion,
    sorted by PercentDamage descending.
    """
    overlap_records = _coerce_records(overlap_records)
    region_voxels = atlas.region_voxels()
    # Full voxel total per network across all labelled regions.
    net_total: dict[str, int] = {}
    for lid, meta in atlas.labels.items():
        vox = int(region_voxels[lid]) if lid < len(region_voxels) else 0
        net_total[meta.Group] = net_total.get(meta.Group, 0) + vox

    lesion_grand_total = sum(r["LesionVoxels"] for r in overlap_records)

    # Group by "Group" (network), preserving first-seen order -- matches the
    # DataFrame's original row order into groupby, before the final sort.
    grouped: "dict[str, int]" = {}
    for r in overlap_records:
        grouped[r["Group"]] = grouped.get(r["Group"], 0) + r["LesionVoxels"]

    rows = []
    for net, les in grouped.items():
        tot = net_total.get(net, 0)
        rows.append({
            "Network": net,
            "NetworkVoxels": tot,
            "LesionVoxels": les,
            "PercentDamage": round(100.0 * les / tot, 2) if tot else 0.0,
            "PercentOfLesion": round(100.0 * les / lesion_grand_total, 2) if lesion_grand_total else 0.0,
        })
    rows.sort(key=lambda r: r["PercentDamage"], reverse=True)
    return _finish(rows, _NETWORK_COLUMNS, frame)


def atlas_has_lobes(atlas: Atlas) -> bool:
    """True if ``atlas`` defines a ``Lobe`` for at least one region.

    ``aal``/``dk`` define a lobe for every region; ``schaefer`` (Yeo-7
    networks only) defines none, so this is ``False`` for it.
    """
    return any(meta.Lobe for meta in atlas.labels.values())


def lobe_rollup(overlap_records, atlas: Atlas, frame: str | None = None):
    """Aggregate per-region damage to per-lobe totals.

    Mirrors :func:`network_rollup` in shape and conventions, but groups by
    each region's own ``Lobe`` field (frontal/parietal/temporal/...) rather
    than the atlas's ``Group`` field (hemisphere for aal/dk, Yeo-7 network
    for schaefer) -- see :class:`~lqtpy.atlases.RegionMeta`. Denominators use
    the atlas's full per-lobe voxel totals (including undamaged regions) so
    PercentDamage is correct.

    ``overlap_records`` is the records (or DataFrame) returned by
    ``region_overlap``/``region_overlap_from_mask``. Returns records with
    keys Lobe, LobeVoxels, LesionVoxels, PercentDamage, PercentOfLesion,
    sorted by PercentDamage descending.

    Raises :class:`~lqtpy.errors.NoLobeMappingError` if ``atlas`` doesn't
    define a lobe for any region (see :func:`atlas_has_lobes`).
    """
    if not atlas_has_lobes(atlas):
        raise NoLobeMappingError(
            f"Atlas '{atlas.key}' has no lobe mapping for any region; "
            "lobe_overlap/lobe_rollup needs RegionMeta.Lobe to be set.")

    overlap_records = _coerce_records(overlap_records)
    region_voxels = atlas.region_voxels()
    # Full voxel total per lobe across all labelled regions.
    lobe_total: dict[str, int] = {}
    for lid, meta in atlas.labels.items():
        if not meta.Lobe:
            continue
        vox = int(region_voxels[lid]) if lid < len(region_voxels) else 0
        lobe_total[meta.Lobe] = lobe_total.get(meta.Lobe, 0) + vox

    lesion_grand_total = sum(r["LesionVoxels"] for r in overlap_records)

    # Group by each region's LabelID -> Lobe (overlap records only carry
    # Group, not Lobe, so look the region back up in the atlas).
    grouped: "dict[str, int]" = {}
    for r in overlap_records:
        meta = atlas.labels.get(r["LabelID"])
        lobe = meta.Lobe if meta is not None else None
        if not lobe:
            continue
        grouped[lobe] = grouped.get(lobe, 0) + r["LesionVoxels"]

    rows = []
    for lobe, les in grouped.items():
        tot = lobe_total.get(lobe, 0)
        rows.append({
            "Lobe": lobe,
            "LobeVoxels": tot,
            "LesionVoxels": les,
            "PercentDamage": round(100.0 * les / tot, 2) if tot else 0.0,
            "PercentOfLesion": round(100.0 * les / lesion_grand_total, 2) if lesion_grand_total else 0.0,
        })
    rows.sort(key=lambda r: r["PercentDamage"], reverse=True)
    return _finish(rows, _LOBE_COLUMNS, frame)


def homolog_symmetry(overlap_records, frame: str | None = None):
    """Pair left/right homologous regions (``_L``/``_R``) and compare damage.

    Intended for AAL, whose region names carry an explicit hemisphere suffix
    (e.g. ``Calcarine_L`` / ``Calcarine_R``). Regions damaged on at least one
    side are included. Returns records with keys Region, LeftPercent,
    RightPercent, AbsAsymmetry, sorted by max(LeftPercent, RightPercent) desc.
    """
    overlap_records = _coerce_records(overlap_records)
    if not overlap_records:
        return _finish([], _HOMOLOG_COLUMNS, frame)
    dmg = {r["RegionName"]: r["PercentDamage"] for r in overlap_records}
    bases: dict[str, dict] = {}
    for name, val in dmg.items():
        if name.endswith("_L"):
            bases.setdefault(name[:-2], {})["L"] = float(val)
        elif name.endswith("_R"):
            bases.setdefault(name[:-2], {})["R"] = float(val)
    rows = []
    for base, sides in bases.items():
        left = sides.get("L", 0.0)
        right = sides.get("R", 0.0)
        rows.append({
            "Region": base,
            "LeftPercent": round(left, 2),
            "RightPercent": round(right, 2),
            "AbsAsymmetry": round(abs(left - right), 2),
        })
    rows.sort(key=lambda r: max(r["LeftPercent"], r["RightPercent"]), reverse=True)
    return _finish(rows, _HOMOLOG_COLUMNS, frame)


def _coerce_records(records_or_frame):
    """Accept either records (list[dict]) or a pandas DataFrame."""
    if hasattr(records_or_frame, "to_dict") and hasattr(records_or_frame, "columns"):
        return records_or_frame.to_dict("records")
    return list(records_or_frame)
