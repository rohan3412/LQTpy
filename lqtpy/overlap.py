"""Atlas region overlap ("parcel damage"), network rollup, L/R symmetry.

Every function here returns **records** (``list[dict]``) by default -- the
internal representation used throughout ``lqtpy``. Pass ``frame="pandas"`` to
get a ``pandas.DataFrame`` instead (imported lazily; raises a clear
``ImportError`` naming the ``tables`` extra if pandas isn't installed).

PercentOfLesion
---------------
Everywhere in lqtpy -- region rows, network/lobe rollups -- ``PercentOfLesion``
has one denominator: **every lesion voxel on the atlas grid**, labelled or
not. The part of the lesion that falls in no region is reported explicitly
by :func:`coverage` (``PercentUnlabelled``) instead of silently inflating
the rollups, so region rows + unlabelled add up to 100%. Lesion voxels
outside the atlas's field of view can't be on its grid at all; ``coverage``
reports those separately (``PercentOutsideFOV``, a share of the lesion's
own voxels).
"""
from __future__ import annotations

import warnings

import numpy as np

from .atlases import Atlas
from .errors import MissingDependencyError, NoLobeMappingError

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


def _labels_under(mask, atlas: Atlas) -> np.ndarray:
    """The atlas label at each lesion voxel (``labels[mask]``); ``mask`` is a
    boolean array on the atlas grid or a :class:`~lqtpy.prepared.GridMask`."""
    from .prepared import flat_indices
    return atlas.label_array().reshape(-1)[flat_indices(mask)]


def region_overlap_from_mask(mask, atlas: Atlas, frame: str | None = None):
    """Per-region damage from a lesion mask already on the atlas grid (a
    boolean array, or a :class:`~lqtpy.prepared.GridMask`).

    Uses the atlas's cached label array and per-label voxel counts, so the
    subject-invariant region volumes are computed once (not per call).

    Returns a list of dicts (records) with keys LabelID, RegionName, Group,
    RegionVoxels, LesionVoxels, PercentDamage, PercentOfLesion, sorted by
    PercentDamage descending. Pass ``frame="pandas"`` for a DataFrame.
    """
    region_voxels = atlas.region_voxels()
    hit = _labels_under(mask, atlas)         # region id at each lesion voxel
    lesion_total = int(hit.size)
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


def _grid_mask(lesion, atlas: Atlas):
    """The lesion on ``atlas``'s grid, as a :class:`~lqtpy.prepared.GridMask`.
    ``lesion`` is a PreparedLesion or a binary image (``> 0`` = lesion)."""
    from .prepared import PreparedLesion, prepare
    p = lesion if isinstance(lesion, PreparedLesion) else prepare(lesion, threshold=0.0)
    return p.on_grid(atlas.img)


def region_overlap(lesion, atlas: Atlas, frame: str | None = None):
    """Resample ``lesion`` onto the atlas grid, then compute region overlap.

    ``lesion`` is a :class:`~lqtpy.prepared.PreparedLesion` or a binary
    image (``> 0`` = lesion)."""
    return region_overlap_from_mask(_grid_mask(lesion, atlas), atlas, frame=frame)


def coverage(lesion, atlas: Atlas) -> dict:
    """How much of the lesion the atlas can describe.

    - ``LesionVoxelsOnGrid``: lesion voxels after resampling to the atlas
      grid (the ``PercentOfLesion`` denominator everywhere).
    - ``LabelledVoxels``/``PercentLabelled``: of those, voxels inside a
      region listed in the atlas's label table.
    - ``UnlabelledVoxels``/``PercentUnlabelled``: the rest -- typically
      white matter or CSF for a grey-matter atlas.
    - ``OutsideFOVVoxels``/``PercentOutsideFOV``: lesion voxels (on the
      lesion's own grid) that fall outside the atlas's field of view.
    """
    gm = _grid_mask(lesion, atlas)
    hit = _labels_under(gm, atlas)
    top = len(atlas.region_voxels()) - 1     # the largest label id in the volume
    known = np.zeros(max(top, max(atlas.labels, default=0)) + 1, dtype=bool)
    ids = np.fromiter((k for k in atlas.labels if k >= 0), dtype=np.int64)
    known[ids[ids < known.size]] = True
    labelled = int(np.count_nonzero(known[hit[hit >= 0]])) if hit.size else 0
    total = gm.n_voxels
    return {
        "LesionVoxelsOnGrid": total,
        "LabelledVoxels": labelled,
        "UnlabelledVoxels": total - labelled,
        "PercentLabelled": round(100.0 * labelled / total, 2) if total else 0.0,
        "PercentUnlabelled": round(100.0 * (total - labelled) / total, 2) if total else 0.0,
        "OutsideFOVVoxels": gm.outside_fov_voxels,
        "PercentOutsideFOV": round(100.0 * gm.outside_fov_fraction, 2),
    }


def _lesion_total(overlap_records, lesion_total, fn_name):
    if lesion_total is not None:
        return int(lesion_total)
    warnings.warn(
        f"{fn_name}() called without lesion_total: PercentOfLesion will be "
        "relative to labelled lesion voxels only, not the whole lesion. Pass "
        "lesion_total (e.g. coverage(...)['LesionVoxelsOnGrid']), or use "
        "lqtpy.api, which does.", stacklevel=3)
    return sum(r["LesionVoxels"] for r in overlap_records)


def network_rollup(overlap_records, atlas: Atlas, frame: str | None = None, *,
                   lesion_total: "int | None" = None):
    """Aggregate per-region damage to per-network (Yeo-7) totals.

    Denominators use the atlas's full per-network voxel totals (including
    undamaged regions) so PercentDamage is correct. Intended for the Schaefer
    atlas, whose ``Group`` column is the Yeo network token.

    ``lesion_total`` is the whole lesion's voxel count on the atlas grid
    (the ``PercentOfLesion`` denominator; see the module docstring).

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

    lesion_grand_total = _lesion_total(overlap_records, lesion_total, "network_rollup")

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


def lobe_rollup(overlap_records, atlas: Atlas, frame: str | None = None, *,
                lesion_total: "int | None" = None):
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

    ``lesion_total``: as for :func:`network_rollup`.

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

    lesion_grand_total = _lesion_total(overlap_records, lesion_total, "lobe_rollup")

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
