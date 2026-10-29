"""Shape measures (extra tier) and their plain-language interpretation.

All measures use only numpy/scipy and run on the lesion's bounding box, so
they cost milliseconds for typical lesions. Because a lesion with several
separate parts has no single meaningful "shape", the headline values are
for the **largest component**; ``per_component`` repeats them for every
component big enough to characterise.

Measures
--------
- ``surface_area_mm2``: exposed voxel-face area x 2/3. Counting faces of a
  voxelised surface overestimates a smooth surface's area by 3/2 on average
  over orientations (the Cauchy-Crofton result used in digital geometry),
  so the 2/3 factor gives an unbiased estimate without a mesh library.
- ``sphericity``: pi^(1/3) (6V)^(2/3) / A -- 1 for a perfect sphere, lower
  for anything flatter, longer or more convoluted.
- ``axis_lengths_mm``: full lengths of the principal axes, from the voxel
  coordinate covariance (a uniform solid ellipsoid with semi-axis a has
  variance a^2/5 along it, so length = 2 sqrt(5 lambda)).
- ``elongation`` = major / middle axis; ``flatness`` = middle / minor axis.
- ``surface_regularity``: sphericity divided by the sphericity of the
  ellipsoid with the same axes. Sphericity alone mixes two things --
  proportions (a clean tube scores low) and surface folding. This ratio
  keeps only the folding: ~1 for any smooth ellipsoid-like shape, lower
  for a crumpled or finger-like surface.
- ``solidity``: lesion voxels / voxels inside the convex hull of the lesion
  voxel centres. Both counts are digital, so the voxelisation bias cancels
  and small lesions aren't penalised. ~1 for a convex blob, low for
  lobulated, branching or C-shaped lesions. ``None`` for a lesion confined
  to one plane (no 3-D hull; the QC module reports those).
- ``major_axis_direction`` / ``minor_axis_direction``: anatomical direction.
- ``centroid_to_midline_mm``: |x| of the centre of mass.

Interpretation
--------------
:func:`describe_shape` turns the numbers into short descriptors. The band
edges were set on synthetic reference shapes (spheres, ellipsoids, plates,
tubes, lobulated, branching and C-shaped lesions at several sizes; run
``tools/calibrate_shape.py``), so each word means the same thing at any
lesion size. They are absolute, not relative to a patient population; a
cohort-relative layer can be added from calibration data later.
"""
from __future__ import annotations

import numpy as np

from ..registry import Descriptor, Field, MeasureSpec

MIN_VOXELS = 64          # below this, shape isn't meaningful at 1 mm
MAX_COMPONENTS = 20      # per_component is capped for very multifocal lesions

# Band edges (see module docstring and tools/calibrate_shape.py).
REGULARITY_IRREGULAR = 0.82
REGULARITY_HIGHLY = 0.68
SOLIDITY_SMOOTH = 0.90
SOLIDITY_MILD = 0.75
ELONGATION_MARKED = 2.5
ELONGATION_MILD = 1.6
FLATNESS_SHEET = 3.0
FLAT_MIN_SOLIDITY = 0.75  # a planar branching/C shape isn't a "sheet"


def _faces_area(sub: np.ndarray, zooms) -> float:
    """Total area of voxel faces between lesion and non-lesion voxels."""
    pad = np.pad(sub, 1)
    area = 0.0
    face_area = (zooms[1] * zooms[2], zooms[0] * zooms[2], zooms[0] * zooms[1])
    for ax in range(3):
        area += np.count_nonzero(np.diff(pad, axis=ax)) * face_area[ax]
    return float(area)


def _hull_voxel_count(ijk: np.ndarray) -> "int | None":
    """Number of grid voxels whose centres lie inside the convex hull of
    the lesion voxel centres ``ijk`` (voxel index space). ``None`` when the
    points are coplanar/collinear (no 3-D hull).

    Slice by slice along the first axis, the 3-D hull's cross-section is a
    convex polygon: intersect every hull edge with the plane, take the 2-D
    hull of those points, and test only that slice's voxels against its few
    edges. This gives the same count as testing every voxel against every
    3-D facet, at a fraction of the cost for large lesions.
    """
    from scipy.spatial import ConvexHull
    try:
        from scipy.spatial import QhullError
    except ImportError:  # older SciPy
        from scipy.spatial.qhull import QhullError

    pts = ijk.astype(float)
    try:
        hull = ConvexHull(pts)
    except (QhullError, ValueError):
        return None
    V = pts[hull.vertices]
    remap = {int(v): i for i, v in enumerate(hull.vertices)}
    edges = set()
    for tri in hull.simplices:
        a, b, c = (remap[int(t)] for t in tri)
        edges.update({(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(a, c), max(a, c))})
    E = np.array(sorted(edges))
    P0, P1 = V[E[:, 0]], V[E[:, 1]]
    lo = ijk.min(axis=0)
    hi = ijk.max(axis=0) + 1
    count = 0
    eps = 1e-9
    for x in range(lo[0], hi[0]):
        d0, d1 = P0[:, 0] - x, P1[:, 0] - x
        cross = (d0 * d1 <= 0) & (d0 != d1)
        t = d0[cross] / (d0[cross] - d1[cross])
        sect = P0[cross][:, 1:] + t[:, None] * (P1[cross][:, 1:] - P0[cross][:, 1:])
        on = V[np.abs(V[:, 0] - x) < eps][:, 1:]
        poly = np.vstack([sect, on]) if len(on) else sect
        if len(poly) == 0:
            continue
        ymin, zmin = np.floor(poly.min(axis=0) - eps).astype(int)
        ymax, zmax = np.ceil(poly.max(axis=0) + eps).astype(int)
        ys, zs = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(zmin, zmax + 1), indexing="ij")
        cand = np.column_stack([ys.ravel(), zs.ravel()]).astype(float)
        try:
            h2 = ConvexHull(poly)
            eq = h2.equations
            inside = np.all(cand @ eq[:, :2].T + eq[:, 2] <= 1e-7, axis=1)
            count += int(np.count_nonzero(inside))
        except (QhullError, ValueError):
            # Degenerate section (a point or a line): count the lattice points on it.
            if len(poly) == 1 or np.allclose(poly, poly[0]):
                count += int(np.all(np.abs(poly[0] - np.round(poly[0])) < 1e-7))
            else:
                a, b = poly[np.argmin(poly[:, 0] + poly[:, 1])], poly[np.argmax(poly[:, 0] + poly[:, 1])]
                seg = b - a
                L = np.dot(seg, seg)
                tt = np.clip(((cand - a) @ seg) / L, 0, 1)
                dist = np.linalg.norm(cand - (a + tt[:, None] * seg), axis=1)
                count += int(np.count_nonzero(dist < 1e-7))
    return count


def _ellipsoid_sphericity(a: float, b: float, c: float) -> float:
    """Sphericity of an ellipsoid with semi-axes a, b, c (Knud Thomsen's
    surface-area approximation, error < 1.1%)."""
    p = 1.6075
    area = 4 * np.pi * (((a * b) ** p + (a * c) ** p + (b * c) ** p) / 3) ** (1 / p)
    vol = 4 / 3 * np.pi * a * b * c
    return float(np.pi ** (1 / 3) * (6 * vol) ** (2 / 3) / area)


_DIRECTIONS = ("left-right", "front-back", "up-down")


def _direction(vec: np.ndarray) -> str:
    w = np.abs(vec) / (np.linalg.norm(vec) or 1.0)
    order = np.argsort(-w)
    if w[order[0]] >= 0.85:
        return _DIRECTIONS[order[0]]
    return f"oblique, mostly {_DIRECTIONS[order[0]]}"


def shape_of(mask_sub: np.ndarray, offset, affine) -> dict:
    """Shape measures for one binary component given as a cropped mask."""
    affine = np.asarray(affine, dtype=float)
    zooms = np.linalg.norm(affine[:3, :3], axis=0)
    ijk = np.argwhere(mask_sub) + np.asarray(offset)
    n = int(ijk.shape[0])
    voxel_vol = float(abs(np.linalg.det(affine[:3, :3])))
    volume = n * voxel_vol
    if n < MIN_VOXELS:
        return {"n_voxels": n, "volume_mm3": round(volume, 1), "characterised": False}
    area = _faces_area(mask_sub, zooms) * 2.0 / 3.0
    sphericity = (np.pi ** (1 / 3)) * (6 * volume) ** (2 / 3) / area if area else float("nan")
    # The hull of the boundary voxels is the hull of the lesion; far fewer points.
    from scipy import ndimage as _nd
    boundary = mask_sub & ~_nd.binary_erosion(mask_sub)
    hull_n = _hull_voxel_count(np.argwhere(boundary) + np.asarray(offset))
    solidity = (n / hull_n) if hull_n else None
    world = ijk @ affine[:3, :3].T + affine[:3, 3]
    evals, evecs = np.linalg.eigh(np.cov(world.T))
    order = np.argsort(evals)[::-1]
    evals = np.clip(evals[order], 0, None)
    evecs = evecs[:, order]
    # Solid-ellipsoid axis length, floored at one voxel's extent.
    lengths = np.maximum(2 * np.sqrt(5 * evals), zooms.min())
    regularity = sphericity / _ellipsoid_sphericity(*(lengths / 2))
    com = world.mean(axis=0)
    return {
        "n_voxels": n,
        "volume_mm3": round(volume, 1),
        "characterised": True,
        "surface_area_mm2": round(area, 1),
        "sphericity": round(float(min(sphericity, 1.0)), 3),
        "surface_regularity": round(float(min(regularity, 1.0)), 3),
        "solidity": None if solidity is None else round(float(min(solidity, 1.0)), 3),
        "axis_lengths_mm": [round(float(v), 1) for v in lengths],
        "elongation": round(float(lengths[0] / lengths[1]), 2),
        "flatness": round(float(lengths[1] / lengths[2]), 2),
        "major_axis_direction": _direction(evecs[:, 0]),
        "minor_axis_direction": _direction(evecs[:, 2]),
        "centroid_to_midline_mm": round(float(abs(com[0])), 1),
    }


def compute(prepared) -> dict:
    from ..morphometry import component_boxes, components

    labels, lo, sizes = components(prepared)
    boxes = component_boxes(prepared)
    per = []
    for i in range(min(len(sizes), MAX_COMPONENTS)):
        if sizes[i] < MIN_VOXELS:
            break  # sizes are sorted, largest first
        sl = boxes[i]                       # the part's own bounding box
        sub = labels[sl] == (i + 1)
        vals = shape_of(sub, np.asarray([s.start for s in sl]) + lo, prepared.affine)
        vals["component"] = i + 1
        per.append(vals)
    if per:
        headline = dict(per[0])
    else:
        headline = {"n_voxels": int(sizes[0]) if len(sizes) else 0, "characterised": False}
    headline.pop("component", None)
    headline["basis"] = "largest component" if len(sizes) > 1 else "whole lesion"
    headline["per_component"] = per
    return headline


_REF = None


def reference() -> "dict | None":
    """The bundled cohort reference (``data/shape_reference.json``): per
    lesion-size band, percentiles of each measure. Aggregates only."""
    global _REF
    if _REF is None:
        import json
        import os
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "shape_reference.json")
        try:
            with open(path, encoding="utf-8") as fh:
                _REF = json.load(fh)
        except OSError:
            _REF = {}
    return _REF or None


def percentile(measure: str, value, volume_mm3: float) -> "float | None":
    """Where ``value`` falls among reference lesions of similar size
    (0-100), or None without a reference for that size band."""
    ref = reference()
    if ref is None or value is None:
        return None
    for band in ref["bands"]:
        hi = band["max_mm3"] if band["max_mm3"] is not None else float("inf")
        if band["min_mm3"] <= volume_mm3 < hi and measure in band:
            return float(np.interp(value, band[measure], ref["percentiles"],
                                   left=2.5, right=97.5))
    return None


_COMPARATIVE = {"irregular": "more irregular", "smooth": "smoother",
               "lobulated": "more lobulated", "compact": "more compact"}


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _relative(measure, value, vol, low_word, high_word):
    """(suffix for the text, detail) -- suffix only when notable (outside
    the 10th-90th percentile of reference lesions of similar size)."""
    pc = percentile(measure, value, vol)
    if pc is None:
        return "", ""
    detail = f"; {_ordinal(int(round(pc)))} percentile of similar-sized reference lesions"
    if pc <= 10:
        return f" ({_COMPARATIVE[low_word]} than 90% of similar-sized reference lesions)", detail
    if pc >= 90:
        return f" ({_COMPARATIVE[high_word]} than 90% of similar-sized reference lesions)", detail
    return "", detail


def describe_shape(v: dict) -> "list[Descriptor]":
    """Descriptors for one component's shape values, in reading order:
    proportions, then surface, then outline."""
    if not v.get("characterised"):
        return [Descriptor("shape", "too small to characterise its shape", "too-small",
                           f"{v.get('n_voxels', 0)} voxels (< {MIN_VOXELS})")]
    out = []
    el, fl = v["elongation"], v["flatness"]
    reg, sol = v["surface_regularity"], v["solidity"]
    solid_enough = sol is None or sol >= FLAT_MIN_SOLIDITY

    # 1. Proportions.
    if fl >= FLATNESS_SHEET and el < ELONGATION_MARKED and solid_enough:
        out.append(Descriptor("shape", "flat and sheet-like", "flat",
                              f"middle/shortest axis {fl}; thinnest {v['minor_axis_direction']}"))
    elif el >= ELONGATION_MARKED:
        out.append(Descriptor("shape", f"markedly elongated ({v['major_axis_direction']})",
                              "elongated-marked", f"longest/middle axis {el}"))
    elif el >= ELONGATION_MILD:
        out.append(Descriptor("shape", f"elongated ({v['major_axis_direction']})",
                              "elongated", f"longest/middle axis {el}"))
    elif fl < ELONGATION_MILD and reg >= REGULARITY_IRREGULAR and (sol is None or sol >= SOLIDITY_MILD):
        out.append(Descriptor("shape", "compact and roughly round", "compact",
                              f"sphericity {v['sphericity']}"))

    vol = v.get("volume_mm3", 0.0)
    # 2. Surface folding, independent of proportions.
    sfx, det = _relative("surface_regularity", reg, vol, "irregular", "smooth")
    if reg < REGULARITY_HIGHLY:
        out.append(Descriptor("shape", f"a highly irregular surface{sfx}", "surface-highly-irregular",
                              f"surface regularity {reg}{det}"))
    elif reg < REGULARITY_IRREGULAR:
        out.append(Descriptor("shape", f"an irregular surface{sfx}", "surface-irregular",
                              f"surface regularity {reg}{det}"))
    elif sfx and "smoother" in sfx and (sol is None or sol < SOLIDITY_SMOOTH):
        # Only worth saying when the outline sentence won't already say smooth.
        out.append(Descriptor("shape", f"a smooth surface{sfx}", "surface-smooth",
                              f"surface regularity {reg}{det}"))

    # 3. Outline (convexity).
    if sol is not None:
        sfx, det = _relative("solidity", sol, vol, "lobulated", "compact")
        if sol >= SOLIDITY_SMOOTH:
            out.append(Descriptor("shape", f"a smooth, convex outline{sfx}", "smooth",
                                  f"solidity {sol}{det}"))
        elif sol >= SOLIDITY_MILD:
            out.append(Descriptor("shape", f"a mildly lobulated outline{sfx}", "lobulated-mild",
                                  f"solidity {sol}{det}"))
        else:
            out.append(Descriptor("shape", f"a lobulated or branching outline{sfx}", "lobulated",
                                  f"solidity {sol}{det}"))
    return out


def interpret(values: dict, context: dict) -> "list[Descriptor]":
    return describe_shape(values)


_FEATURE_LEVELS = {"surface-highly-irregular", "surface-irregular", "surface-smooth",
                   "smooth", "lobulated-mild", "lobulated"}


def phrase(descriptors: "list[Descriptor]") -> str:
    """Join descriptors into a predicate: 'elongated (front-back), with a
    smooth, convex outline' or, with no proportion word, 'with an irregular
    surface and a lobulated or branching outline'."""
    head = [d.text for d in descriptors if d.level not in _FEATURE_LEVELS]
    feats = [d.text for d in descriptors if d.level in _FEATURE_LEVELS]
    out = ", ".join(head)
    if feats:
        out += (", with " if out else "with ") + " and ".join(feats)
    return out


def sentence(descriptors: "list[Descriptor]", subject: str = "The lesion") -> str:
    """A full sentence: 'The lesion is compact and roughly round, with a
    smooth, convex outline.' / 'The lesion has an irregular surface and a
    lobulated or branching outline.'"""
    head = [d.text for d in descriptors if d.level not in _FEATURE_LEVELS]
    feats = [d.text for d in descriptors if d.level in _FEATURE_LEVELS]
    if head and head[0].startswith("too small"):
        return f"{subject} is {head[0]}."
    if head:
        return f"{subject} is {phrase(descriptors)}."
    if feats:
        return f"{subject} has {' and '.join(feats)}."
    return ""


SHAPE = MeasureSpec(
    name="shape",
    label="Shape",
    tier="extra",
    version="1",
    compute=compute,
    interpret=interpret,
    min_voxels=MIN_VOXELS,
    description="Surface area, sphericity, surface regularity, solidity, principal axes.",
    fields=(
        Field("surface_area_mm2", "mm²", "Estimated area of the lesion's outer surface."),
        Field("sphericity", "0-1", "How close to a ball the lesion is: 1 is a perfect sphere; "
              "flatter, longer or more folded shapes score lower."),
        Field("surface_regularity", "0-1", "How smooth the surface is for the lesion's "
              "proportions: near 1 is smooth; low values mean a folded or crumpled surface."),
        Field("solidity", "0-1", "How much of its convex envelope the lesion fills: near 1 is a "
              "compact blob; low values mean lobes, branches or a curved (C-shaped) lesion."),
        Field("axis_lengths_mm", "mm", "Lengths of the lesion's longest, middle and shortest axes."),
        Field("elongation", "ratio", "Longest axis / middle axis: 1 is not elongated, 2.5+ is "
              "clearly elongated."),
        Field("flatness", "ratio", "Middle axis / shortest axis: 3+ means the lesion is a thin sheet."),
        Field("major_axis_direction", "", "Anatomical direction of the longest axis."),
        Field("centroid_to_midline_mm", "mm", "Distance from the lesion's centre to the midline."),
    ),
)
