"""Geometric statistics for a single lesion, computed in its native space.

Physical sizes come from the image **affine** (the matrix that actually
places voxels in the world), never from the header's ``pixdim`` zooms: the
two can disagree in files from some converters, and every other lqtpy step
already trusts the affine. A disagreement is reported as
``header_zooms_mismatch``.

Whole-grid SciPy passes (connected-component labelling, centre of mass) run
on the lesion's bounding box only, so cost scales with the lesion, not the
7-million-voxel MNI grid.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

#: World-x within this many mm of 0 counts as midline (neither hemisphere).
MIDLINE_TOL_MM = 1e-3

_STRUCTURE_26 = np.ones((3, 3, 3), dtype=bool)


def voxel_sizes(affine) -> np.ndarray:
    """Voxel edge lengths (mm) along each array axis: column norms of the
    affine's 3x3 part. Correct for oblique affines too."""
    return np.linalg.norm(np.asarray(affine, dtype=float)[:3, :3], axis=0)


def voxel_volume(affine) -> float:
    """Volume (mm^3) of one voxel: |det| of the affine's 3x3 part."""
    return float(abs(np.linalg.det(np.asarray(affine, dtype=float)[:3, :3])))


def components(prepared):
    """Connected components (26-connectivity) of a prepared lesion, cached.

    Returns ``(labels, offset, sizes)``: ``labels`` is an int32 label array
    over the lesion's padded bounding box (0 = background, 1..n ordered by
    size, largest first), ``offset`` is that box's origin on the lesion
    grid, ``sizes[i]`` is the voxel count of component ``i + 1``.
    """
    def build():
        sub, lo = prepared.cropped(pad=1)
        lab, n = ndimage.label(sub, structure=_STRUCTURE_26)
        if n == 0:
            return np.zeros(sub.shape, np.int32), lo, np.zeros(0, np.int64)
        sizes = np.bincount(lab.ravel())[1:]
        # Relabel so component 1 is the largest (stable for ties).
        order = np.argsort(-sizes, kind="stable")
        remap = np.zeros(n + 1, dtype=np.int32)
        remap[order + 1] = np.arange(1, n + 1, dtype=np.int32)
        lab = remap[lab]
        lab.flags.writeable = False
        return lab, lo, sizes[order]
    return prepared.cached("components", build)


def component_boxes(prepared):
    """``ndimage.find_objects`` of :func:`components`' labels, cached."""
    labels = components(prepared)[0]
    return prepared.cached("component_boxes", lambda: ndimage.find_objects(labels))


def part_voxels(labels, boxes, k, offset=(0, 0, 0)) -> np.ndarray:
    """``np.argwhere(labels == k) + offset`` (same rows, same order), scanning
    only part ``k``'s bounding box ``boxes[k - 1]`` (from
    ``ndimage.find_objects``) instead of the whole label array."""
    from .prepared import _nonzero
    sl = boxes[k - 1]
    ijk = np.column_stack(_nonzero(labels[sl] == k))
    return ijk + (np.asarray(offset) + np.array([s.start for s in sl]))


def label_centres(labels, n: int, offset=(0, 0, 0)) -> np.ndarray:
    """``(n, 3)`` voxel-space centres of labels 1..n, shifted by ``offset``:
    the values ``ndimage.center_of_mass`` gives for the same labels in a
    larger array whose origin is ``-offset`` (exact integer sums, one
    division), without a full-size float array and label sort."""
    nz = np.flatnonzero(labels)
    lab = labels.reshape(-1)[nz]
    ijk = np.unravel_index(nz, labels.shape)
    cnt = np.bincount(lab, minlength=n + 1)[1:n + 1].astype(float)
    return np.column_stack([np.bincount(lab, weights=c + o, minlength=n + 1)[1:n + 1] / cnt
                            for c, o in zip(ijk, offset)])


def compute_morphometry(lesion) -> dict:
    """Morphometric measures for a lesion.

    ``lesion`` is a :class:`~lqtpy.prepared.PreparedLesion`, or a binary
    image (anything ``> 0`` is lesion, for backward compatibility).
    """
    from .prepared import PreparedLesion, prepare

    p = lesion if isinstance(lesion, PreparedLesion) else prepare(lesion, threshold=0.0)
    return dict(p.cached("morphometry", lambda: _compute(p)))


def _compute(p) -> dict:
    affine = p.affine
    zooms = voxel_sizes(affine)
    voxel_vol = voxel_volume(affine)
    header_zooms = None
    try:
        header_zooms = np.abs(np.asarray(p.img.header.get_zooms()[:3], dtype=float))
    except Exception:
        pass
    mismatch = bool(header_zooms is not None
                    and not np.allclose(header_zooms, zooms, rtol=1e-3, atol=1e-3))

    n_vox = p.n_voxels
    volume_mm3 = n_vox * voxel_vol

    ijk = p.voxel_coords()
    com_vox = ijk.mean(axis=0)
    com_world = affine[:3, :3] @ com_vox + affine[:3, 3]

    mins = np.asarray(p.bbox[0])
    maxs = np.asarray(p.bbox[1]) - 1
    bbox_vox = maxs - mins + 1
    bbox_mm = bbox_vox * zooms

    _, _, comp_sizes = components(p)
    n_comp = int(comp_sizes.size)
    largest_vox = int(comp_sizes[0]) if n_comp else 0

    # Hemisphere split by world-x sign (MNI convention: x < 0 = left).
    world_x = ijk @ affine[0, :3] + affine[0, 3]
    n_left = int(np.count_nonzero(world_x < -MIDLINE_TOL_MM))
    n_right = int(np.count_nonzero(world_x > MIDLINE_TOL_MM))
    n_mid = n_vox - n_left - n_right

    # Laterality index: +1 fully left, -1 fully right, ~0 bilateral. A lesion is
    # "bilateral" when each hemisphere holds at least 5% of its voxels.
    denom = n_left + n_right
    laterality_index = round((n_left - n_right) / denom, 3) if denom else 0.0
    pct_left = 100.0 * n_left / n_vox if n_vox else 0.0
    pct_right = 100.0 * n_right / n_vox if n_vox else 0.0
    pct_mid = 100.0 * n_mid / n_vox if n_vox else 0.0
    bilateral = bool(pct_left >= 5.0 and pct_right >= 5.0)

    return {
        "n_voxels": n_vox,
        "voxel_volume_mm3": round(voxel_vol, 4),
        "volume_mm3": round(volume_mm3, 1),
        "volume_cc": round(volume_mm3 / 1000.0, 2),
        "voxel_size_mm": [round(float(z), 3) for z in zooms],
        "shape": [int(s) for s in p.shape[:3]],
        "center_of_mass_mm": [round(float(c), 1) for c in com_world],
        "bbox_size_mm": [round(float(b), 1) for b in bbox_mm],
        "bbox_size_vox": [int(b) for b in bbox_vox],
        "n_components": n_comp,
        "largest_component_voxels": largest_vox,
        "largest_component_pct": round(100.0 * largest_vox / n_vox, 1) if n_vox else 0.0,
        "component_sizes": [int(s) for s in comp_sizes[:10]],
        "pct_left": round(pct_left, 1),
        "pct_right": round(pct_right, 1),
        "pct_midline": round(pct_mid, 1),
        "laterality_index": laterality_index,
        "bilateral": bilateral,
        "header_zooms_mismatch": mismatch,
    }
