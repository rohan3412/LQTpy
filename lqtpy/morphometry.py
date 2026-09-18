"""Geometric statistics for a single lesion, computed in its native space."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def _voxel_to_world(affine, ijk):
    ijk = np.asarray(ijk, dtype=float)
    return affine[:3, :3] @ ijk + affine[:3, 3]


def compute_morphometry(lesion_img) -> dict:
    """Return a dict of morphometric measures for a binary lesion image.

    All physical measures use the image's own affine/voxel sizes, so results
    are correct regardless of the atlas grid used later for overlap.
    """
    data = np.asarray(lesion_img.get_fdata()) > 0
    affine = lesion_img.affine
    zooms = np.abs(np.asarray(lesion_img.header.get_zooms()[:3], dtype=float))
    voxel_vol = float(np.prod(zooms))  # mm^3 per voxel

    n_vox = int(data.sum())
    volume_mm3 = n_vox * voxel_vol

    # Centre of mass (voxel -> world mm).
    com_vox = ndimage.center_of_mass(data)
    com_world = _voxel_to_world(affine, com_vox)

    # Bounding box in voxels and its physical size.
    idx = np.array(np.nonzero(data))
    mins = idx.min(axis=1)
    maxs = idx.max(axis=1)
    bbox_vox = (maxs - mins + 1)
    bbox_mm = bbox_vox * zooms

    # Connected components (26-connectivity).
    structure = np.ones((3, 3, 3), dtype=int)
    labelled, n_comp = ndimage.label(data, structure=structure)
    if n_comp > 0:
        comp_sizes = np.bincount(labelled.ravel())[1:]
        comp_sizes = np.sort(comp_sizes)[::-1]
        largest_vox = int(comp_sizes[0])
    else:
        comp_sizes = np.array([], dtype=int)
        largest_vox = 0

    # Hemisphere split by world-x sign (MNI convention: x < 0 = left).
    world_x = (affine[0, 0] * idx[0] + affine[0, 1] * idx[1]
               + affine[0, 2] * idx[2] + affine[0, 3])
    n_left = int((world_x < 0).sum())
    n_right = int((world_x > 0).sum())

    # Laterality index: +1 fully left, -1 fully right, ~0 bilateral. A lesion is
    # "bilateral" when each hemisphere holds at least 5% of its voxels.
    denom = n_left + n_right
    laterality_index = round((n_left - n_right) / denom, 3) if denom else 0.0
    pct_left = 100.0 * n_left / n_vox if n_vox else 0.0
    pct_right = 100.0 * n_right / n_vox if n_vox else 0.0
    bilateral = bool(pct_left >= 5.0 and pct_right >= 5.0)

    return {
        "n_voxels": n_vox,
        "voxel_volume_mm3": round(voxel_vol, 4),
        "volume_mm3": round(volume_mm3, 1),
        "volume_cc": round(volume_mm3 / 1000.0, 2),
        "voxel_size_mm": [round(float(z), 3) for z in zooms],
        "shape": [int(s) for s in data.shape[:3]],
        "center_of_mass_mm": [round(float(c), 1) for c in com_world],
        "bbox_size_mm": [round(float(b), 1) for b in bbox_mm],
        "bbox_size_vox": [int(b) for b in bbox_vox],
        "n_components": int(n_comp),
        "largest_component_voxels": largest_vox,
        "largest_component_pct": round(100.0 * largest_vox / n_vox, 1) if n_vox else 0.0,
        "component_sizes": [int(s) for s in comp_sizes[:10]],
        "pct_left": round(pct_left, 1),
        "pct_right": round(pct_right, 1),
        "laterality_index": laterality_index,
        "bilateral": bilateral,
    }
