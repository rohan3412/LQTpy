"""Runnable quickstart: build a synthetic lesion in memory (no data files, no
patient data) and run lqtpy's core numeric API on it.

Mirrors the quickstart in README.md exactly -- if you edit one, edit both.
Uses only the core install (numpy, scipy, nibabel); no plotting/report extras
are imported.
"""
from __future__ import annotations

import numpy as np

import lqtpy

# An MNI152-like 1mm grid (182 x 218 x 182) and affine (FSL convention: -x
# increases to the left). A synthetic spherical "lesion" centered in the
# right frontal lobe -- no patient data anywhere.
shape = (182, 218, 182)
affine = np.array([
    [-1, 0, 0, 90],
    [0, 1, 0, -126],
    [0, 0, 1, -72],
    [0, 0, 0, 1],
], dtype=float)

grid = np.indices(shape) - np.array([56, 130, 100])[:, None, None, None]
lesion_mask = ((grid ** 2).sum(0) < 12 ** 2).astype(np.uint8)
lesion = (lesion_mask, affine)  # (array, affine) tuple -- no file on disk needed

stats = lqtpy.lesion_stats(lesion)
print("Volume (cm3):", stats["volume_cc"])
print("Voxels:", stats["n_voxels"])
print("Laterality index:", stats["laterality_index"])

overlap = lqtpy.atlas_overlap(lesion, "aal")
print("\nTop AAL regions by % damaged:")
for row in overlap[:3]:
    print(f"  {row['RegionName']:<20} {row['PercentDamage']:5.1f}% damaged, "
          f"{row['PercentOfLesion']:5.1f}% of lesion")
