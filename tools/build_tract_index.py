"""Precompute HCP842 tract voxel indices into a compact committed .npz.

Reading the 4D tract nifti one volume at a time via ``dataobj[..., i]`` re-
decompresses the whole gzip stream per slice (~130 s). Instead we load the full
4D array ONCE, extract each tract's flat voxel indices, and store them (plus the
reference shape/affine and tract names) so ``lqtpy/tracts.py`` can load instantly.

Run once (build-time): python tools/build_tract_index.py
"""
import os
import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
NII = os.path.join(ROOT, "extdata", "dsi_studio_64", "atlas", "HCP1065",
                   "HCP842_tractography.nii.gz")
TXT = os.path.join(ROOT, "extdata", "dsi_studio_64", "atlas", "HCP1065",
                   "HCP842_tractography.txt")
OUT = os.path.join(ROOT, "lqtpy", "data", "hcp842_tracts.npz")

names = []
with open(TXT) as fh:
    for line in fh:
        parts = line.split(None, 1)
        if len(parts) == 2:
            names.append(parts[1].strip())

img = nib.load(NII)
print("loading full 4D array once:", img.shape)
data = np.asarray(img.dataobj)          # single decompression, native dtype
print("loaded", data.dtype, round(data.nbytes / 1e9, 2), "GB")

save = {"shape": np.array(img.shape[:3], dtype=np.int32),
        "affine": img.affine.astype(np.float64),
        "names": np.array(names, dtype=object)}
kept = []
for i, name in enumerate(names):
    idx = np.flatnonzero((data[..., i] > 0).ravel()).astype(np.int32)
    if idx.size:
        save[f"idx_{i}"] = idx
        kept.append((name, int(idx.size)))

np.savez_compressed(OUT, **save)
print("wrote", OUT, round(os.path.getsize(OUT) / 1e6, 2), "MB",
      "tracts:", len(kept))
