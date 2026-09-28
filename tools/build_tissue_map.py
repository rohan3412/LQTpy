"""Build the bundled tissue-class map (``tissue`` atlas) on the FSL MNI152 1 mm grid.

Each brain voxel gets one class, split by hemisphere where that applies:
cortex, cerebral white matter, deep grey matter, lateral ventricle, other
ventricles, cerebellum, brainstem, and sulcal/cisternal CSF.

Sources (see lqtpy/data/atlases/SOURCE.md):

- Structures with a clear anatomical label come from the bundled DKT31/CMA
  atlas: cortex (labels >= 1000, plus the hippocampus), deep grey nuclei,
  ventricles, cerebellum and brainstem. This keeps the map consistent with
  the regions the description names.
- Cerebral white matter comes from the ICBM 2009a symmetric white-matter
  probability map shipped with nilearn (Fonov et al. 2011), where it is at
  least as likely as grey matter, plus the JHU white-matter labels. Voxels
  the ICBM grey-matter map calls cortex next to DKT cortex fill the cortical
  ribbon where DKT is thin.
- Whatever is left inside the brain is labelled CSF.

The ICBM template and the FSL grid the atlases use line up to about a
millimetre (checked by overlaying the DKT cortex on the ICBM T1), which is
why the class boundaries are used with a tolerance of a couple of
millimetres in :mod:`lqtpy.tissue`.

Run: python tools/build_tissue_map.py   (needs nilearn)
"""
from __future__ import annotations

import csv
import os

import numpy as np
import nibabel as nib
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "lqtpy", "data")

# LabelID -> (RegionName, DisplayName, Hemisphere, Tissue, Group)
LABELS = {
    1: ("cortex_L", "cerebral cortex", "L", "Cortex"),
    2: ("cortex_R", "cerebral cortex", "R", "Cortex"),
    3: ("white_matter_L", "cerebral white matter", "L", "White matter"),
    4: ("white_matter_R", "cerebral white matter", "R", "White matter"),
    5: ("deep_grey_L", "deep grey matter", "L", "Deep grey matter"),
    6: ("deep_grey_R", "deep grey matter", "R", "Deep grey matter"),
    7: ("lateral_ventricle_L", "lateral ventricle", "L", "CSF"),
    8: ("lateral_ventricle_R", "lateral ventricle", "R", "CSF"),
    9: ("ventricles_3rd_4th", "third and fourth ventricles", "Midline", "CSF"),
    10: ("cerebellum_L", "cerebellum", "L", "Cerebellum"),
    11: ("cerebellum_R", "cerebellum", "R", "Cerebellum"),
    12: ("brainstem", "brainstem", "Midline", "Brainstem"),
    13: ("csf_L", "sulcal and cisternal CSF", "L", "CSF"),
    14: ("csf_R", "sulcal and cisternal CSF", "R", "CSF"),
}

DK_DEEP = [10, 11, 12, 13, 18, 26, 28, 49, 50, 51, 52, 54, 58, 60, 91, 92]
DK_HIPPO = [17, 53]
DK_LAT_VENT = [4, 5, 43, 44]
DK_MID_VENT = [14, 15]
DK_CEREB = [6, 7, 45, 46, 630, 631, 632]
DK_STEM = [16]


def build():
    import nilearn
    from nilearn.image import resample_to_img

    dk_img = nib.load(os.path.join(DATA, "atlases",
                                   "OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz"))
    d = np.asarray(dk_img.dataobj).astype(np.int32)
    jhu = np.asarray(nib.load(os.path.join(DATA, "atlases", "JHU-ICBM-labels-1mm.nii.gz")).dataobj) > 0
    tdir = os.path.join(os.path.dirname(nilearn.__file__), "datasets", "data")

    def prob(name):
        im = nib.load(os.path.join(tdir, f"mni_icbm152_{name}_tal_nlin_sym_09a_converted.nii.gz"))
        r = resample_to_img(im, dk_img, interpolation="continuous", force_resample=True,
                            copy_header=True).get_fdata()
        return np.clip(r / float(np.asarray(im.dataobj).max()), 0, 1)

    gm, wm = prob("gm"), prob("wm")

    ctx = (d >= 1000) | np.isin(d, DK_HIPPO)
    deep = np.isin(d, DK_DEEP)
    latv = np.isin(d, DK_LAT_VENT)
    midv = np.isin(d, DK_MID_VENT)
    cereb = np.isin(d, DK_CEREB)
    stem = np.isin(d, DK_STEM)
    infra = cereb | stem
    # Cerebral white matter: ICBM WM at least as likely as GM, or a JHU tract;
    # never where DKT names a structure.
    named = ctx | deep | latv | midv | infra
    wmm = (((wm >= 0.5) & (wm >= gm)) | jhu) & ~named
    # Keep only white matter connected to the main cerebral body (drops
    # specks of WM probability outside the brain).
    lab, n = ndimage.label(wmm)
    if n > 1:
        sizes = ndimage.sum(wmm, lab, range(1, n + 1))
        wmm = np.isin(lab, 1 + np.flatnonzero(sizes >= 0.01 * sizes.max()))
    # Thicken the cortical ribbon where DKT is thin: ICBM GM voxels next to
    # DKT cortex that aren't white matter or another named structure.
    near_ctx = ndimage.binary_dilation(ctx, iterations=2)
    ctx = ctx | ((gm >= 0.5) & (gm > wm) & near_ctx & ~named & ~wmm)
    # CSF: the rest of the (filled) brain.
    brain = ctx | deep | latv | midv | infra | wmm
    filled = ndimage.binary_fill_holes(brain)
    for z in range(filled.shape[2]):
        filled[:, :, z] = ndimage.binary_fill_holes(filled[:, :, z])
    csf = filled & ~brain

    aff = dk_img.affine
    i = np.arange(d.shape[0])
    x = aff[0, 0] * i + aff[0, 3]
    left = np.broadcast_to((x < 0)[:, None, None], d.shape)

    out = np.zeros(d.shape, np.uint8)

    def put(mask, lid_l, lid_r=None):
        if lid_r is None:
            out[mask] = lid_l
        else:
            out[mask & left] = lid_l
            out[mask & ~left] = lid_r

    put(csf, 13, 14)
    put(wmm, 3, 4)
    put(ctx, 1, 2)
    put(deep, 5, 6)
    put(cereb, 10, 11)
    put(stem, 12)
    put(midv, 9)
    put(latv, 7, 8)
    # Hemisphere of DKT-labelled structures follows DKT, not world x.
    for lid_l, lid_r, mask in ((1, 2, ctx), (5, 6, deep), (7, 8, latv), (10, 11, cereb)):
        rn = np.zeros(d.shape, bool)
        for lab_id in np.unique(d[mask]):
            row = _dk_name(int(lab_id))
            if row.startswith("right_"):
                rn |= d == lab_id
        sel = mask & (d > 0)
        out[sel & rn] = lid_r
        out[sel & ~rn & _dk_left(d)] = lid_l

    img = nib.Nifti1Image(out, aff)
    img.header.set_data_dtype(np.uint8)
    img.header["descrip"] = b"LQTpy tissue classes (DKT31/CMA + ICBM2009a WM)"
    nib.save(img, os.path.join(DATA, "atlases", "tissue_MNI152_1mm.nii.gz"))

    with open(os.path.join(DATA, "tissue_labels.csv"), "w", newline="", encoding="utf-8") as fh:
        fh.write("# Tissue classes on the FSL MNI152 1 mm grid, built by tools/build_tissue_map.py "
                 "from the DKT31/CMA atlas and the ICBM 2009a white-matter probability map.\n")
        w = csv.writer(fh)
        w.writerow(["LabelID", "RegionName", "Group", "Lobe", "DisplayName", "Hemisphere",
                    "Tissue", "Function", "Territory"])
        for lid, (name, disp, hemi, tissue) in LABELS.items():
            w.writerow([lid, name, tissue, "", disp, hemi, tissue, "", ""])
    for lid, (name, *_r) in LABELS.items():
        print(f"{lid:3d} {name:22s} {int((out == lid).sum()) / 1000:8.1f} cm3")


_DK_NAMES = None


def _dk_name(lid: int) -> str:
    global _DK_NAMES
    if _DK_NAMES is None:
        _DK_NAMES = {}
        with open(os.path.join(DATA, "dkt31_cma_labels.csv"), encoding="utf-8") as fh:
            rows = [r for r in fh if not r.startswith("#")]
        for row in csv.DictReader(rows):
            _DK_NAMES[int(row["LabelID"])] = row["RegionName"]
    return _DK_NAMES.get(lid, "")


def _dk_left(d):
    ids = [k for k in np.unique(d) if _dk_name(int(k)).startswith("left_")]
    return np.isin(d, ids)


if __name__ == "__main__":
    build()
