"""Stage the vascular-territory and white-matter atlases into lqtpy/data.

Sources (see lqtpy/data/atlases/SOURCE.md for citations/licences):

- Johns Hopkins arterial territory atlas (Liu et al. 2023, Sci Data 10:74;
  CC BY-SA 4.0): github.com/Chin-Fu-Liu/Arterial_Atlas,
  data/Atlas_182/ArterialAtlas.nii + data/ArterialAtlasLables.txt.
- JHU ICBM-DTI-81 white-matter labels (Mori et al. 2008; Oishi et al.):
  neurovault.org/media/images/264/JHU-ICBM-labels-1mm.nii.gz, label names
  from FSL's JHU-labels.xml.

THE ARTERIAL ATLAS HEADER HAS NO ORIGIN. Both released grids store a zero
translation (sform/qform origin 0,0,0), which would place the atlas ~120 mm
from the brain. Its 182x218x182 grid is FSL's MNI152 1 mm grid, and
registering it (translation only) to the bundled brain mask and to the DKT
lateral ventricles gives FSL's affine plus a small shift: +2 mm in y and
+1 mm in z (ventricle Dice 0.703 vs 0.657 unshifted; whole-brain Dice
0.947 vs 0.936). That corrected affine is written into the bundled file;
tests/test_atlas_schema.py pins it.

Run: python tools/build_extra_atlases.py <download dir>
"""
from __future__ import annotations

import csv
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import nibabel as nib

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "lqtpy", "data")

FSL_AFFINE = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])
ARTERIAL_SHIFT_MM = np.array([0.0, 2.0, 1.0])  # see module docstring

# Level-1 id -> (display name, territory, deep?)
ARTERIAL = {
    1: ("anterior cerebral artery territory", "ACA", False),
    3: ("medial lenticulostriate territory", "ACA", True),
    5: ("lateral lenticulostriate territory", "MCA", True),
    7: ("middle cerebral artery territory, frontal part", "MCA", False),
    9: ("middle cerebral artery territory, parietal part", "MCA", False),
    11: ("middle cerebral artery territory, temporal part", "MCA", False),
    13: ("middle cerebral artery territory, occipital part", "MCA", False),
    15: ("middle cerebral artery territory, insular part", "MCA", False),
    17: ("posterior cerebral artery territory, temporal part", "PCA", False),
    19: ("posterior cerebral artery territory, occipital part", "PCA", False),
    21: ("posterior choroidal and thalamoperforator territory", "PCA", True),
    23: ("anterior choroidal and thalamoperforator territory", "PCA", True),
    25: ("basilar artery territory", "VB", True),
    27: ("superior cerebellar artery territory", "VB", False),
    29: ("inferior cerebellar arteries territory", "VB", False),
    31: ("lateral ventricle", "Ventricle", False),
}


def build_arterial(src_dir):
    img = nib.load(os.path.join(src_dir, "ArterialAtlas.nii"))
    data = np.asanyarray(img.dataobj)
    if data.ndim == 4:
        data = data[..., 0]
    data = data.astype(np.int8)
    assert data.shape == (182, 218, 182)
    aff = FSL_AFFINE.copy()
    aff[:3, 3] += ARTERIAL_SHIFT_MM
    out = nib.Nifti1Image(data, aff)
    out.set_sform(aff, code=4)  # MNI152
    out.set_qform(aff, code=4)
    path = os.path.join(DATA, "atlases", "ArterialAtlas_level1_MNI152_1mm.nii.gz")
    nib.save(out, path)
    rows = []
    for base, (name, terr, deep) in ARTERIAL.items():
        for lid, side in ((base, "L"), (base + 1, "R")):
            code = f"{terr}_{'deep' if deep else 'cortical'}" if terr != "Ventricle" else "Ventricle"
            rows.append({"LabelID": lid,
                         "RegionName": f"{name.replace(' ', '_').replace(',', '')}_{side}",
                         "Group": terr, "Territory": terr,
                         "DisplayName": name, "Hemisphere": side,
                         "Tissue": "CSF" if terr == "Ventricle" else "",
                         "Function": "", "Lobe": "",
                         "Supply": "deep" if deep else ("" if terr == "Ventricle" else "cortical"),
                         "Code": code})
    _write(os.path.join(DATA, "arterial_labels.csv"), rows,
           "# Johns Hopkins arterial territory atlas, level 1 (Liu et al. 2023, Sci Data "
           "10:74; CC BY-SA 4.0). Group/Territory = major territory (ACA/MCA/PCA/VB); "
           "Supply = deep (perforator) or cortical (branch) territory. Built by "
           "tools/build_extra_atlases.py.")
    print("wrote", path, int(data.max()), "labels")


def build_jhu(src_dir):
    img = nib.load(os.path.join(src_dir, "JHU-ICBM-labels-1mm.nii.gz"))
    assert np.allclose(img.affine, FSL_AFFINE)
    data = np.asanyarray(img.dataobj).astype(np.int8)
    out = nib.Nifti1Image(data, FSL_AFFINE)
    out.set_sform(FSL_AFFINE, code=4)
    out.set_qform(FSL_AFFINE, code=4)
    path = os.path.join(DATA, "atlases", "JHU-ICBM-labels-1mm.nii.gz")
    nib.save(out, path)
    tree = ET.parse(os.path.join(src_dir, "JHU-labels.xml"))
    rows = []
    for lab in tree.getroot().iter("label"):
        idx = int(lab.get("index"))
        if idx == 0:
            continue
        name = " ".join(lab.text.split())
        side = ""
        if name.endswith(" R"):
            side, name = "R", name[:-2]
        elif name.endswith(" L"):
            side, name = "L", name[:-2]
        disp = name[0].lower() + name[1:]
        for a, b in (("longitidinal", "longitudinal"),
                     (" (can not be resolved with current resolution)", ""),
                     (" of internal capsule", " of the internal capsule"),
                     (" of corpus callosum", " of the corpus callosum"),
                     (" of MCP", " of the MCP")):
            disp = disp.replace(a, b)
        rows.append({"LabelID": idx,
                     "RegionName": name.replace(" ", "_") + (f"_{side}" if side else ""),
                     "Group": {"L": "Left", "R": "Right"}.get(side, "Midline"),
                     "Territory": "", "DisplayName": disp,
                     "Hemisphere": side or "Midline", "Tissue": "White matter",
                     "Function": "", "Lobe": "White matter", "Supply": "", "Code": ""})
    _write(os.path.join(DATA, "jhu_labels.csv"), rows,
           "# JHU ICBM-DTI-81 white-matter labels (Mori et al. 2008). Names from FSL "
           "JHU-labels.xml. Built by tools/build_extra_atlases.py.")
    print("wrote", path, len(rows), "labels")


def _write(path, rows, comment):
    cols = ["LabelID", "RegionName", "Group", "Lobe", "DisplayName", "Hemisphere", "Tissue",
            "Function", "Territory", "Supply", "Code"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(comment + "\n")
        w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    src = sys.argv[1]
    build_arterial(src)
    build_jhu(src)
