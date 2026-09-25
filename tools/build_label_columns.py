"""Add the schema-v2 columns (DisplayName, Tissue, Function) to the bundled
AAL and DKT31+CMA label tables.

The curated vocabulary lives here, in one reviewable place; the CSVs are
regenerated from it. Re-running is idempotent: existing v2 columns are
replaced, and the ``#`` provenance comment block at the top of each CSV is
kept (with a note on the new columns appended once).

Function labels are deliberately conservative: only well-established
functional names that apply to the *whole* labelled structure (primary
motor/somatosensory/visual/auditory cortex, Broca's area on the left).
Wernicke's area is left out on purpose: it is the posterior part of the
superior temporal gyrus, and neither AAL nor DKT splits that gyrus, so a
lesion in its anterior part would be mislabelled. They're location
facts ("involves the primary motor cortex"), never deficit predictions.

Run: python tools/build_label_columns.py
"""
from __future__ import annotations

import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "lqtpy", "data")

NOTE = ("# Schema v2 columns (tools/build_label_columns.py): DisplayName = readable "
        "name without hemisphere; Tissue = Cortex / Deep grey matter / Cerebellum / "
        "White matter / CSF / Brainstem / Other; Function = conservative common "
        "functional name, blank when none.")

# --------------------------------------------------------------------- AAL
AAL_NAMES = {
    "Precentral": "precentral gyrus",
    "Frontal_Sup": "superior frontal gyrus (dorsolateral)",
    "Frontal_Sup_Orb": "superior frontal gyrus, orbital part",
    "Frontal_Mid": "middle frontal gyrus",
    "Frontal_Mid_Orb": "middle frontal gyrus, orbital part",
    "Frontal_Inf_Oper": "inferior frontal gyrus, opercular part",
    "Frontal_Inf_Tri": "inferior frontal gyrus, triangular part",
    "Frontal_Inf_Orb": "inferior frontal gyrus, orbital part",
    "Rolandic_Oper": "Rolandic operculum",
    "Supp_Motor_Area": "supplementary motor area",
    "Olfactory": "olfactory cortex",
    "Frontal_Sup_Medial": "superior frontal gyrus, medial part",
    "Frontal_Med_Orb": "medial orbitofrontal cortex",
    "Rectus": "gyrus rectus",
    "Insula": "insula",
    "Cingulum_Ant": "anterior cingulate gyrus",
    "Cingulum_Mid": "middle cingulate gyrus",
    "Cingulum_Post": "posterior cingulate gyrus",
    "Hippocampus": "hippocampus",
    "ParaHippocampal": "parahippocampal gyrus",
    "Amygdala": "amygdala",
    "Calcarine": "calcarine cortex",
    "Cuneus": "cuneus",
    "Lingual": "lingual gyrus",
    "Occipital_Sup": "superior occipital gyrus",
    "Occipital_Mid": "middle occipital gyrus",
    "Occipital_Inf": "inferior occipital gyrus",
    "Fusiform": "fusiform gyrus",
    "Postcentral": "postcentral gyrus",
    "Parietal_Sup": "superior parietal lobule",
    "Parietal_Inf": "inferior parietal lobule",
    "SupraMarginal": "supramarginal gyrus",
    "Angular": "angular gyrus",
    "Precuneus": "precuneus",
    "Paracentral_Lobule": "paracentral lobule",
    "Caudate": "caudate nucleus",
    "Putamen": "putamen",
    "Pallidum": "globus pallidus",
    "Thalamus": "thalamus",
    "Heschl": "Heschl's gyrus",
    "Temporal_Sup": "superior temporal gyrus",
    "Temporal_Pole_Sup": "temporal pole (superior part)",
    "Temporal_Mid": "middle temporal gyrus",
    "Temporal_Pole_Mid": "temporal pole (middle part)",
    "Temporal_Inf": "inferior temporal gyrus",
    "Cerebelum_Crus1": "cerebellar crus I",
    "Cerebelum_Crus2": "cerebellar crus II",
    "Cerebelum_3": "cerebellar lobule III",
    "Cerebelum_4_5": "cerebellar lobules IV-V",
    "Cerebelum_6": "cerebellar lobule VI",
    "Cerebelum_7b": "cerebellar lobule VIIb",
    "Cerebelum_8": "cerebellar lobule VIII",
    "Cerebelum_9": "cerebellar lobule IX",
    "Cerebelum_10": "cerebellar lobule X (flocculonodular)",
    "Vermis_1_2": "cerebellar vermis, lobules I-II",
    "Vermis_3": "cerebellar vermis, lobule III",
    "Vermis_4_5": "cerebellar vermis, lobules IV-V",
    "Vermis_6": "cerebellar vermis, lobule VI",
    "Vermis_7": "cerebellar vermis, lobule VII",
    "Vermis_8": "cerebellar vermis, lobule VIII",
    "Vermis_9": "cerebellar vermis, lobule IX",
    "Vermis_10": "cerebellar vermis, lobule X",
}
AAL_DEEP = {"Caudate", "Putamen", "Pallidum", "Thalamus", "Amygdala"}
AAL_FUNCTION = {
    "Precentral": "primary motor cortex",
    "Postcentral": "primary somatosensory cortex",
    "Paracentral_Lobule": "leg motor and sensory area",
    "Calcarine": "primary visual cortex",
    "Heschl": "primary auditory cortex",
    ("Frontal_Inf_Oper", "L"): "Broca's area",
    ("Frontal_Inf_Tri", "L"): "Broca's area",
}


def _aal_row(name):
    hemi = name[-1] if name.endswith(("_L", "_R")) else ""
    base = name[:-2] if hemi else name
    disp = AAL_NAMES[base]
    if base.startswith(("Cerebelum", "Vermis")):
        tissue = "Cerebellum"
    elif base in AAL_DEEP:
        tissue = "Deep grey matter"
    else:
        tissue = "Cortex"
    func = AAL_FUNCTION.get((base, hemi)) or AAL_FUNCTION.get(base, "")
    return disp, tissue, func


# --------------------------------------------------------------------- DKT31 + CMA
DK_CORTEX = {
    "caudalanteriorcingulate": "caudal anterior cingulate cortex",
    "caudalmiddlefrontal": "caudal middle frontal gyrus",
    "cuneus": "cuneus",
    "entorhinal": "entorhinal cortex",
    "fusiform": "fusiform gyrus",
    "inferiorparietal": "inferior parietal cortex",
    "inferiortemporal": "inferior temporal gyrus",
    "isthmuscingulate": "isthmus of the cingulate gyrus",
    "lateraloccipital": "lateral occipital cortex",
    "lateralorbitofrontal": "lateral orbitofrontal cortex",
    "lingual": "lingual gyrus",
    "medialorbitofrontal": "medial orbitofrontal cortex",
    "middletemporal": "middle temporal gyrus",
    "parahippocampal": "parahippocampal gyrus",
    "paracentral": "paracentral lobule",
    "parsopercularis": "inferior frontal gyrus, pars opercularis",
    "parsorbitalis": "inferior frontal gyrus, pars orbitalis",
    "parstriangularis": "inferior frontal gyrus, pars triangularis",
    "pericalcarine": "pericalcarine cortex",
    "postcentral": "postcentral gyrus",
    "posteriorcingulate": "posterior cingulate cortex",
    "precentral": "precentral gyrus",
    "precuneus": "precuneus",
    "rostralanteriorcingulate": "rostral anterior cingulate cortex",
    "rostralmiddlefrontal": "rostral middle frontal gyrus",
    "superiorfrontal": "superior frontal gyrus",
    "superiorparietal": "superior parietal lobule",
    "superiortemporal": "superior temporal gyrus",
    "supramarginal": "supramarginal gyrus",
    "transversetemporal": "transverse temporal (Heschl's) gyrus",
    "insula": "insula",
}
DK_FUNCTION = {
    "precentral": "primary motor cortex",
    "postcentral": "primary somatosensory cortex",
    "paracentral": "leg motor and sensory area",
    "pericalcarine": "primary visual cortex",
    "transversetemporal": "primary auditory cortex",
    ("parsopercularis", "left"): "Broca's area",
    ("parstriangularis", "left"): "Broca's area",
}
DK_OTHER = {  # base name -> (display, tissue)
    "lateral_ventricle": ("lateral ventricle", "CSF"),
    "inf_lat_vent": ("inferior horn of the lateral ventricle", "CSF"),
    "cerebellum_exterior": ("cerebellar cortex", "Cerebellum"),
    "cerebellum_white_matter": ("cerebellar white matter", "White matter"),
    "thalamus": ("thalamus", "Deep grey matter"),
    "caudate": ("caudate nucleus", "Deep grey matter"),
    "putamen": ("putamen", "Deep grey matter"),
    "pallidum": ("globus pallidus", "Deep grey matter"),
    "3rd_ventricle": ("third ventricle", "CSF"),
    "4th_ventricle": ("fourth ventricle", "CSF"),
    "brain_stem": ("brainstem", "Brainstem"),
    "hippocampus": ("hippocampus", "Cortex"),
    "amygdala": ("amygdala", "Deep grey matter"),
    "csf": ("extra-cerebral CSF", "CSF"),
    "accumbens_area": ("nucleus accumbens", "Deep grey matter"),
    "ventraldc": ("ventral diencephalon", "Deep grey matter"),
    "vessel": ("vessel", "Other"),
    "basal_forebrain": ("basal forebrain", "Deep grey matter"),
    "cerebellar_vermal_lobules_i_v": ("cerebellar vermis, lobules I-V", "Cerebellum"),
    "cerebellar_vermal_lobules_vi_vii": ("cerebellar vermis, lobules VI-VII", "Cerebellum"),
    "cerebellar_vermal_lobules_viii_x": ("cerebellar vermis, lobules VIII-X", "Cerebellum"),
}


def _dk_row(name):
    side = ""
    base = name
    for s in ("left", "right"):
        if name.startswith(s + "_"):
            side, base = s, name[len(s) + 1:]
    if base in DK_CORTEX:
        func = DK_FUNCTION.get((base, side)) or DK_FUNCTION.get(base, "")
        return DK_CORTEX[base], "Cortex", func
    disp, tissue = DK_OTHER[base]
    return disp, tissue, ""


def _rewrite(path, row_fn):
    with open(path, encoding="utf-8", newline="") as fh:
        lines = fh.read().splitlines()
    comments = [ln for ln in lines if ln.lstrip().startswith("#")]
    body = [ln for ln in lines if not ln.lstrip().startswith("#")]
    rows = list(csv.DictReader(io.StringIO("\n".join(body))))
    base_cols = [c for c in rows[0].keys() if c not in ("DisplayName", "Tissue", "Function")]
    if NOTE not in comments:
        comments.append(NOTE)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=base_cols + ["DisplayName", "Tissue", "Function"],
                       lineterminator="\n")
    w.writeheader()
    for r in rows:
        disp, tissue, func = row_fn(r["RegionName"])
        w.writerow({**{c: r[c] for c in base_cols},
                    "DisplayName": disp, "Tissue": tissue, "Function": func})
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(comments) + "\n" + out.getvalue())
    print(f"wrote {path}: {len(rows)} rows")


if __name__ == "__main__":
    _rewrite(os.path.join(DATA, "aal_labels.csv"), _aal_row)
    _rewrite(os.path.join(DATA, "dkt31_cma_labels.csv"), _dk_row)
