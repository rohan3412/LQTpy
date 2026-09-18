# Atlas provenance

This directory bundles five files so that `pip install lqtpy` works offline. Each is
redistributed here under its own academic/citation terms (see below) — none of these
carry patient data; they are standard neuroimaging reference atlases.

The white-matter tract voxel-overlap proxy index (`lqtpy/data/hcp842_tracts.npz`, used
by `lqtpy.tracts`/`tract_proxy_overlap`, not an atlas file so not listed above but
documented here for the same reason) and the `dk` atlas below were both re-sourced as of
2026-09-18 after their prior CC BY-NC-SA-encumbered
sources were replaced with cleanly-licensed ones.

---

## `aal.nii.gz`

- **Full name:** AAL — Automated Anatomical Labeling atlas (116-region cortical/subcortical
  parcellation), MNI space.
- **Origin:** Copied from this project's `parcellation/aal.nii.gz`.
- **Version/date:** Not determinable from the source tree. (Filesystem timestamp on the
  source file is 2014-03-15, but that reflects when the file was staged locally, not a
  confirmed release/version tag of the atlas itself.)
- **Licence/attribution:** Free for academic/research use with citation. Not licensed for
  commercial redistribution without permission from the original authors.
- **Citation:** Tzourio-Mazoyer N, Landeau B, Papathanassiou D, Crivello F, Etard O,
  Delcroix N, Mazoyer B, Joliot M. "Automated Anatomical Labeling of activations in SPM
  using a Macroscopic Anatomical Parcellation of the MNI MRI single-subject brain."
  *NeuroImage* 15(1):273-289, 2002.

---

## `Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.nii.gz` and
## `Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.csv`

- **Full name:** Schaefer 2018 cortical parcellation, 100 parcels, 7 Yeo intrinsic
  connectivity networks, FSL MNI152 space, resliced to 1 mm.
- **Origin:** Copied from this project's `parcellation/Schaefer2018_100Parcels_7Networks_
  order_FSLMNI152_1mm_resliced.{nii.gz,csv}`. A non-resliced 1mm and a 2mm variant also
  exist in the source `parcellation/` directory but were not staged (the resliced 1mm pair
  is what `lqtpy/atlases.py` references).
- **Version/date:** "2018" per the atlas's own naming (Schaefer2018). No finer version/date
  is determinable from the source tree.
- **Licence/attribution:** Free for academic/research use with citation. See the Schaefer
  parcellation project (Thomas Yeo Lab, CBIG) for full terms.
- **Citation:** Schaefer A, Kong R, Gordon EM, Laumann TO, Zuo XN, Holmes AJ, Eickhoff SB,
  Yeo BTT. "Local-Global Parcellation of the Human Cerebral Cortex from Intrinsic
  Functional Connectivity MRI." *Cerebral Cortex* 28(9):3095-3114, 2018.
  Network labels: Yeo BTT et al. "The organization of the human cerebral cortex estimated
  by intrinsic functional connectivity." *J Neurophysiol* 106(3):1125-1165, 2011.

---

## `OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz` and `../dkt31_cma_labels.csv`

- **Full name:** Mindboggle-101 joint-fusion volume atlas: DKT31 cortical parcellation (31
  regions/hemisphere, the Desikan-Killiany-Tourville protocol) plus CMA (Center for
  Morphometric Analysis) subcortical/ventricle/cerebellar structures, built by joint-fusion
  label combination (ANTs, Wang 2013) over 20 manually-labeled OASIS-TRT brains, resampled
  to MNI152 space at 1 mm. 96 total labelled regions (62 cortical + 34 non-cortical).
- **Origin:** Downloaded 2026-09-18 from Zenodo record **10.5281/zenodo.22070005**
  ("Mindboggle-101: 101 human brain MRI volumes and surfaces, derived templates, and
  anatomically labeled atlases", Arno Klein), file
  `Mindboggle101_atlases.zip` → `jointfusion_volume_atlases/OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz`
  (sha256 `73ff2bc35e9309fde34521e4e60a0fffa494a8e283ac209613af25023797626a`, 342,689 bytes).
  This record explicitly separates this joint-fusion **population atlas** (this file) from
  a companion record holding the raw manually-labeled **individual** OASIS-TRT-20 subcortical
  volumes, which carries a more restrictive CC BY-NC-ND 4.0 licence
  (10.5281/zenodo.22071825) — that companion data is **not** used here. A 2 mm variant
  (`..._MNI152_2mm_v2.nii.gz`, same record) also exists; the 1 mm variant was chosen for
  resolution consistency with the other two bundled atlases (`aal.nii.gz` and the Schaefer
  volume are both effectively 1 mm, 181×217×181/182×218×182-class grids, despite AAL's
  original template historically being distributed at 2 mm elsewhere).
  This replaces the previous `FreeSurferDKT.nii.gz`/`.txt` pair (sourced via a vendored DSI
  Studio distribution, CC BY-NC-SA 2.0 via DSI Studio's blanket licence) — see
  a re-sourcing done for licence reasons. `aal` is now the default/first atlas (`lqtpy/atlases.py`).
- **Version/date:** Mindboggle-101 release published 2016-08-09, "release 3" data updated
  2019-04-03 per the Zenodo record's README.
- **Licence/attribution:** **CC BY 4.0** (Creative Commons Attribution 4.0 International),
  as stated on the Zenodo record (`license: cc-by-4.0` in the record's own metadata, and
  its README: "License: CC BY 4.0. See LICENSE."). Redistribution and reuse are permitted
  with attribution.
- **Citation:** Klein A, Tourville J. "101 labeled brain images and a consistent human
  cortical labeling protocol." *Frontiers in Neuroscience* 6:171, 2012.
  DOI: 10.3389/fnins.2012.00171 (verified against PubMed/Frontiers 2026-09-18; Zenodo's own
  metadata blurb names the Frontiers in Neuroscience specialty section "Brain Imaging
  Methods", not a separate journal).
- **`dkt31_cma_labels.csv` (label id → RegionName, `lqtpy/data/`, not under `atlases/`,
  matching `aal_labels.csv`'s convention):** authored for this project, **not** a file from
  the Mindboggle download. Built by mapping every non-zero label id actually present in the
  volume above through the **FreeSurferColorLUT.txt** id→name table (the standard,
  ubiquitously-redistributed FreeSurfer label lookup table; fetched 2026-09-18 from
  `github.com/freesurfer/freesurfer` `distribution/FreeSurferColorLUT.txt`), which is also
  exactly what generates these ids: the atlas build script
  (`Mindboggle101_atlases/code/make_jointfusion_atlas.txt` in the Zenodo download) calls
  `mindboggle.mio.labels.DKTprotocol`, whose `return_numbers_names_colors()` itself parses
  `FreeSurferColorLUT.txt`. Five ids the CMA protocol reassigned away from the LUT's literal
  text are overridden per `mindboggle/mindboggle/mio/labels.py`'s own documented exceptions
  (`nipy/mindboggle`, fetched 2026-09-18): **91/92** ("left/right basal forebrain" — the LUT
  has no entry for these two ids at all) and **630/631/632** (the LUT's literal text is
  generic `Cbm_Vermis_*`; the CMA protocol repurposes these three ids specifically for
  "cerebellar vermal lobules I-V / VI-VII / VIII-X"). All 96 ids resolved; none were
  hand-invented. RegionName format matches the pre-existing `dk` convention (`left_`/`right_`
  prefix + squished region token, e.g. `left_fusiform`, `left_basal_forebrain`), which keeps
  `lqtpy/atlases.py`'s `_fs_to_ggseg` ggseg-key mapping working unchanged. `Group` is
  `Left`/`Right`/`Midline` by name prefix, matching `aal`'s convention.
  **ggseg flat-map coverage** (`lqtpy/data/ggseg_dk.csv.gz`, derived from the R `ggseg`
  package's `dk()` atlas — the classic 34/35-region-per-hemisphere Desikan-Killiany, not
  DKT31): all 62 of this atlas's cortical region keys match a ggseg polygon exactly (100%
  coverage on our side). ggseg has 8 extra per-hemisphere keys with no data here —
  `bankssts`, `corpuscallosum`, `frontalpole`, `temporalpole` (×2 hemispheres) — because the
  DKT protocol deliberately dropped those 4 regions from full DK (Klein & Tourville 2012);
  they render as "no data" grey in the flat map rather than breaking it.

---

## `hcp842_tracts.npz` (not an atlas file, but documented here for the same reason)

- **Full name:** Per-tract voxel-overlap proxy index for `lqtpy.tracts`/
  `tract_proxy_overlap`, precomputed by `tools/build_tract_index.py` from the HCP-842
  population tractography atlas's 4D tract-mask volume (`HCP842_tractography.nii.gz`/`.txt`).
- **True origin:** the HCP-842 population-averaged tractography atlas is Frank Yeh's own
  dataset, published at https://brain.labsolver.org/hcp_trk_atlas.html. A vendored DSI
  Studio distribution also ships a copy of the same files (which is the copy
  `tools/build_tract_index.py` was originally pointed at) — DSI Studio is a redistribution
  vector for this data, **not** its origin.
- **Licence/attribution:** **CC BY-SA 4.0** (Creative Commons Attribution-ShareAlike 4.0
  International), per the licence statement on the labsolver.org page above (verified
  2026-09-18). This is **separate from and unrelated to** whatever licence `lqtpy`'s own
  code takes (MIT; see `DATA-LICENSES.md`) — ShareAlike attaches to this
  specific derived data file, not automatically to the surrounding codebase. Neither of the
  authors of this note is a lawyer; if redistributing this file matters for a specific use
  case, get your own legal read on what "ShareAlike" requires for a derived voxel-index
  `.npz` versus the original tractography volume.
- **Citation:** Yeh FC, Panesar S, Fernandes D, Meola A, Yoshino M, Fernandez-Miranda JC,
  Vettel JM, Verstynen T. "Population-averaged atlas of the macroscale human structural
  connectome and its network topology." *NeuroImage* 178:57-68, 2018.
  DOI: 10.1016/j.neuroimage.2018.05.027.
- **Additional acknowledgment required by the source:** the underlying tractography atlas
  was built from Human Connectome Project data; per HCP's Open Access Data Use Terms
  (https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms),
  any use of data derived from the HCP must acknowledge: *"Data were provided [in part] by
  the Human Connectome Project, WU-Minn Consortium (Principal Investigators: David Van Essen
  and Kamil Ugurbil; 1U54MH091657) funded by the 16 NIH Institutes and Centers that support
  the NIH Blueprint for Neuroscience Research; and by the McDonnell Center for Systems
  Neuroscience at Washington University."*

---

## Derived flat-map geometry (`lqtpy/data/ggseg_dk.csv.gz`, `ggseg_schaefer100.csv.gz`)

Not atlas volumes, but checked for the same reason: both are
exported (`tools/export_ggseg_geometry.R`) from R packages **`ggseg`** (core, `dk()` atlas)
and **`ggsegSchaefer`** (`schaefer7_100()`). Both packages declare `License: MIT + file
LICENSE` in their `DESCRIPTION` (verified 2026-09-18 against CRAN's `ggseg` listing and both
packages' GitHub `DESCRIPTION`/`LICENSE` files, `LCBC-UiO/ggseg` and
`ggseg/ggsegSchaefer`) — permissive, not GPL-family, and does not restrict redistributing
derived polygon-coordinate data with attribution. This is **not** a publish blocker.
