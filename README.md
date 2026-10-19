<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="LQTpy" width="720">
</picture>

`lqtpy` measures what a brain lesion has damaged, and says so in plain language. Give it a
lesion mask in MNI space and it returns lesion morphometry, how much of each atlas region is
destroyed, what fraction of each white-matter tract is disconnected, quality flags for badly
drawn masks, and a written description — as records you can use directly, or as a
self-contained HTML report.

It is a lean reimplementation of the ideas behind the Lesion Quantification Toolkit (Griffis
et al. 2021), written from the method rather than ported from its code. Core needs only
numpy, scipy and nibabel; importing it pulls in no plotting stack.

What it does **not** do: lesion segmentation, registration to MNI (bring your own MNI-space
mask), retinotopy, or the original method's parcel-to-parcel and shortest-path measures.

## Install

```bash
pip install .              # core: numpy, scipy, nibabel -- reports included
pip install ".[report]"    # + the scroll-through slice viewer (nilearn) and subcortical map (ggseg)
pip install ".[formats]"   # + DICOM-SEG input (highdicom); NRRD and MGZ need nothing
pip install ".[all]"       # + pandas frames, merged PDF bundles
```

Atlases ship inside the package, so everything except tract disconnection and the optional
cerebellar atlas works offline. Template maps (distance maps, border zones) are built once
per installation and cached (`lqtpy cache` shows where).

## Usage

```python
import lqtpy

les = lqtpy.prepare("lesion_MNI.nii.gz")   # load and binarize once, reuse everywhere

print(lqtpy.describe(les))
# A single 15.6 cm³ lesion in the left hemisphere.
#
# It involves the left frontal lobe (23%), insula (16%), deep grey matter (2%) and parietal
# lobe (2%). ... In the left frontal lobe it mainly involves the inferior frontal gyrus, pars
# opercularis (Broca's area; 52% of it damaged) ...
#
# It lies in the left middle cerebral artery (MCA) territory (100%): mainly its frontal part
# (58%) and insular part (29%), with 9% in the lateral lenticulostriate (deep perforator)
# territory.
#
# It reaches the left Broca's area, left arcuate fasciculus, left corticospinal tract and left
# primary motor cortex; the left primary auditory cortex is 2 mm away.

lqtpy.qc.check(les)                # [] -- or flags such as "drawn on some axial slices only"
lqtpy.lesion_stats(les)            # volume, centre of mass, components, laterality, ...
lqtpy.atlas_overlap(les, "aal")    # one record per damaged region
lqtpy.watershed.border_zones(les)  # share in arterial border zones
lqtpy.structures.distances(les)    # mm to the corticospinal tract, arcuate, motor cortex, ...
```

A lesion can be a path (`.nii`, `.nii.gz`, `.mgz`, `.nrrd`, `.nhdr`, DICOM-SEG `.dcm`), a
nibabel image, an `(array, affine)` tuple, or a `PreparedLesion`. A voxel counts as lesioned
when its value is **strictly greater** than `threshold` (0.5), and the mask is resampled onto
each atlas grid with nearest-neighbour interpolation.

| Function | Returns |
|---|---|
| `prepare(lesion)` | a `PreparedLesion`: loaded once, cached resampling per atlas grid |
| `describe(lesion, profile=...)` | a plain-language `Description` (`.text`, `.sections`, `.facts`) |
| `qc.check(lesion)` | quality findings (see below) |
| `qc.assess(lesion)` | the graded result: findings, verdict, score, JSON ([docs/QC.md](docs/QC.md)) |
| `qc.repair_gaps(lesion)` / `qc.clean(lesion)` | a **new** repaired / cleaned mask (opt-in) |
| `qc.check_folder(folder)` / `qc.clean_folder(src, dst)` | the same for a folder; cleaning writes copies |
| `qc.fix(lesion, ids)` / `qc.problem_mask(lesion, finding)` | a fixed copy plus the changed voxels / the affected voxels |
| `qc.rank(copies)` / `qc.assess_folder(folder)` | copies of one lesion best first / a folder as JSON with a summary |
| `tissue.locate(lesion)` | tissue composition, and a location class for each separate part |
| `watershed.border_zones(lesion)` | share in the ACA-MCA, MCA-PCA, ACA-PCA and internal border zones |
| `structures.distances(lesion)` | shortest distance (mm) to key tracts and cortical areas |
| `structures.cross_section_load(lesion)` | how much of each tract's cross-section the lesion covers |
| `mirror.homologue(lesion)` | regions under the lesion's mirror image in the other hemisphere |
| `lesion_stats(lesion)` | volume, centre of mass, components, laterality, bounding box |
| `atlas_overlap(lesion, atlas)` | one record per damaged region |
| `atlas_coverage(lesion, atlas)` | labelled / unlabelled / outside-field-of-view shares |
| `network_rollup_for(lesion, atlas)` / `lobe_overlap(lesion, atlas)` | grouped by network or lobe |
| `tract_proxy_overlap(lesion)` | voxel overlap with tract masks — a coarse proxy, **not** disconnection |
| `tract_disconnection(lesion)` | percent of each tract's streamlines cut by the lesion |

`PercentOfLesion` has one meaning everywhere: the share of *all* lesion voxels on the atlas
grid; `atlas_coverage` reports the part no region covers as `PercentUnlabelled`.

Bundled atlases: `aal` (116 regions), `schaefer` (100 parcels, Yeo-7 networks), `dk`
(Mindboggle DKT31 + subcortical), `jhu` (48 white-matter structures) and `arterial`
(vascular territories, Liu et al. 2023, with deep perforator territories). `tissue` classifies
every voxel (cortex, white matter, deep grey matter, ventricle, cerebellum, brainstem, CSF)
for the location words. `lqtpy fetch suit` downloads the cerebellar lobule atlas (Diedrichsen
2009) for `--atlas suit`. Register your own with `lqtpy.register_atlas(key, name, img, labels)`.

### One vocabulary

Every output uses the same words and numbers: volumes in cm³ with one rounding rule
(15.6 cm³, 0.42 cm³), quality as **OK / Note / Check / Problem**, "deep grey matter" (short
"deep GM") for the basal ganglia and thalamus, and territories as `L MCA` / `R PCA` /
`bilateral VB` in tiles and tables but "the left middle cerebral artery (MCA) territory" in
text. The midline share is always shown (a brainstem lesion is "on the midline (brainstem),
45% left, 45% right, 9% on the midline"); a lesion straddling one territory on both sides is
one bilateral territory.

### Descriptions

`describe()` builds structured facts (hemisphere → lobe → region, then each separate part)
and renders them with deterministic templates. Pick a `profile`:

| Profile | For |
|---|---|
| `stroke` (default) | one or a few lesions; lobe and region detail, each part described |
| `small_vessel` | many small lesions (MS, small-vessel disease); counts and distribution |
| `tumour` | one dominant mass; midline crossing, ventricle involvement |
| `research` | neutral anatomical wording, lower mention thresholds |

Sections: summary, location, tissue, vascular territory (with border zones when the location
fits a watershed pattern), nearby structures (tracts and cortical areas the lesion reaches or
comes within 5 mm of), functional areas, networks, shape (extra) and quality notes. Key
findings are marked (`Description.marked`, `Description.html()`); functional labels describe
location only and never predict symptoms. "Lacunar-sized" means one subcortical part at most
20 mm across **and** at most 1.5 cm³.

### Tissue and location classes

`lqtpy.tissue.locate(lesion)` gives tissue composition and a summary class (cortical,
subcortical, cortico-subcortical, infratentorial); each separate part gets the MS /
small-vessel location class:

| Class | Rule (checked in this order) |
|---|---|
| outside the brain | at least half outside the brain (usually a drawing artefact) |
| infratentorial | at least half in the brainstem or cerebellum |
| cortical | at least half in the cortex |
| deep grey matter | at least half in the basal ganglia or thalamus |
| periventricular | reaches the lateral ventricles |
| juxtacortical | white matter reaching the cortex |
| deep white matter | white matter touching neither |

"Reaches" means within 3 mm in the MNI template (warped masks don't line up to the
millimetre).

### Border zones, distances, cross-sections, mirror image

- **Border zones** (`lqtpy.watershed`): the strips where two arterial territories meet
  (ACA-MCA, MCA-PCA, ACA-PCA: cortex and the white matter within 10 mm of it) and the
  internal border zone in the deep white matter between the lenticulostriate and superficial
  MCA supply. The description says a location *fits* a border-zone (watershed) pattern when
  at least half the lesion lies in the zones, or three or more parts line up along the
  internal zone. Zones come from the territory atlas and are approximate.
- **Distances** (`lqtpy.structures.distances`): mm to the corticospinal tracts, arcuate
  fasciculi, optic radiations, primary motor and auditory cortex, and Broca's area (and its
  right homologue); 0 means the lesion reaches it.
- **Cross-section load** (`cross_section_load`): voxel overlap under-rates a lesion that cuts
  a narrow part of a tract. Each tract is cut into 1 mm slabs along its main axis; reports
  show the largest share of any slab covered, and the share of its narrowest slab.
- **Mirror image** (`lqtpy.mirror`): the lesion flipped across the midline, and the regions
  under it. The homologous region is used as an intact control region and in recovery
  research (e.g. the right-hemisphere homologue of Broca's area in aphasia).

### Quality checks

`qc.check()` flags masks that would give misleading numbers, tuned for MNI-space masks where a
warp has smeared the artefacts: drawn on some slices only (`sparse_slices`), a single slice
(`flat`, including a lesion that is mostly one thin slab with other pieces attached), an
unfilled outline (`hollow`), lines or sheets drawn by mistake (`strokes`), fragments on the
image edge (`edge_artifact`), stray specks, volume outside the brain, a mask not in MNI space, header problems, and a left/right
mismatch when the expected side is known. It also finds holes, rectangle-tool blocks,
odd label values and probability maps, inverted masks, masks shifted out of the brain, masks
shattered into many parts, and implausible lesions (tiny, huge, only CSF, a mirror copy).
Nothing is changed unless you ask.

Each finding is **graded by its extent**. It says how much ("Check, moderate: 14% of the lesion
lies outside the brain") and **where** (a slice or a point, so a viewer can jump there). It
also carries a fix hint. The mask gets a **verdict** (usable / usable with caution / fix before
use / unusable) and a **score** for ranking copies. Other points:

- Three tiers (`header`, `fast`, `full`).
- An optional **drawing context** from a drawing tool (the slices actually drawn, the drawing
  plane), which gives exact gap slices.
- A versioned JSON schema: key caches on `qc.QC_VERSION`.

Tested on a synthetic library of every problem type at several extents, and on 1,335 real
clinical masks. The full reference is [docs/QC.md](docs/QC.md).

**Native space**: gaps are clearest on the scan the mask was drawn on. Pass the native mask as
well (`analyze_single(mni_mask, native=native_mask)`) and the drawing checks run there; or
pass only the native mask and its transform (`transform=` a 4x4 native-to-MNI matrix in mm, or
an SPM-style deformation field) and LQTpy warps it for you (`lqtpy.native.warp_to_mni`,
`lqtpy warp`).

### Core and extras

Core covers everything above. **Extras** are measures in `lqtpy.registry`: the built-in
`shape` extra (surface area, sphericity, solidity, principal axes) with plain-word
interpretations calibrated on 898 real lesions, and the `sensitivity` extra: how much the
volume and region damage change if the outline is drawn one voxel smaller, larger or smoother
(how much weight the exact numbers can bear). Plugins add more through the
`lqtpy.measures` / `lqtpy.atlases` entry points; `lqtpy new-plugin fractal` writes a
ready-to-install package (pyproject with the entry point, `compute()`, `interpret()`, tests).
A plugin that fails to import is reported as unavailable; it never breaks core. Extras are
off by default everywhere (`extras="all"` / `--extras all` to add them).

## Reports

```python
lqtpy.analyze_single("lesion_MNI.nii.gz", out_dir="out", plots=True)
lqtpy.analyze_group("lesions/", out_dir="out", plots=True, metadata="subjects.csv",
                    compare="aphasia", n_jobs=4)
```

Each report is **one self-contained HTML file** with a **dashboard** (dark, with a light mode)
and a **clinical report** (a paper-style document, also what **Export PDF** prints). Figures
are drawn with numpy as themeable vector graphics; a full report takes under a second.

- Key numbers, the description (with **Copy text** and glossary popovers for terms such as
  periventricular, lacunar-sized or border zone), quality flags, glass-brain projections and
  slices, and a **scroll-through slice viewer**.
- Where the lesion is: share of the lesion, most damaged structures, territories and border
  zones, nearby structures.
- Atlas tables with vector region maps: **hover a table row to find the region on the map**,
  and back. Every table sorts by any column and **remembers its sort**.
- Tracts (voxel overlap and cross-section load), disconnection, the mirror image, extras.
- **Download tables**: the report carries its own CSVs, plus `data_dictionary.csv` explaining
  every column and unit (also written next to every output).
- **About this report**: what ran and why not, versions, input checksums, and the command
  that repeats the analysis.

Group reports add a subjects table with **thumbnails** and metadata columns, a **search box
and filter chips** (quality, territory, tissue, metadata groups) that recompute the cohort
bars, **cohort checks** (duplicate masks, outliers, unmatched metadata), a summary per
`group_by` group, overlap maps and, with `compare=`, a **two-group comparison** (overlap maps
side by side, per-region Fisher and Mann-Whitney tests with FDR q-values: screening
statistics, not lesion-symptom mapping). One full report per subject is linked from the group
table. `pdf="bundle"` also writes every subject's PDF and, with `pypdf`, one merged PDF with
bookmarks.

### Group options

| Option | What it does |
|---|---|
| `metadata="subjects.csv"` | one row per subject (age, condition, side, scores), joined into tables and filters |
| `group_by="condition"` | summary per group |
| `compare="aphasia"` or `("aphasia", "yes", "no")` | two-group comparison |
| a side column (`lesion_side`, or `symptom_side` = opposite) | left/right flip check per subject |
| `anonymise=True` | IDs (`sub-0001`) instead of names everywhere; a private key file maps them back |
| `n_jobs=4` | parallel workers (on Windows, run from under `if __name__ == "__main__":`) |
| `incremental=True` | reuse cached results for unchanged files and settings |

### Reproducibility

Every report embeds a run record: versions, settings and each input's SHA-256.
`lqtpy rerun report.html` repeats the analysis; moved inputs are found by checksum with
`--search FOLDER`, and a changed file is refused unless `--allow-changed`.

## Command line

```bash
lqtpy single lesion.nii.gz -o out              # full report
lqtpy group lesions/ -o out -m subjects.csv --compare aphasia -j 4
lqtpy describe lesion.nii.gz --profile stroke
lqtpy qc lesion.nii.gz                         # graded findings for one mask
lqtpy qc lesion.nii.gz --json                  # the same as JSON (--tier fast, --context ctx.json)
lqtpy qc lesions/ --clean-to cleaned/          # one page for a folder; cleaned copies
lqtpy repair lesion.nii.gz -o repaired.nii.gz  # interpolate gap-drawn masks (new file)
lqtpy warp native.nii.gz --transform T.txt -o mni.nii.gz
lqtpy rerun out/lesion_report.html
lqtpy watch incoming/ -o out                   # rebuild the group report as masks arrive
lqtpy serve                                    # upload page for colleagues without Python
lqtpy new-plugin fractal                       # starter package for an extra measure
lqtpy fetch suit                               # cerebellar lobule atlas
```

Output is plain, aligned text (colour only for the quality word, and only in a terminal);
`--quiet` prints just the report path. Errors print one line and exit with 2 (bad input) or
1 (environment).

**Settings file**: defaults for atlases, profile, extras, output folder, threshold, workers
and more go in `lqtpy.toml` (current folder, `LQTPY_CONFIG`, or the user config folder;
`lqtpy config` shows which is used). The command line wins over the file; the Python API
ignores it.

```toml
[defaults]
atlases = ["aal", "dk", "arterial"]
profile = "stroke"
out_dir = "D:/lesion_reports"
jobs = 4
```

**Web front end**: `lqtpy serve` starts an upload page on `http://localhost:8765` (this
computer only unless `--host`); a colleague uploads a mask and gets the report back. Uploads
are anonymised and deleted straight after.

## Documentation

`python tools/build_docs.py` builds a documentation site (`docs/site/`: overview, command-line
reference, Python API from the docstrings) and an example gallery (`docs/gallery/`: a stroke,
small-vessel disease and a tumour report from synthetic lesions). `python tools/bench.py`
prints the latency table; `python tools/visual_check.py` compares report screenshots with
`tests/visual/baseline/` (`LQTPY_VISUAL=1 pytest` runs it as a test).

Result dictionaries use snake_case keys; table columns keep the CamelCase names of the
original toolkit's outputs (`PercentDamage`, `RegionName`), so existing CSV readers keep
working.

## Tract disconnection

A streamline counts as disconnected if it passes through any lesion voxel, and the reported
percentage is that count over the tract's reference streamline population. This matches
DIPY's `target()` semantics — verified to select exactly the same streamlines.

It needs a voxel→streamline index, built once from the HCP-842 tractography atlas (70
`*.trk.gz` bundles plus `tract_info.csv`). At ~18 MB built, it isn't shipped:

```bash
python tools/build_discon_index.py --source /path/to/All_Tracts --out "$LQTPY_DATA_HOME/discon_index_v1.npz"
```

Queries then take about 90 ms. Without an index, `tract_disconnection` raises
`DataUnavailableError` explaining this; everything else still works.

The tract geometry is easy to get wrong: those TRK files carry no translation in their
headers, so the obvious `load_tractogram(..., to_space=RASMM)` lands roughly 120 mm off. The
correct frame, and the Dice validation behind it, are in
[docs/VALIDATION.md](docs/VALIDATION.md).

## Data and citation

The bundled atlases are third-party data, and two carry terms that follow the files wherever
they go:

- **DKT** — Mindboggle-101 (Klein & Tourville), [CC BY 4.0](https://doi.org/10.5281/zenodo.22070005)
- **Tract proxy index** — derived from the HCP-842 atlas (Yeh et al. 2018),
  [CC BY-SA 4.0](https://brain.labsolver.org/hcp_trk_atlas.html); using it also requires
  acknowledging the [HCP data use terms](https://www.humanconnectome.org/study/hcp-young-adult/document/wu-minn-hcp-consortium-open-access-data-use-terms)
- **AAL** and **Schaefer-2018** — free for academic use with citation
- **SUIT cerebellar lobules** (optional, `lqtpy fetch suit`) — Diedrichsen et al. 2009, downloaded
  from the DiedrichsenLab `cerebellar_atlases` repository; academic use with citation

Per-file provenance and checksums are in
[lqtpy/data/atlases/SOURCE.md](lqtpy/data/atlases/SOURCE.md).

If you use this, cite the method it reimplements:

> Griffis JC, Metcalf NV, Corbetta M, Shulman GL. Lesion Quantification Toolkit: A MATLAB
> software tool for estimating grey matter damage and white matter disconnections in patients
> with focal brain lesions. *NeuroImage: Clinical* 2021;30:102639.
