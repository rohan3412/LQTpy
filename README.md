<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="Lcutiepie (lqtpy)" width="720">
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
pip install .            # core
pip install ".[all]"     # + pandas frames, matplotlib/nilearn/ggseg figures, HTML reports
```

Atlases ship inside the package, so everything except tract disconnection works offline.

## Usage

```python
import lqtpy

les = lqtpy.prepare("lesion_MNI.nii.gz")   # load and binarize once, reuse everywhere

print(lqtpy.describe(les))
# A single 14 cm³ lesion in the left hemisphere.
#
# It involves the left frontal lobe (26%), insula (19%) and deep grey nuclei (3%). About half
# of it (51%) lies in white matter or other tissue the DKT atlas doesn't label. In the left
# frontal lobe it mainly involves the inferior frontal gyrus, pars opercularis (Broca's area;
# 54% of it damaged) and the precentral gyrus (primary motor cortex; 4% of it damaged). ...

lqtpy.qc.check(les)                # [] -- or flags such as "drawn on some axial slices only"
lqtpy.lesion_stats(les)            # volume, centre of mass, components, laterality, ...
lqtpy.atlas_overlap(les, "aal")    # one record per damaged region
lqtpy.atlas_coverage(les, "aal")   # how much of the lesion the atlas labels at all
lqtpy.lobe_overlap(les, "dk")      # the same, grouped by lobe
```

A lesion can be a path, a nibabel image, an `(array, affine)` tuple, or a `PreparedLesion`.
A voxel counts as lesioned when its value is **strictly greater** than `threshold` (0.5), and
the mask is resampled onto each atlas grid with nearest-neighbour interpolation. Preparing
once matters: the five calls above take about 50 ms together on a prepared lesion.

| Function | Returns |
|---|---|
| `prepare(lesion)` | a `PreparedLesion`: loaded once, cached resampling per atlas grid |
| `describe(lesion, profile=...)` | a plain-language `Description` (`.text`, `.sections`, `.facts`) |
| `qc.check(lesion)` | quality flags (see below) |
| `qc.repair_gaps(lesion)` | an interpolated **new** mask for gap-drawn lesions (opt-in) |
| `lesion_stats(lesion)` | volume, centre of mass, components, laterality, bounding box |
| `atlas_overlap(lesion, atlas)` | one record per damaged region |
| `atlas_coverage(lesion, atlas)` | labelled / unlabelled / outside-field-of-view shares |
| `network_rollup_for(lesion, atlas)` | the same, grouped by the atlas's region groups |
| `lobe_overlap(lesion, atlas)` | the same, grouped by anatomical lobe |
| `tract_proxy_overlap(lesion)` | voxel overlap with tract masks — a coarse proxy, **not** disconnection |
| `tract_disconnection(lesion)` | percent of each tract's streamlines cut by the lesion |

`PercentOfLesion` has one meaning everywhere: the share of *all* lesion voxels on the atlas
grid. The part no region covers (usually white matter for a grey-matter atlas) is reported by
`atlas_coverage` as `PercentUnlabelled`, so region rows plus the unlabelled share add up to 100.

Bundled atlases: `aal` (116 regions), `schaefer` (100 parcels, Yeo-7 networks), `dk`
(Mindboggle DKT31 + subcortical), `jhu` (48 white-matter structures, JHU ICBM-DTI-81) and
`arterial` (vascular territories, Liu et al. 2023, with deep perforator territories such as
the lenticulostriate). Every region carries a readable name, hemisphere, tissue class and,
where well established, a functional label (primary motor cortex, Broca's area, …). Register
your own with `lqtpy.register_atlas(key, name, img, labels)`.

### Descriptions

`describe()` builds structured facts (hemisphere → lobe → region, then each separate part of
the lesion) and renders them with deterministic templates. Pick a `profile`:

| Profile | For |
|---|---|
| `stroke` (default) | one or a few lesions; lobe and region detail, each part described |
| `small_vessel` | many small lesions (MS, small-vessel disease); counts and distribution |
| `tumour` | one dominant mass; midline crossing, ventricle involvement |
| `research` | neutral anatomical wording, lower mention thresholds |

Descriptions name the vascular territory ("the left middle cerebral artery (MCA) territory
(86%): almost entirely its lateral lenticulostriate (deep perforator) territory (86%)") and the
white-matter structures involved ("the posterior limb of the internal capsule"). Key findings
are marked: `Description.marked` wraps them in `**`, `Description.html()` renders them bold,
and `.text` stays plain. Functional labels describe location only; the text never predicts
symptoms.

### Quality checks

`qc.check()` flags masks that would give misleading numbers, tuned for MNI-space masks where
a warp has smeared the artefacts: drawn on some slices only (`sparse_slices`), drawn on a
single slice (`flat`), an outline never filled in (`hollow`), fragments left on the outermost
slice of the image by drawing/export tools (`edge_artifact`), stray specks, meaningful volume
outside the brain (a rim of overdrawing is ignored; the ventricles count as brain), a mask
that isn't in MNI space, header problems, and — only if you pass `expected_side` — a
left/right mismatch. Nothing is changed unless you ask: `qc.repair_gaps()` interpolates
gap-drawn masks and `qc.clean()` removes edge artefacts and specks, each returning a new
mask. Gap-drawn slabs are counted as one lesion in descriptions.

Checked on a clinical database of 1,231 real MNI masks: every mask ran without error, and a
visual review of flagged and unflagged samples found the sparse-slice and single-slice flags
correct. 28% of those masks carried image-edge artefacts, up to 1.2 cm³ each.

### Core and extras

Core (always available, fast enough for a web request) covers everything above. **Extras**
are measures registered in `lqtpy.registry`: the built-in `shape` extra (surface area,
sphericity, surface regularity, solidity, principal axes) comes with plain-word
interpretations ("elongated front-to-back, with a lobulated outline"). Word bands were set on
synthetic reference shapes, and a reference built from 898 real lesions adds size-matched
context when a lesion is unusual ("more lobulated than 90% of similar-sized reference
lesions"); only aggregate percentiles are shipped. Plugins add more
through the `lqtpy.measures` / `lqtpy.atlases` entry points:

```toml
# pyproject.toml of a plugin package
[project.entry-points."lqtpy.measures"]
fractal = "lqtpy_fractal:MEASURES"
```

A plugin that fails to import is reported as unavailable; it never breaks core.

### Reports

```python
lqtpy.analyze_single("lesion_MNI.nii.gz", out_dir="out", plots=True, extras="all")
lqtpy.analyze_group("lesions/", out_dir="out", plots=True)   # + one report per subject
```

Every result carries a `manifest` saying which modules ran, which were skipped or
unavailable, and why. Each report is one self-contained HTML file with two views, switched
from its toolbar: a **dashboard** (dark by default, with a light mode) that puts key numbers,
the description, quality flags and figures first and keeps atlas tables collapsed below; and
a **clinical report**, a paper-style document that is also what the **Export PDF** button
prints. Extras appear in their own, visually separate section. Save a PDF from code with
`lqtpy.report.to_pdf(result["report_path"])` or `analyze_single(..., pdf=True)`; it uses a
headless Edge/Chrome/Chromium already on the machine (or WeasyPrint if installed). Open a
specific view with `report.html#clinical` or `#light`. Group reports link to one full
report per subject and give, per region, subjects affected, the cohort-wide mean (unaffected
subjects count as 0), the mean among affected subjects, and the maximum.

To embed sections in your own page instead, use `lqtpy.report.fragments` — every rule is
scoped to `lqt-` classes, so the fragments survive being dropped into a page that has its own
CSS:

```python
from lqtpy.report import fragments
html = fragments.stylesheet() + fragments.parcel_damage_fragment(records, "AAL")
```

`ReportStyle` controls theme colours, which sections appear and in what order, units and
precision, and a detail-level preset.

### Command line

```bash
lqtpy single lesion_MNI.nii.gz -o out          # full report
lqtpy group lesions/ -o out                    # group report + one report per subject
lqtpy describe lesion_MNI.nii.gz --profile stroke
lqtpy qc lesion_MNI.nii.gz
lqtpy repair lesion_MNI.nii.gz -o repaired.nii.gz
lqtpy clean lesion_MNI.nii.gz -o clean.nii.gz  # remove image-edge artefacts and specks
```

`single`/`group` accept `--pdf`, `--profile`, `--extras all|none|name,...`, `--no-qc`,
`--no-describe`, `--no-tracts` and `--no-disconnection`. Errors print one line and exit with
2 (bad input) or 1 (environment).

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

Per-file provenance and checksums are in
[lqtpy/data/atlases/SOURCE.md](lqtpy/data/atlases/SOURCE.md).

If you use this, cite the method it reimplements:

> Griffis JC, Metcalf NV, Corbetta M, Shulman GL. Lesion Quantification Toolkit: A MATLAB
> software tool for estimating grey matter damage and white matter disconnections in patients
> with focal brain lesions. *NeuroImage: Clinical* 2021;30:102639.
