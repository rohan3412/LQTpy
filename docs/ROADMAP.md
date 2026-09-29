# LQTpy roadmap

Written 2026-09-25, after the full audit
([report](https://claude.ai/artifact/SZyQHJCh25GL2qsnt7oGPN)). Phases are ordered by
dependency: each one builds on the one before it.

## Status (2026-09-29)

| Phase | State |
|---|---|
| 0. Quick fixes | Done |
| 1. Numbers that can be quoted | Done. 1.7 needed no change: SciPy's default filter prints that warning once per process, not per call. |
| 2. Foundations | Done: `prepare`, schema v2, registries, manifest, `tools/bench.py` (all budgets met) |
| 3. QC module | Done. The brain mask is derived from the bundled atlases until Phase 4 adds a real one. |
| 4. Atlases | Done: `jhu` white matter, `arterial` territories (header origin corrected). The brain mask stays atlas-derived (FSL grid) after comparison with nilearn's ICBM-2009a mask. |
| 5. Description | Done, including territories, white-matter structures and emphasis of key findings |
| 6. Shape extras | Done: synthetic bands plus a 898-lesion cohort reference (aggregates only) |
| 7. Reports | Done: dashboard (dark/light) + clinical view, Export PDF, `to_pdf()` |
| Tissue and location classes (added 2026-09-28) | Done: `lqtpy.tissue`, bundled `tissue` atlas; cortical / juxtacortical / periventricular / deep white matter / deep grey / infratentorial per part, tissue composition per lesion, in descriptions, results and reports |
| Design and consistency review (2026-09-28) | Done: one vocabulary for volumes, quality, deep grey matter and territories; midline share; six tiles; figures without matplotlib (report < 1 s) |
| Ideas inbox (2026-09-29) | Done: metadata, border zones, duplicates, outliers, parallel and incremental runs, anonymised output, group comparison, flip check, folder QC, web front end, report conveniences, reproducibility and rerun, config file, thumbnails, distances, cross-section load, data dictionary, SUIT atlas, mirror image, watch mode, MGZ/NRRD/DICOM-SEG, native space, plugin template, visual regression, accessibility, table-map linking, remembered sort, CLI polish, API tidy-up, gallery, docs site. See CHANGELOG.md |

Real-data validation: 1,231 clinical masks, no failures. QC was tuned on them (new
`edge_artifact` check); a visual review of flagged samples confirmed the sparse-slice and
single-slice flags.

See [CHANGELOG.md](../CHANGELOG.md) for everything that changed.

## Goals

1. **Natural-language lesion description.** Hierarchical: hemisphere → lobe → atlas region
   (with functional labels such as "Broca's area"), then one description per disconnected
   cluster. Built from templates over structured facts, so it is deterministic and testable.
   Profiles for stroke, small-vessel/MS, tumour/TBI/resection, and neutral research wording.
2. **Quality control of drawn masks.** Gaps between drawn slices, single-slice (flat)
   lesions, hollow outlines, stray specks, meaningful volume outside the brain, and masks
   that aren't in MNI space. Always flag problems. Repair is an explicit opt-in that returns
   a new mask; the input is never changed silently.
3. **Core + extras.** Core stays fast enough to call in-process from a web API. Extra
   measures and atlases are plugins discovered through entry points (the user's
   fractal-dimension package will be the first external one).
4. **Reports that state their scope** (what ran, what didn't, and why), keep core and extras
   visibly distinct, interpret shape measures in words, and write one report per subject
   in a group run, linked from the group report.

Masks are primarily MNI-space. Licensing is not a constraint (private use), but every data
file's provenance still goes in `lqtpy/data/atlases/SOURCE.md`.

---

## Phase 0: Quick fixes (audit high-severity items and security)

One commit per item, each with a regression test.

| # | Fix | Where |
|---|---|---|
| 0.1 | CI wheel job checks for the current dk files (`OASIS-TRT-20_…_v2.nii.gz`, `data/dkt31_cma_labels.csv`) | `.github/workflows/ci.yml:116-117` |
| 0.2 | Escape `top_region`; check every other interpolation in the standalone report | `lqtpy/report/_standalone.py:274` |
| 0.3 | `analyze_group(<file path>)` and `lqtpy group one.nii.gz`: a non-directory str/PathLike becomes `[path]`; also accept a PathLike directory | `lqtpy/analysis.py:439`, `lqtpy/cli.py:59` |
| 0.4 | Bool arrays → uint8 before `Nifti1Image` | `lqtpy/io.py:44-57` |
| 0.5 | Tighten `_COLOR_RE`: function arguments restricted to `[0-9a-z.,%/\s-]` | `lqtpy/report/style.py:28` |
| 0.6 | Pickle-free data files: disconnection index format v2 (`tract_codes` stored as `U` dtype, `allow_pickle=False`, typed error for v1); convert `hcp842_tracts.npz` `names` to `U` | `lqtpy/_discon_index.py`, `lqtpy/tracts.py:73` |
| 0.7 | `atlases="aal"` (bare string) is treated as `["aal"]` | `lqtpy/analysis.py:350, 470` |

**Done when:** CI is green on all three jobs and every item has a test that fails before the fix.

## Phase 1: Numbers that can be quoted

The description will quote these numbers, so they have to mean one thing each.

- **1.1 Voxel volume from the affine.** Voxel size = column norms of `affine[:3,:3]`,
  voxel volume = `|det|`. Report a `header_zooms_mismatch` warning when the header disagrees.
- **1.2 One definition of `PercentOfLesion`.** The denominator is always the total lesion
  voxels on the atlas grid. Rollups receive that total instead of re-summing labelled rows.
  Add explicit `UnlabelledVoxels` / `PercentUnlabelled` and `OutsideAtlasFOVVoxels`, so
  the numbers add up to 100% and nothing disappears.
- **1.3 Group means.** Keep the mean over affected subjects, rename it
  `MeanPercentDamageAffected`, and add `CohortMeanPercentDamage` (÷ n_subjects) and
  `NSubjectsTotal`.
- **1.4 Typed errors everywhere.** Add `IndexFormatError` (subclass of
  `DataUnavailableError`) and `InvalidAtlasError`, guard the empty index, validate custom
  label volumes at registration (negative labels, non-integer floats beyond tolerance), and
  flag a lesion entirely outside the atlas field of view.
- **1.5 Input edge cases.** Non-finite voxels are treated as 0 and counted in `n_nonfinite`.
  Midline voxels (world x = 0) are reported as `pct_midline`, so left + right + midline = 100.
- **1.6 `register_atlas`** keeps `Lobe` (and the later schema-v2 columns) from dict labels.
- **1.7 Quieter resampling.** Suppress the SciPy 1-D-matrix `UserWarning` locally in
  `_resample.py`, without touching global filters (the hygiene tests guard this).
- **1.8 CLI.** Typed errors become one-line messages with exit code 2 (bad input) or 1
  (environment); add `--version`.
- **1.9 Documentation drift.** Stale `pyproject` comment, `SOURCE.md` file count, the
  missing `DATA-LICENSES.md` reference, and `ggseg` added to the `plots` extra.

Release as **0.5.0**. Any change to output fields is noted in a CHANGELOG.

## Phase 2: Foundations

### 2.1 Prepared lesion (performance and reuse)
`lqtpy.prepare(lesion, threshold=0.5) -> PreparedLesion` is loaded once and holds:
- the binary mask thresholded in its native dtype (no float64 copy);
- the affine and bounding box, with connected components computed lazily;
- a thread-safe cache of the mask resampled onto each atlas grid, keyed by grid
  (shape + affine), with the source cropped to its bounding box before resampling.

Every `lqtpy.api` function accepts either a `PreparedLesion` or anything it accepts today.
Morphometry crops to the bounding box before `ndimage.label` and `center_of_mass`.

**Latency budgets** (1 mm FSL-grid lesion, warm; checked by `tools/bench.py`):

| Call | Today | Target |
|---|---|---|
| `lesion_stats` | 201 ms | ≤ 30 ms |
| `atlas_overlap` (resampled atlas) | 120 ms | ≤ 60 ms |
| all five API calls on one prepared lesion | 647 ms | ≤ 200 ms |
| `analyze_group`, per subject, no plots | ~470 ms | ≤ 200 ms |

### 2.2 Atlas label schema v2
Every atlas label table gets these columns (all optional except the first two):
`LabelID, RegionName, DisplayName, Hemisphere (L/R/Midline), Lobe, Tissue
(GM/WM/Subcortical/Cerebellar/CSF/Brainstem), Network, Function, Territory`.

- `RegionMeta` gains these as trailing fields with `None` defaults, so existing code keeps working.
- `DisplayName` and `Function` are curated for `aal` and `dk`: "left inferior frontal
  gyrus, opercular part" and "Broca's area (part)", and so on.
- An atlas-table validator runs in tests for every bundled atlas.

### 2.3 Plugin registries (core + extras)
`lqtpy.registry` defines two plugin kinds.

- **Measure:**
  - `name`, `tier` (`core`/`extra`), `version`, `requires` (optional dependencies)
  - `fields`: name, unit and plain-language meaning for each output
  - `min_voxels`: below this the measure reports "too small to characterise"
  - `compute(prepared) -> dict`
  - `interpret(values, context) -> list[Descriptor]` (words, used by Phase 6)
- **Atlas:**
  - `key`, `name`, `tier`, `license`, `citation`
  - a loader, or fetch info (URL + sha256, using the existing `datasets.fetch`)

Discovery uses the entry-point groups `lqtpy.measures` and `lqtpy.atlases`. It runs lazily
on first use, never at import, so `import lqtpy` stays lightweight. A plugin that fails to
import is listed as *unavailable* with its reason; it never breaks core. Core morphometry
is re-expressed as `core` measures.

### 2.4 Run manifest
Every analysis result carries a `manifest`:
- report kind (single / group / group-member);
- each module with status `ran | skipped | unavailable | failed` and a reason
  (e.g. "no disconnection index installed", "plugin lqtpy-fractal not installed");
- settings (threshold, atlases), and lqtpy and plugin versions.

The report scope block in Phase 7 is rendered directly from this.

---

## Phase 3: QC module (`lqtpy.qc`)

Every check returns `{id, severity: info|warn|error, message, evidence, suggestion}`. QC
runs **before** clustering and description, and its flags appear in both.

| Check | Approach (MNI-space masks, where warping smears artefacts) |
|---|---|
| Gaps / sparse slice drawing | For each component and each axis, compute the per-slice area profile. Flag dips (area well below neighbouring slices) and periodicity (autocorrelation peak at k = 2–6 slices) along one axis only. |
| Single-slice / flat lesion | Principal-axis extents: the thinnest extent ≤ ~3 voxels while the other two are large. Report the likely drawing plane. |
| Hollow outline | `binary_fill_holes`, slice by slice in the drawing plane; the filled volume is much larger than the drawn volume. |
| Stray specks | Components below a size threshold (default a few voxels), reported by count and location. Not an error on their own. |
| Outside the brain | Against a bundled MNI brain mask. Flag only a contiguous chunk over *max(100 mm³, 5% of lesion)*. Scattered voxels and ventricle overlap are not flagged. |
| Not in MNI space | Grid and field-of-view sanity checks, a large fraction outside the brain, qform/sform disagreement or a missing sform. |
| Side mismatch | Only when the caller passes `expected_side=`. The mask alone can't show a left/right flip. |

- **Repair (opt-in):** `lqtpy.qc.repair_gaps(lesion, axis=None) -> (repaired_img, report)`
  uses shape-based interpolation (signed-distance interpolation between drawn slices).
  It never runs implicitly.
- **Gap-aware clustering:** components separated along the drawing axis by a few empty
  slices, with overlapping in-plane footprints, are merged into one cluster so the
  description doesn't report fake separate lesions.
- **Test data:** a synthetic generator that draws lesions on thick native slices (every
  1st, 3rd or 5th slice, single slice, outline-only), warps them to MNI with realistic
  affines, and checks each detector on the smeared result.

## Phase 4: Atlases

- **Vascular territories** (ACA/MCA/PCA, deep/lenticulostriate, watershed zones), e.g. the
  Johns Hopkins arterial territory atlas (Liu et al.). Verify the source and levels before
  bundling.
- **White-matter labels** (e.g. JHU ICBM-DTI-81), so deep lesions aren't just "unlabelled".
- **MNI brain mask** (needed by QC).
- **Optional:** cortical / juxtacortical / periventricular / deep descriptors, only if
  validation shows they're reliable.

Each atlas fills in schema v2 and is registered as a plugin (core or extra, decided per size).

## Phase 5: Natural-language description (`lqtpy.describe`)

1. **Facts model:** a nested structure (whole lesion → hemisphere → lobe → region, then
   → cluster) carrying volume, % of lesion, % of structure damaged, territory, tissue,
   functional labels, the QC flags that apply, and the unlabelled share.
2. **Templates** render the facts. Mention thresholds (for example, a region is named if it
   holds ≥ 5% of the lesion *or* is ≥ 10% damaged) and phrasing live in profiles:
   `stroke`, `small_vessel`, `tumour`, `research`. `small_vessel` summarises many clusters
   ("23 lesions, predominantly periventricular…") instead of one paragraph each.
3. **Caveats** come from QC ("drawn with gaps on axial slices; the extent may be
   underestimated").
4. **Tests** compare full output against checked-in expected text for each synthetic
   patient and profile.

## Phase 6: Extra shape measures with interpretation

- **Measures:** surface area (marching cubes), sphericity, principal-axis elongation and
  flatness, solidity, distance to midline, cortex and ventricles. Later, fractal dimension
  (the user's repo) as an external plugin.
- **Interpretation:** each measure maps to word bands. Combinations produce descriptors
  such as "compact, roughly round", "elongated front-to-back", "flat, sheet-like",
  "irregular, lobulated margins" and "branching". Lesions under `min_voxels` read as
  "too small to characterise shape".
- **Calibration:**
  1. Synthetic reference shapes at several sizes fix the meaning of each band.
  2. The user's real lesion sets, grouped by condition, give relative wording ("more
     irregular than 80% of the reference lesions").
  - Sample so each condition contributes equally, keep per-condition quantiles as well,
    and ship **only aggregate quantiles**, never masks.
  - Correct for size, since sphericity and fractal dimension depend on voxel count.
- **Where the output goes:** reports show the words, with numbers in collapsible detail;
  the CSV has every number.

## Phase 7: Reports, then a design pass

- **Scope block** from the manifest: report kind, what ran, what didn't and why, versions.
- **Core vs extras** sections look clearly different. A module that didn't run shows a
  short placeholder explaining how to enable it.
- **Content:** description and QC flags go first, with real disconnection when an index
  exists. Group reports get lobe tables and **one report per subject**, linked from the
  group report.
- **Design pass** across the whole project (reports, CLI output, docs): done. Reports
  first (dark dashboard, light mode, clinical view); CLI output as plain aligned text with
  colour only for quality words (chosen 2026-09-28); docs as a generated site and gallery.

---

## Cross-cutting

- Work on a feature branch, with small commits and each phase in its own PR. CI must stay green.
- A shared synthetic-lesion library in `tests/synthetic.py` (spheres at MNI coordinates,
  multi-cluster, thick-slice drawings warped to MNI, outline-only, specks).
- `tools/bench.py` prints the Phase 2 latency table. CI runs it for information but
  doesn't gate on absolute times.
- No patient data in the repo. Calibration commits aggregate statistics only.
