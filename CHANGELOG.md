# Changelog

## 0.5.0 (2026-09-29)

Implements Phases 0–7 of [docs/ROADMAP.md](docs/ROADMAP.md), then the design and consistency
review and the ideas inbox (listed first below).

### New since the roadmap phases
- **One vocabulary** (`lqtpy/_format.py`): volumes always in cm³ with one rounding rule;
  quality words OK / Note / Check / Problem everywhere (tiles, tables, CSV, CLI, PDF);
  "deep grey matter" (short "deep GM") instead of Subcortical / deep grey nuclei; territories
  as `L MCA` in tiles and tables, "the left middle cerebral artery (MCA) territory" in text.
- **Descriptions**: the midline share is never dropped ("on the midline (brainstem), 45% left,
  45% right, 9% on the midline"); a lesion straddling one territory is one bilateral territory;
  "lacunar-sized" also needs at most 1.5 cm³; unnamed tissue is split into unnamed white
  matter, cortex outside named regions, CSF spaces and outside the brain in every table;
  "Main location" is the largest compartment; no more "in the brainstem it mainly involves the
  brainstem"; the ACA's cortical part reads "cortical branches".
- **Border zones** (`lqtpy.watershed`): ACA-MCA, MCA-PCA, ACA-PCA cortical and internal
  border zones; the description says when a location fits a watershed pattern.
- **Key structures** (`lqtpy.structures`): distance in mm to the corticospinal tracts, arcuate
  fasciculi, optic radiations, primary motor and auditory cortex and Broca's area (and its
  homologue); **tract cross-section load** (largest share of a 1 mm cross-section covered, and
  of the narrowest one). New "Nearby structures" description section and report card.
- **Mirror image** (`lqtpy.mirror`): the homologous region in the other hemisphere.
- **Figures without matplotlib** (`lqtpy.figures`): glass brain, slices, overlap maps and
  thumbnails drawn with numpy as themeable inline SVG; region maps are vector graphics. A
  report with figures went from ~8 s to under 1 s; `plots=True` no longer needs matplotlib,
  nilearn or pandas.
- **Reports**: six tiles (Volume and Parts merged into Size) that fit; Copy text for the
  description; glossary popovers; a scroll-through slice viewer (brainsprite); table rows and
  map regions highlight each other; sort order remembered per table; the report's own CSVs
  downloadable from inside it; a reproducibility block with checksums and the rerun command;
  skip link, landmarks, labelled figures; a colour-blind-checked palette (left blue, right
  teal: tested under protanopia, deuteranopia and tritanopia).
- **Group runs**: subject metadata (`metadata=`, joined into tables and filters), `group_by`
  summaries, a two-group comparison (`compare=`: side-by-side overlap maps, Fisher and
  Mann-Whitney tests with FDR), left/right flip check from a side column, duplicate and
  near-duplicate masks, cohort outliers, parallel workers (`n_jobs`), incremental re-runs,
  anonymised output with a private key file, thumbnails, search and filter chips that
  recompute the cohort bars, and `pdf="bundle"` (every subject's PDF, merged with bookmarks).
- **Reproducibility** (`lqtpy.repro`): every report embeds a run record (versions, settings,
  input SHA-256); `lqtpy rerun report.html` repeats the analysis, finding moved files by
  checksum.
- **Inputs**: MGZ, NRRD (built-in reader) and DICOM-SEG (`highdicom`); native-space masks
  (`native=` for drawing checks, `transform=` to warp; `lqtpy warp`).
- **Folder QC**: `lqtpy qc FOLDER` writes one page of every mask's flags; `--clean-to` writes
  cleaned copies to a new folder, never overwriting.
- **Command line**: plain, aligned output with colour only for the quality word; the report
  path printed once, in the platform's form; `--quiet` prints only the path; `--atlas` no
  longer offers `tissue`; new `warp`, `rerun`, `watch`, `serve`, `fetch`, `new-plugin`,
  `config` and `cache` commands; `lqtpy.toml` settings file.
- **Web front end** (`lqtpy serve`): an upload page on top of the core; uploads are
  anonymised and deleted after the report is sent.
- **Watch-folder mode** (`lqtpy watch`), **plugin starter template** (`lqtpy new-plugin`),
  **optional cerebellar lobule atlas** (`lqtpy fetch suit`), **data dictionary**
  (`data_dictionary.csv` next to every output), **documentation site and gallery**
  (`tools/build_docs.py`), **visual regression tests** (`tools/visual_check.py`).
- Template maps (distance maps, border zones, brain mask) are cached on disk once per
  installation: a cold process makes a report in ~3.5 s instead of ~15 s.

### Changed (check your code)
- **Extras are off by default everywhere**: the CLI no longer runs `--extras all` unless asked
  (it used to, while `analyze_single()` didn't).
- `analyze_single`'s first argument is `lesion` (`lesion_path=` still works); arguments after
  `out_dir` are keyword-only — positional use still works with a `DeprecationWarning`.
- Result keys `glass_uri`/`slices_uri`/`overlap_uri`(+`_light`) and `flat_uris` are replaced
  by `glass_svg`, `slices_svg`, `overlap_svgs` and `flat_svgs`.
- The group overlap map (`lesion_overlap_map.nii.gz`) is on the FSL MNI152 1 mm grid.
- The group CSV's `QC` column uses OK / Note / Check / Problem; a `GroupFlags` column is added.
- Result dictionaries keep snake_case keys and tables keep CamelCase columns (documented, not
  changed, so existing CSV readers keep working).

## 0.5.0 roadmap phases


### New in the roadmap phases
- `lqtpy.prepare()` / `PreparedLesion`: load and binarize a lesion once, then reuse it across
  every API call. Resampling to each atlas grid is cached and cropped to the lesion, so it
  scales with lesion size, not grid size.
- `lqtpy.describe()`: plain-language, hierarchical descriptions (hemisphere → lobe → region,
  then each separate part), with `stroke`, `small_vessel`, `tumour` and `research` profiles.
  Deterministic templates over structured facts (`Description.facts`).
- `lqtpy.qc`: mask quality checks (`sparse_slices`, `flat`, `hollow`, `specks`,
  `outside_brain`, `not_mni`, `header`, `side_mismatch`, `input`), gap-aware clustering, and
  opt-in `repair_gaps()` (shape-based interpolation that returns a new mask).
- `lqtpy.registry`: core/extra measure registry with entry-point plugins
  (`lqtpy.measures`, `lqtpy.atlases`); broken plugins are reported, never fatal.
- Built-in `shape` extra: surface area, sphericity, surface regularity, solidity, principal
  axes, elongation/flatness, with calibrated plain-word interpretations
  (`tools/calibrate_shape.py`).
- `lqtpy.atlas_coverage()`: labelled / unlabelled / outside-field-of-view shares.
- Reports: every table sorts by clicking a column header; the group subjects list scrolls
  inside its card (sticky header) past 12 subjects; brain figures (glass brain, slices, group
  overlap) come in a white-background version for light mode, the clinical view and the PDF
  (`glass_uri_light`, `slices_uri_light`, `overlap_uri_light` in results).
- `lqtpy.tissue`: tissue composition (cortex, white matter, deep grey matter, ventricles,
  cerebellum, brainstem, CSF, outside) and location classes per separate part
  (periventricular, juxtacortical, cortical, deep white matter, deep grey matter,
  infratentorial), from a new bundled `tissue` atlas built by `tools/build_tissue_map.py`
  (DKT31/CMA + ICBM 2009a white matter). Descriptions gain a Tissue section
  ("cortico-subcortical", "small subcortical lesion ... lacunar-sized"), small-vessel counts
  by location, and a breakdown of the part no atlas names. Results carry
  `result["tissue"]`; single runs write `<name>_parts.csv`; group subject tables gain tissue
  columns; reports show a Tissue tile and, in the clinical view, a parts-by-location table.
- Atlas label schema v2: `DisplayName`, `Hemisphere`, `Tissue`, `Network`, `Function`,
  `Territory` on every `RegionMeta` (bundled tables filled in by
  `tools/build_label_columns.py`).
- Run manifests: every result records which modules ran, were skipped, unavailable or
  failed, and why. Reports open with this scope.
- `analyze_single` / `analyze_group` run QC, the description, extras (`extras=`), and
  streamline disconnection automatically when an index is installed. Group runs write one
  report per subject, linked from the group report, and lobe/network group tables.
- CLI: `describe`, `qc`, `repair` commands; `--version`, `--profile`, `--extras`,
  `--no-qc`, `--no-describe`, `--no-disconnection`, `--no-subject-reports`,
  `--expected-side`. Errors are one line with exit code 2 (input) or 1 (environment).
- `tools/bench.py` latency table; `tests/synthetic.py` synthetic lesion library, including
  thick-slice native drawings warped to MNI.
- Atlases `jhu` (JHU ICBM-DTI-81 white matter) and `arterial` (Johns Hopkins arterial
  territories, Liu et al. 2023; its missing header origin is corrected and pinned by a test).
  Both are defaults. Descriptions gain a vascular-territory section (with deep perforator
  territories) and name white-matter structures.
- Key findings are emphasised in descriptions (`Description.marked`, `Description.html()`).
- QC: `edge_artifact` (fragments on the image's outermost slice, found in 28% of a real
  database) and `qc.clean()` / `lqtpy clean` to remove them on request.
- Validated on 1,231 real clinical masks (no failures); shape descriptors get size-matched,
  cohort-relative context from 898 of them (`lqtpy/data/shape_reference.json`, aggregate
  percentiles only; `tools/calibrate_shape_cohort.py`).
- Report redesign: one self-contained file with a dashboard (dark by default, light mode)
  and a clinical report view; an Export PDF button; `lqtpy.report.to_pdf()`,
  `analyze_*(pdf=True)` and `--pdf` via a headless Edge/Chrome/Chromium. HTML bars replace
  matplotlib bar charts; brain figures use one lesion colour on black plates; flat maps are
  rendered for both themes.

### Changed (check your code)
- **`PercentOfLesion` in network and lobe rollups** now uses the whole lesion on the atlas
  grid as denominator, like region rows. Previously it used only labelled voxels, so a lobe
  table could say 98% where the true share was 6%. `network_rollup`/`lobe_rollup` take
  `lesion_total=`; without it they warn and keep the old behaviour.
- **Group tables**: `MeanPercentDamage` is replaced by `CohortMeanPercentDamage` (all
  subjects, unaffected = 0) and `MeanPercentDamageAffected`, plus `NSubjectsTotal`. Same for
  networks, lobes and tract overlap.
- **Voxel size and volume come from the affine**, not header `pixdim`; a disagreement is
  reported as `header_zooms_mismatch`.
- Morphometry adds `pct_midline` (left + right + midline = 100) and
  `header_zooms_mismatch`.
- NaN/inf voxels are treated as 0 (previously +inf counted as lesion).
- Disconnection index format v2 (no pickle). Rebuild v1 indexes with
  `tools/build_discon_index.py`; loading one raises `IndexFormatError`.
- Custom atlases are validated at registration (`InvalidAtlasError` for negative or
  non-integer labels; float labels are rounded, not truncated). Dict labels keep `Lobe` and
  the other schema fields.
- Default atlases are now `aal, schaefer, dk, jhu, arterial`.
- The "Wernicke's area" functional label is not used: AAL and DKT don't split the superior
  temporal gyrus, so it would overclaim.

### Fixed
- CI wheel job checked for atlas files that no longer exist.
- Region names were written unescaped into the group report (HTML injection).
- `ReportStyle` colours could break out of the `<style>` block.
- `lqtpy group one.nii.gz` crashed; `atlases="aal"` was iterated per character.
- Bool `(array, affine)` input raised a misleading `GeometryError`.
- Empty disconnection index query raised `IndexError`.
- CLI printed tracebacks for bad input.

### Performance (warm, 1 mm lesion; `python tools/bench.py`)
| | 0.4.0 | 0.5.0 |
|---|---|---|
| `lesion_stats` | 201 ms | 20 ms |
| `atlas_overlap` (resampled atlas) | 120 ms | 26 ms |
| five API calls on one lesion | 647 ms | 49 ms (prepared once) |
| `analyze_single`, no plots | 367 ms (3 atlases) | 143 ms (5 atlases + QC + description) |
| `analyze_single`, no plots, all 0.5.0 modules | | 390 ms (+ border zones, distances, cross-sections, mirror) |
| `analyze_single` with the full HTML report | ~8 s (matplotlib/nilearn figures) | 0.8 s (numpy figures) |
