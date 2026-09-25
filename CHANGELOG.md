# Changelog

## 0.5.0 (unreleased)

Implements Phases 0–7 of [docs/ROADMAP.md](docs/ROADMAP.md).

### New
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
