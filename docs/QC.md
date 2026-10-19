# Mask quality checks (`lqtpy.qc`)

`lqtpy.qc` looks for lesion masks that would give misleading numbers: drawn on some slices only,
left as an outline, drawn on the wrong plane, spilling outside the brain, not in MNI space, and
so on. Each problem is reported as a **finding**. A finding says:

- what is wrong;
- how much of it there is;
- how bad that makes it;
- where it is;
- how it could be fixed.

The mask as a whole also gets a **verdict** and a **score**. Nothing is ever changed unless you
ask.

```python
from lqtpy import qc

res = qc.assess("lesion.nii.gz")          # QCResult
res.verdict, res.score                    # 'fix before use', 45
for f in res.findings:
    print(f.label, f.message, f.locations[:1])
    # Problem, major  87% of the lesion lies in one coronal slab ...  [{'kind': 'slice', 'plane': 'coronal', 'index': 81, ...}]

qc.check("lesion.nii.gz")                 # just the findings (the list older code expects)
```

```bash
lqtpy qc lesion.nii.gz                    # plain text
lqtpy qc lesion.nii.gz --json             # the QCResult as JSON (schema below)
lqtpy qc lesions/ --json summary.json     # every mask plus a summary
lqtpy qc lesion.nii.gz --tier fast        # skip the slower checks
lqtpy qc lesion.nii.gz --context ctx.json # drawing context from a drawing tool
```

## Words, grades and verdicts

Every finding keeps one of lqtpy's four quality words and, when it is a real problem, a grade
for its extent:

| Word | Meaning |
|---|---|
| **OK** | no findings |
| **Note** | informational, or too small to matter (below the minor band) |
| **Check** | graded **minor** or **moderate** |
| **Problem** | graded **major**, or the mask is in the wrong space, inverted or on the wrong side |

The grades mean:

- **minor**: results barely change;
- **moderate**: worth fixing before the results are used;
- **major**: the numbers would be wrong.

A finding's `label` reads "Check, moderate", "Problem, major" or "Note", and its `where` reads
"axial slice 107" or "at -40, 10, 5 mm" (empty when it has no location). Both are in
`to_dict()`, so a tool can show lqtpy's wording as it is.

`res.needs_attention` is true when any finding is a Check or a Problem. Ask it before
interrupting someone (a save dialog, a warning): Notes alone never need attention.

The **verdict** sums up the mask:

| Verdict | When |
|---|---|
| usable | no Check or Problem, or at most two minor findings |
| usable with caution | a moderate finding, or three or more minor ones |
| fix before use | any Problem |
| unusable | not in MNI space, or inverted: every number would be wrong |

The **score** runs from 100 (no findings) down to 1. Each major finding costs 40 points, each
moderate one 15, each minor one 5 and each Note 1. An unusable mask scores 0. Use the score to
choose between copies of the same lesion (`qc.rank`), not as a clinical measure.

## Findings

The category says what kind of problem a finding is:

- `space`: the mask is in the wrong place;
- `drawing`: the lesion was drawn badly;
- `extra`: voxels that don't belong to the lesion;
- `values`: the voxel values are odd;
- `plausibility`: the lesion is unusual rather than wrong. These findings never go above
  moderate.

**Band** is the metric the grade comes from, with the minor, moderate and major lower bounds.
Below the minor bound the finding is a Note.

| id | Category | What it means | Band (metric: minor / moderate / major) | Locations | Fix |
|---|---|---|---|---|---|
| `sparse_slices` | drawing | Drawn on some slices only: empty or nearly empty slices between drawn ones (periodic, or wide gaps between thin slabs), or three or more separate thin sheets in one plane. The space between two separate lesions (one in each hemisphere, say) is not a gap: they taper toward it, slabs end in flat faces. One or two empty slices (about 2 mm) between large, overlapping slices are always a gap, however round the pieces: a slice forgotten in a complete lesion. Those are graded on what filling just them adds, and the fix fills just them. With a drawing context, the exact undrawn slices. | `missing_pct` (share of the lesion's slices missing): 1 / 10 / 25 | the gap slices | `qc.repair_gaps` (interpolate) |
| `flat` | drawing | Drawn on one slice (or a thin slab) only, including a warped 5 mm slice, a wide plate up to 12 mm thick that ends in two flat faces, and a lesion that is mostly one slab with pieces attached. A flat part must be at least 12 mm across (lacunes aren't flat), and plates count only when they hold half the lesion (a small second lesion seen on one thick slice is ordinary). | `flat_pct` (share of the lesion in the slab): 5 / 25 / 50 | the slab's slice | redraw |
| `hollow` | drawing | Outline only, not filled. | `missing_pct` (volume that filling adds): 5 / 15 / 30 | the outlined slices | `qc.fix(…, ["hollow"])` |
| `holes` | drawing | Enclosed cavities or "swiss cheese" (pores). Cavities that are CSF or ventricle don't count, and it stays quiet when `sparse_slices` or `hollow` already explains the empty space. | `missing_pct` (share of the filled volume missing): 2 / 10 / 25 | hole centres | `qc.fix(…, ["holes"])` |
| `strokes` | drawing | Marks one voxel thick off the plane the lesion was drawn in: flat sheets (at least 200 mm² and 6 mm wide), or strips at least 15 mm long where a stroke crosses or leaves the lesion. | `affected_pct`: 0.5 / 5 / 15 | the stroke slices | `qc.clean(strokes=True)` |
| `box` | drawing | Rectangular blocks, as a rectangle tool leaves (the bounding box is 85% or more filled). | `affected_pct`: 5 / 25 / 60 | block centres | redraw |
| `outside_brain` | extra | Lesion outside the brain mask, beyond a 5 mm rim tolerance. A rim covering 1% or more of the lesion is a Note. Anything more than 10 mm from the brain surface is at least Check, minor, however small: stray marks, not overdrawing. Image-edge fragments are not counted here (they are `edge_artifact`). | `outside_pct`: 1 / 5 / 15 | outside chunks | `qc.clean(outside=True)` |
| `misregistered` | space | 25% or more lies outside the brain, attached to the lesion: the mask looks shifted, not overdrawn. Always Problem, major. | none | outside chunks | check the registration |
| `edge_artifact` | extra | Fragments touching the image edge. Always at least minor. | `affected_pct`: 0 / 5 / 20 | fragment centres | `qc.clean(edge=True)` |
| `specks` | extra | Stray specks. Moderate when they make up 5% or more of the lesion. | `n_specks`: 4 / 100 / – | speck centres | `qc.clean(specks=True)` |
| `fragmentation` | plausibility | Shattered into many parts (Note at 50 or more). | `n_parts`: 149 / 400 / – (moderate at most) | none | check the drawing |
| `values` | values | Several label values, or a probability map whose volume depends on the threshold (0.3 / 0.5 / 0.7). | `volume_spread_pct`: 0 / 20 / – | none | binarise deliberately |
| `inverted` | space | Background filled instead of the lesion (a volume near the whole image). Replaces the findings it causes (outside the brain, edge, box). | fatal | none | invert the mask |
| `not_mni` | space | Not in MNI space. | fatal | none | register to MNI152 |
| `side_mismatch` | space | The lesion is on the other side from `expected_side`. | Problem, major | none | check for a left/right flip |
| `plausibility` | plausibility | Tiny (under 50 mm³), huge (over 400 cm³), almost only CSF/ventricle, or a mirror copy across the midline (92% or more overlap when flipped; bilateral lesions reach 70–90% by nature). | fixed grade, at most moderate | none | check |
| `header` | space | Header problems: qform and sform disagree (reported only when this moves the lesion by more than 2 mm), 4D data, a misnamed compression. | Check, minor or Note | none | fix the header |
| `input` | values | The file couldn't be read as expected. | – | none | – |

Every finding also carries:

- **`extent`**: the named measurements, with units in the names (`outside_mm3`,
  `max_depth_mm`, `n_gap_slices`, `thickness_mm`, …).
- **`affected_pct`**: the estimated share of the lesion affected, a common number across
  types.
- **`evidence`**: the raw detector output (unchanged from older versions).

## Locations ("Go to")

`f.locations` lists where to look, most important first. Each location is a dict:

| Key | Meaning |
|---|---|
| `kind` | `"slice"` (a whole slice: gaps, slabs, strokes) or `"point"` (a centre: holes, chunks, fragments) |
| `ijk` | voxel coordinates **on the mask's own grid** (the file you passed in) |
| `mm` | world coordinates (scanner/MNI mm) |
| `axis`, `plane`, `index` | slices only: array axis 0/1/2, `"sagittal"`/`"coronal"`/`"axial"`, and slice index on that axis |
| `n_voxels` | points only: how many voxels this piece has |

A viewer can jump straight to `mm` (a crosshair) or to slice `index` along `axis`.

`qc.problem_mask(lesion, finding)` returns a boolean mask of the affected voxels on the
lesion's grid, for highlighting or erasing as one undoable edit. `qc.PROBLEM_KIND[id]` says
whether those voxels should be `"remove"`d (specks, strokes, outside, edge), `"add"`ed (holes,
hollow, gaps) or only `"inspect"`ed.

## Fixing

`f.fix` is a machine-readable hint, for example
`{"call": "qc.repair_gaps", "kwargs": {"axis": 2}}` or
`{"call": "qc.clean", "kwargs": {"strokes": true}}`.

`qc.fix(lesion, ["specks", "holes"])` applies the fixable findings (`qc.FIXABLE`). It returns
the new image and a report:

- `applied` and `skipped`;
- `voxels_added` and `voxels_removed`;
- `added` and `removed` masks, so a caller can show the difference or apply it as an undoable
  edit.

It never writes files and never changes the input. On the synthetic library, every fixable
finding is gone after `fix`.

## Drawing context

A drawing tool knows things a saved mask doesn't show. Pass them as `context=` (a dict, JSON or a
`qc.DrawingContext`). Every field is optional. Unknown keys are ignored and bad values are
dropped, never raised.

```python
qc.assess(mask, context={
    "drawn_slices": {2: [40, 42, 44, 48]},     # axis -> slices the user drew on (mask grid)
    "interpolated_slices": {2: [41, 43]},      # filled by interpolation: never reported as gaps
    "drawing_axis": 2,                         # the plane drawn on: strokes in it are not flagged
    "base": "mni_template", "space": "mni",    # what the mask was drawn over
    "expected_side": "left",
    "source": "drawn",
    "labels_are_colours": True,                # pen colours of one lesion: no `values` finding
})
```

With `drawn_slices`, `sparse_slices` reports the exact undrawn slices inside the lesion's span
(minus the interpolated ones) instead of inferring them.

## Tiers

| Tier | What runs | Typical time |
|---|---|---|
| `header` | Header only; no voxel data is read. It checks grid and space (recognises the MNI grids, including 207×256×215 at 0.74 mm and 99×117×95 at 2 mm, by shape, voxel size and affine), the qform/sform distance, dtype, 4D data and misnamed compression. | under 5 ms |
| `fast` | Every voxel check except tissue and symmetry (plausibility). | 35 ms (1 mm grid), 90 ms (0.74 mm drawing) |
| `full` (default) | Everything. | 45 ms (1 mm grid), 220 ms (0.74 mm drawing) |

Times are from `python tools/bench.py` on a desktop PC, with warm caches.

## Folders and copies

- `qc.assess_folder(folder, tier=..., jobs=...)` returns `{"results": [...], "summary": {...}}`.
  The summary counts findings by id and grade, and masks by verdict.
- `qc.check_folder(folder)` (older) returns one row per mask, now with `verdict` and `score`.
- `qc.rank(copies)` orders copies of one lesion from best to worst: by verdict, then score, then
  voxel size (the finer grid first).

## JSON schema

`QCResult.to_dict()` / `to_json()`, and `lqtpy qc --json`:

```json
{
  "schema": "lqtpy.qc",
  "qc_version": "1.1",
  "tier": "full",
  "verdict": "fix before use",
  "score": 45,
  "worst": "error",
  "needs_attention": true,
  "findings": [
    {"id": "flat", "severity": "error", "message": "...", "evidence": {}, "suggestion": "...",
     "category": "drawing", "grade": "major", "extent": {"flat_pct": 87.0, "thickness_mm": 7.4},
     "affected_pct": 87.0, "label": "Problem, major", "where": "coronal slice 81",
     "locations": [{"kind": "slice", "axis": 1, "plane": "coronal", "index": 81,
                    "ijk": [103, 81, 97], "mm": [-11.0, -51.0, 1.0]}],
     "fix": null}
  ],
  "context": {},
  "options": {},
  "seconds": 0.21,
  "source": "lesion.nii.gz"
}
```

`severity` is `"info"` / `"warn"` / `"error"` (Note / Check / Problem), as before.

`analyze_single(...)` results carry the findings under `"qc"`, plus `"qc_verdict"`,
`"qc_score"` and `"qc_version"`.

## Caching: key on `qc_version`

`QC_VERSION` changes whenever QC rules, thresholds or this JSON shape change. It is separate from
the package version. A tool that stores QC results should key its cache on `qc_version`: then a
new lqtpy release that doesn't touch QC doesn't force a re-run, and one that does, does.

| qc_version | lqtpy | Change |
|---|---|---|
| 1.0 | 0.6.0 | Graded findings, locations, fix hints, verdict and score, tiers, drawing context; new checks (holes, box, values, inverted, misregistered, fragmentation, plausibility) |
| 1.1 | 0.6.1 | A short, clear gap through a large lesion is a gap even when the pieces either side are round; such gaps are filled slice by slice (`repair_gaps(slices=...)`); lesion more than 10 mm outside the brain is at least Check, minor |

## How the checks were tested

- **Synthetic library** (`tests/qc_zoo.py`; full run with `python tools/qc_zoo_report.py`):
  - problem cases: every type at several extents, lesion sizes and positions, and native slice
    thicknesses from 1 to 7 mm, warped and unwarped, on both lab grids;
  - 13 clean cases: cortical ribbons, lacunes, large MCA infarcts, multifocal small-vessel
    disease, jagged slice-by-slice drawings.

  Every problem case is detected and graded at least as badly as expected. No clean case gets a
  Check or Problem. Locations land within 15 mm of the true problem. The fast subset runs in CI
  (`tests/test_qc_graded.py`).
- **Real masks** (`tools/qc_survey.py`) on a clinical database of 1,335 MNI-space masks. The
  survey prints how often each finding and grade occurs, and compares that with a baseline. It
  names masks by a short SHA-1 of the file name only.
- **Labelled review** (`tools/qc_review.py`): a page of 200 masks, stratified by finding and
  grade, to label by eye. `tools/qc_review.py score` then reports precision, recall and grade
  agreement, on a tuning half and a validation half.

### Labelled review, first round

The first 200 labels were made by an AI assistant (Claude), not by a clinician, so they are a
proxy. They were made blind to lqtpy's findings, from detailed per-mask sheets and a rubric
written beforehand; 21 were revised after a second look. Rules were changed on the tuning
half only (106 masks), then checked once on the validation half (94 masks).

"Does this mask have a problem?" (Check or Problem, against the labels):

| | Tuning half | Validation half |
|---|---|---|
| Before | precision 0.91, recall 0.88 | precision 0.93, recall 0.91 |
| After | precision 0.99, recall 0.89 | precision 0.94, recall 0.92 |

The gain on the validation half is small: most of the tuning-half gain did not carry over as a
change in which masks are flagged. What did carry over is the kind of finding:

- stray strokes: recall 0.55 to 0.82 on the validation half (precision 1.00 to 0.82). A first
  version flagged far more masks across the database; a sample outside the review set showed
  it was calling the slice rims of thick-slice drawings strokes, and it was made stricter.
  Without a drawing context, a stroke through an otherwise smooth lesion can still be missed
  (its own plane is taken for the drawing plane);
- masks labelled majorly wrong but rated usable or usable with caution: 5 of 15 to 2 of 15;
- edge fragments double-reported as outside the brain: 27 masks to none;
- mirror-copy false alarms on bilateral lesions: 4 to none.

Not improved, and still open:

- `box` never fires on the 16 masks labelled as blocks (mostly outlines repeated over many
  slices, which is a different shape from a rectangle-tool box);
- `flat` and `sparse_slices` are often swapped for each other on stacked thin sheets (as one
  family: precision 0.81, recall 0.83 on the validation half, unchanged);
- small specks and thin rims are deliberately Notes, where the labels often said minor.

Where the type matched, the grade matched exactly on 82% of findings and was one step off on
14%. The grade bands were left as they were.
