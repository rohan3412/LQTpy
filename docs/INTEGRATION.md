# Using lqtpy from another program

For applications that bundle lqtpy: a drawing tool such as MRLatte, a lesion
registry, a pipeline. Everything here is opt-in: calling none of it changes
nothing. Function-level detail is in the API reference (`python tools/build_docs.py`).

## Stability promises

| What | Promise |
|---|---|
| `lqtpy.capabilities()["features"]` names | only ever added |
| `finding.action` names (`lqtpy.qc_advice`) | only ever added, never renamed |
| `QCResult.to_dict()` | versioned by `qc_version`; changes only when QC rules or the JSON shape change |
| `lqtpy.summary()` | `summary_version` "major.minor": a minor bump only adds keys |
| `lqtpy.compare()` | `compare_version` |

Key caches of QC results on `qc_version`, not on the package version.

## Which features does this lqtpy have?

```python
caps = lqtpy.capabilities()          # cheap: no image read, nothing heavy imported
caps["lqtpy_version"], caps["qc_version"]
if caps["features"].get("qc.fix_plan"):
    ...
```

A feature is `True` only when it works in this installation (for example
`xlsx` needs `openpyxl`). Prefer this over `hasattr` checks: lqtpy may be an
older or newer version than the application was built against.

## Quality checks while drawing: `qc.Session`

Keep one session per open drawing in the backend, and send only what changed:

```python
s = lqtpy.qc.Session(shape, affine, context={"drawing_axis": 2})
s.set_slice(2, 107, slice_array)          # after a stroke on axial slice 107
s.apply(added=ijk_added, removed=ijk_removed)   # or voxel lists / full-grid masks
s.set_mask(whole_array)                   # e.g. after a large Undo
result = s.check(tier="fast")             # QCResult, or None for an empty drawing
```

- `check()` returns the same result as `qc.assess()` on a saved file of the
  same voxels, and the previous result instantly when nothing changed.
- A full check of a 20 cc lesion takes about 35 ms; the session saves
  re-sending and re-reading the volume on every stroke.
- The session has no file, so it can't report header problems (`header`,
  `values`). Check the saved file once on Save.
- Thread-safe: edits and checks may come from different threads.

## Words and buttons: advice and actions

```python
for f in result.findings:
    f.label       # "Check, moderate"
    f.where       # "axial slice 107"
    f.advice      # plain words for the person fixing the mask
    f.action      # "interpolate_gaps", "remove_specks", "fill_outline", ...
result.to_dict(advice=True)   # adds {"action", "text", "fix_id", "adds_voxels"} per finding
```

`f.suggestion` is unchanged and may name lqtpy functions (it suits
programmers); show `f.advice` to clinicians. Map `f.action` to your own
controls. Actions with a `fix_id` can be done by `qc.fix`; `redraw`,
`review`, `check_registration`, `check_side` and `check_file` need a person.

## Fix previews: `qc.fix_plan()`

```python
for e in lqtpy.qc.fix_plan(mask):
    e["id"], e["action"], e["advice"]
    e["adds_voxels"], e["voxels_added"], e["voxels_removed"]
    e["added"], e["removed"]          # boolean masks on the mask's own grid
```

Each entry is that one fix alone, worked out from the original mask, so a
preview can show it and the tool can apply it as one undoable edit. After
applying one, plan again: fixes can overlap (filling gaps can also fill a
hole). Nothing is written.

## One summary per lesion: `lqtpy.summary()`

```python
s = lqtpy.summary("lesion.nii.gz")        # runs the analysis in memory, writes nothing
s = lqtpy.summary(analyze_single_result)  # or reuses a result
```

Plain JSON with every key always present (`null` when a part didn't run) and
no file paths. Check `schema == "lqtpy.summary"` and the major version.

## Two masks of one lesion: `lqtpy.compare()`

```python
c = lqtpy.compare(baseline, follow_up)
c["volume"]["change_pct"], c["overlap"]["dice"], c["centroid_shift_mm"]
c["regions"]["gained"], c["regions"]["lost"], c["regions"]["changed"]
```

Exact on a shared grid; on the 1 mm MNI grid otherwise. Both masks must be
in the same space.

## Native-space lesions: ANTs transforms

Pass the forward transforms of an ANTs registration (fixed = MNI template,
moving = the patient's scan) in ANTs order:

```python
reg = ants.registration(fixed=mni, moving=t1, type_of_transform="SyN",
                        write_composite_transform=False)
img = lqtpy.native.warp_to_mni(lesion_native, reg["fwdtransforms"])
lqtpy.analyze_single(lesion_native, transform=reg["fwdtransforms"])
```

```bash
lqtpy single lesion_native.nii.gz --transform reg_1Warp.nii.gz --transform reg_0GenericAffine.mat
```

Checked against ANTsPy's own `apply_transforms`: identical masks, also with
an exaggerated warp. Not supported: composite `.h5` files and inverse warps.
A 4 x 4 text matrix (native mm to MNI mm) and SPM-style deformation fields
work as before.

## Many masks

- Group runs: `n_jobs="auto"` picks the number of worker processes; if
  workers can't start, the run finishes in one process instead of failing.
- `lqtpy.qc_tools.assess_folder(folder, jobs="auto")` for QC of a folder.
- BIDS datasets: `lqtpy group DATASET --bids` (see `lqtpy.bids`).

## Proving a change leaves outputs unchanged

`tools/golden.py` records every output for a fixed set of inputs and
compares two recordings; see its docstring. Real masks can be added with
`--masks FOLDER` (kept outside the repository; recordings name them by a
hash only).
