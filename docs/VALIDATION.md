# Geometry and equivalence validation

This document records the read-only validation run that established two things about
`lqtpy`'s streamline-disconnection machinery: (1) the corrected coordinate frame for the
HCP-842 tractography atlas is right, and the naive frame is not, and (2) `lqtpy`'s
streamline-selection rule matches DIPY's `target()` exactly. Nothing in the validated
codebase (`lqtpy/`) was modified to produce these results; the check reads
`lqtpy/tracts.py`, `lqtpy/_discon_index.py`, and `lqtpy/data/hcp842_tracts.npz`, plus the
HCP-842 atlas's own distribution files (the `HCP842_tractography.nii.gz`/`.txt` header and
the 70 `*.trk.gz` bundles + `tract_info.csv`), which are not part of this repository (see
"Building the index" in the main README).

## The ~120 mm trap

The HCP-842 atlas's individual tract `.trk.gz` files store streamline points in an internal
(integer) voxel grid that is **already the atlas's own 157×189×136 grid** — but the trk
header's `voxel_to_rasmm` field records **zero translation** instead of the true origin
`(78, 76, -50)`. A loader that trusts the trk header at face value (e.g. treating the trk's
world space as already RAS-mm-equivalent to the atlas) places every streamline about
`sqrt(78² + 76² + 50²) ≈ 119.8 mm` away from where it belongs — outside the 157×189×136 grid
entirely for every tract tested. `lqtpy._discon_index`/`tools/build_discon_index.py` apply the
corrected shift once, at index-build time (see the "THE ~120 MM TRAP" warning at the top of
both files); this document is the evidence that the correction is right and the naive
approach is not.

## Gate criterion (stated before looking at results)

> The hypothesis passes if the median hypothesis Dice is clearly high, the naive-frame Dice
> is near zero for essentially every tract, and no matched tract has a hypothesis Dice
> anywhere near the naive control.

## Verdict: PASS

- Median hypothesis Dice = **0.7846** (n=62 matched tracts), minimum = **0.3866** (RST_R).
- Naive-frame Dice = **exactly 0.0 for all 62 matched tracts** — every naive-frame
  rasterization lands entirely outside the (157, 189, 136) grid, not just "low."
- Both axis-flip controls (x-flip, y-flip) = **exactly 0.0 for all 62 tracts**, same reason.
- No matched tract's hypothesis Dice (minimum 0.3866) comes anywhere near the naive/flip
  controls (maximum 0.0 across all three, all tracts) — the gap is total, not marginal.
- Header confirmation: the atlas's actual affine and shape **exactly equal** the hypothesized
  values (`TRACT_AFFINE`/`TRACT_SHAPE` in `lqtpy/_discon_index.py`), and
  `lqtpy/data/hcp842_tracts.npz`'s stored affine/shape also match exactly.
- Streamline-count check: **0 mismatches** across all 70 bundles against `tract_info.csv`.
- `target()` equivalence: **0 disagreement** between DIPY's `target()` and `lqtpy`'s plain
  voxel-lookup implementation, for both hit and miss spheres, on all 3 tested bundles.

The low end of the correct-frame Dice distribution (0.39–0.92) is explained below by
streamline-count sparsity in the reference bundles, not by any geometric problem — centroid
offsets stay under ~4 mm throughout, including for the worst-Dice tracts.

One correction to the source atlas's own documentation: it describes "71 bundles," but the
distribution actually contains 70 `.trk.gz` files and `tract_info.csv` has exactly 70 rows,
fully self-consistent (every row has a matching file and vice versa) — a discrepancy in the
atlas's stated count, not a missing/extra file.

## Header confirmation

```
HCP842_tractography.nii.gz  affine:
[[ -1.   0.   0.  78.]
 [  0.  -1.   0.  76.]
 [  0.   0.   1. -50.]
 [  0.   0.   0.   1.]]
shape: (157, 189, 136, 80)      <- exactly matches TRACT_AFFINE / TRACT_SHAPE

lqtpy/data/hcp842_tracts.npz    affine: identical to TRACT_AFFINE
                                 shape:  (157, 189, 136), identical to TRACT_SHAPE

AF_L.trk.gz header:
  voxel_to_rasmm = diag(-1,-1,1), zero translation   <- matches the naive-frame premise
  dimensions     = [157, 189, 136]                    <- same shape as the atlas grid
  affine_to_rasmm (nibabel) = identity
raw AF_L point range: x in [-140.3, -104.2], y in [-144.4, -44.8], z in [45.9, 101.9]
  -> adding [78,76,-50] to x gives [-62.3,-26.2], the expected left-arcuate anchor.
```

## Per-tract summary (worst 12 of 62 matched tracts, by hypothesis Dice)

Streamlines were densified along each segment at ~0.5 mm spacing before voxel-rounding, to
avoid gaps on long inter-point steps. The atlas frames are already binary, so the requested
threshold sweep (0.1/0.3/0.5/0.7/0.9) is degenerate — every threshold in (0,1) produces an
identical binarization and Dice (reported as-is rather than glossed over).

| tract | n streamlines | hyp Dice | naive/flipx/flipy Dice | centroid offset (mm) |
|---|---:|---:|---:|---:|
| RST_R | 6 | 0.3866 | 0.0 / 0.0 / 0.0 | 0.86 |
| DLF_R | 4 | 0.4429 | 0.0 / 0.0 / 0.0 | 1.09 |
| RST_L | 5 | 0.4539 | 0.0 / 0.0 / 0.0 | 0.62 |
| DLF_L | 3 | 0.4806 | 0.0 / 0.0 / 0.0 | 0.66 |
| MLF_R | 11 | 0.5410 | 0.0 / 0.0 / 0.0 | 1.49 |
| MLF_L | 10 | 0.5509 | 0.0 / 0.0 / 0.0 | 0.75 |
| LL_L | 76 | 0.5616 | 0.0 / 0.0 / 0.0 | 0.96 |
| CTT_R | 21 | 0.6290 | 0.0 / 0.0 / 0.0 | 0.79 |
| CTT_L | 15 | 0.6511 | 0.0 / 0.0 / 0.0 | 0.82 |
| CS_R | 2636 | 0.6549 | 0.0 / 0.0 / 0.0 | 2.30 |
| CS_L | 2921 | 0.6674 | 0.0 / 0.0 / 0.0 | 3.19 |
| PC | 705 | 0.6707 | 0.0 / 0.0 / 0.0 | 0.80 |
| *(remaining 50 tracts)* | — | up to 0.9195 (CB_R) | all 0.0 | all ≤ 3.87 |

**Median hypothesis Dice: 0.7846. Minimum: 0.3866 (RST_R). Maximum naive/flip Dice across
all 62 tracts: 0.0.**

Every tract below Dice 0.66 has an extremely small reference streamline count (3–76
streamlines) *except* CS_L/CS_R (2636–2921 streamlines). In every low-Dice case the centroid
offset between the rasterized bundle and the atlas frame stays under ~1.5 mm, and the
rasterized-voxel count is close to the atlas frame's own voxel count (e.g. RST_R: 175 vs 213
voxels). That combination — low Dice, small centroid offset, comparable voxel counts — is the
signature of a density/sampling artefact (the atlas frame is pooled across hundreds of
subjects; the reference bundle here has only a handful of streamlines from one representative
subject), not a geometric problem. A translation or orientation error would instead produce
either a large centroid offset or complete non-overlap, neither of which is observed.

CS_L/CS_R have the largest centroid offsets among matched tracts (2.30/3.19 mm) and the
lowest Dice among the well-sampled tracts, consistent with the same density-artefact
explanation (cortico-striatal projections are anatomically diffuse/fanning) rather than a
naming or geometric error — and their naive/flip controls are still exactly 0.0.

No matched tract shows a large centroid offset combined with low Dice; the largest centroid
offsets overall are TPT_L (3.87 mm) and FPT_L (3.78 mm), both still well-matched (Dice 0.71,
0.74).

## Unmatched bundles and frames

70 `tract_info.csv` rows map onto 80 atlas frames: **62 matched, 8 unmatched CSV rows** (and
correspondingly 18 unmatched frames).

Non-trivial mapping decisions, each independently confirmed by Dice against its assigned
frame (not taken on naming convention alone):

- **CS_L/CS_R vs CST_L/CST_R** — the atlas's own pathway-description text labels *both*
  pairs "Corticospinal Tract," which cannot both be right. The atlas has two distinct named
  frames, `Cortico_Striatal_Pathway_{L,R}` and `Cortico_Spinal_Tract_{L,R}`. Standard
  short-code convention (CS = Cortico-Striatal, CST = CorticoSpinal Tract) resolves the
  ambiguity, confirmed by Dice: CS_L/R → 0.667/0.655; CST_L/R → 0.810/0.829 — both pairs land
  on plausible, non-degenerate Dice with their assigned frame.
- **PPT_L/PPT_R** — mislabelled "Pontine Tegmentum Tract" in the source description; no such
  frame exists. PPT is the standard abbreviation for **Parietopontine Tract**. Dice
  0.788/0.806 confirms.
- **TPT_L/TPT_R** — mislabelled "Temporoparietal Tract"; TPT is the standard abbreviation for
  **Temporopontine Tract**. Dice 0.711/0.686 confirms.
- **EMC_L/EMC_R** — mislabelled "External Medullary Lamina" (a real but different structure
  with no matching frame). The only plausible frame is **Extreme_Capsule**, the standard
  bundle abbreviated EMC in this atlas family. Dice 0.721/0.778 confirms.
- **CB_L/CB_R** — labelled "Cerebellar Peduncles," but ICP/MCP/SCP are already separately and
  unambiguously matched (individual peduncles). The only remaining candidate is the combined
  **Cerebellum_L/R** frame. Dice 0.910/0.920 — the highest values in the whole set — strongly
  confirms this match.

(These four labelling corrections are also recorded in
`lqtpy/data/tract_disconnection_names.csv`'s `corrected`/`note` columns, which
`lqtpy.datasets.get_tract_display_names()` serves.)

Unmatched, all with a stated reason rather than silently dropped:

| item | reason |
|---|---|
| CCAnterior, CCCentral, CCMidAnterior, CCMidPosterior, CCPosterior (5 CSV rows) | Atlas has a single undivided `Corpus_Callosum` frame; no frame-level subdivision exists to disambiguate. |
| OPT_L, OPT_R | Labelled "Optic Tract"; no `Optic_Tract` frame exists (only `Optic_Radiation_L/R`, already matched to OR_L/OR_R). |
| V | Labelled "Vestibular Pathway"; no frame in the atlas names a vestibular pathway. |
| Occipitopontine_Tract_L/R frames | No corresponding row in `tract_info.csv`. |
| `Corpus_Callosum` frame | Many-to-one with the 5 CC subdivisions above. |
| `Vermis` frame | No corresponding row. |
| 12 cranial-nerve frames (II/III/IV/V/VII/VIII/X, L/R) | No cranial-nerve rows among the 70 `tract_info.csv` entries. |

## In-grid orientation controls

The naive, flip-x, and flip-y controls above all score exactly 0.0, but they do so by
pushing points **outside the grid** — that proves the translation is right, but not that the
orientation is, because a wrong frame loses by going out of bounds rather than by competing
on overlap. These controls stay inside the grid instead: own frame vs. contralateral frame,
an in-grid left/right mirror, and ±3 mm shifts on each axis.

| tract | own frame | contralateral frame | L/R mirror (in grid) | ±3 mm shift, x−/x+/y−/y+/z−/z+ |
|---|---|---|---|---|
| AF_L | 0.831 | 0.000 | 0.000 | 0.630 0.489 0.679 0.618 0.568 0.635 |
| AF_R | 0.814 | 0.000 | 0.000 | 0.482 0.432 0.607 0.570 0.515 0.558 |
| UF_L | 0.840 | 0.000 | 0.000 | 0.582 0.531 0.693 0.613 0.469 0.565 |
| ILF_L | 0.814 | 0.000 | 0.000 | 0.497 0.376 0.705 0.606 0.544 0.637 |
| IFOF_L | 0.821 | 0.000 | 0.000 | 0.551 0.465 0.672 0.625 0.520 0.637 |
| OR_L | 0.701 | 0.000 | 0.000 | 0.299 0.253 0.510 0.356 0.275 0.527 |
| CST_L | 0.810 | 0.001 | 0.002 | 0.476 0.344 0.639 0.578 0.613 0.735 |
| CB_L | 0.910 | 0.000 | 0.000 | 0.847 0.796 0.831 0.809 0.778 0.813 |

- **Left/right is confirmed**: bundles don't overlap the contralateral frame, and an in-grid
  mirror scores zero.
- **The translation is a peak, not just a nearby value**: every ±3 mm shift on every axis
  lowers Dice for every tract tested.

Verdict unchanged: **PASS**, with orientation evidence that doesn't depend on out-of-bounds
losses.

## DIPY `target()` equivalence

Tested on AF_L (large, well-matched), AC (commissural), and UF_L (a smaller well-matched
tract). For each: streamlines shifted to the corrected frame, a "hit" sphere (radius 4
voxels) centered on an actual point on the tract's longest streamline, and a "miss" sphere
offset 60 voxels away, confirmed (via the tract's own rasterization) to have zero true
overlap with the bundle.

| bundle | n streamlines | hit: DIPY count | hit: plain count | hit sym. diff. | miss: DIPY count | miss: plain count | miss sym. diff. |
|---|---:|---:|---:|---:|---:|---:|---:|
| AF_L | 2358 | 1801 | 1801 | 0 | 0 | 0 | 0 |
| AC | 2124 | 2124 | 2124 | 0 | 0 | 0 | 0 |
| UF_L | 1112 | 223 | 223 | 0 | 0 | 0 | 0 |

No disagreement in any of the 6 comparisons (3 bundles × hit/miss). Both miss spheres
correctly select zero streamlines by both methods, and every hit-sphere selection matches
exactly between DIPY's `target()` (via `dipy.tracking.utils._to_voxel_coordinates`, which
truncates `inv(affine) @ point + 0.5`, i.e. round-half-up) and `lqtpy`'s plain implementation
(`np.round`, round-half-to-even). The two rounding conventions only disagree on points that
land exactly on a `.5` voxel boundary; none of the tested streamlines happened to do so, so
this run does not exercise that edge case — a caveat, not a contradiction. Given all 6
comparisons agree exactly, this is a clean, non-marginal pass for `target()` equivalence, but
not an exhaustive proof of identical rounding at every possible boundary voxel.

## Bottom line

**PASS.** The corrected frame's affine and shape are confirmed byte-for-byte against both the
atlas's own header and `lqtpy/data/hcp842_tracts.npz`. All 70 bundle streamline counts match
`tract_info.csv` exactly. Under the corrected frame, 62 of 70 bundles map onto a named atlas
frame with median Dice 0.78 (min 0.39, explained by streamline-count sparsity, not
geometry); under the naive frame and both orientation-flip controls, Dice is exactly 0.0 for
every one of those 62 tracts, because the wrong frames place the bundle entirely outside the
157×189×136 grid. `dipy.tracking.utils.target()` and `lqtpy`'s plain voxel-lookup selection
agree exactly on 3 tested bundles (AF_L, AC, UF_L), for both a true-positive and a
true-negative sphere. The 8 unmatched bundles/rows and 18 unmatched frames are explicitly
reasoned about rather than dropped.
