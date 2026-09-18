# Porting notes: the lesion-disconnection method behind the original toolkit

This is an **independent, prose description of the method**, written for someone who must
reimplement it without a DSI Studio dependency. It describes what the pipeline computes and
why, in original words. It does not reproduce source code, variable names, comments, or
docstrings from the original project's dead `src/` tree (a near line-for-line Python port of
the `jdwor/LQT` R package, never carried into this repository) or from the R package it was
ported from. See "Licence background" below for why that separation matters.

Where the original code was ambiguous, incomplete, or internally inconsistent, that is called
out explicitly as "unclear from the source" rather than guessed at.

## 1. Overall pipeline

**Inputs:**
- A binary lesion mask, already registered/normalized into the same standard space
  (MNI152-family) that the reference atlas and tractography data use.
- A pre-built reference tractography atlas: a large normative tractogram (the code uses the
  HCP-842 group-averaged tractography atlas) plus a library of ~80 pre-segmented named white
  matter tracts derived from it.
- A brain parcellation (atlas) in the same space — the original code defaults to a 100-parcel
  Schaefer/Yeo cortical parcellation with subcortical structures added, but is written to
  accept any parcellation image plus an accompanying region-label/region-group table.

**Outputs, computed in this order (later steps consume earlier ones' output files rather
than passing values in memory):**

1. **Per-tract percent disconnection** — for each of the ~80 named tracts, what fraction of
   that tract's streamlines pass through the lesion.
2. **A reference (atlas-level) parcel-to-parcel structural connectivity matrix** — computed
   once, from the *undamaged* reference tractogram, giving a baseline streamline count
   between every pair of parcels. Not patient-specific; computed once and reused.
3. **A patient-level parcel-to-parcel connectivity matrix restricted to streamlines that
   intersect the lesion** — the "damaged" version of step 2's matrix, from which percent
   disconnection per edge is derived.
4. **Parcel-level overall disconnection** — one number per brain region, aggregating that
   region's edges.
5. **Network-level and network-to-network disconnection** — the same aggregation, but
   grouped by functional network (e.g., Yeo-7) instead of individual parcel.
6. **Parcel-level grey-matter damage** — a separate, non-tractography measure: the percent of
   each parcel's own voxels that the lesion mask directly overlaps.
7. **Structural shortest-path-length (SSPL) changes** — using the parcel-to-parcel matrices
   as weighted graphs, the increase in graph-theoretic shortest path between region pairs
   caused by the lesion, which captures "indirect" disconnection (no direct edge lost, but
   the shortest route between two regions got longer).
8. A final compilation step gathers per-patient outputs from steps 1–6 across a whole cohort
   into group-level tables, keyed by patient ID.

`lqtpy` currently implements items 1 (`tract_disconnection`) and 6 (`atlas_overlap`, via
direct spatial overlap, not tractography); items 2–5 and 7–8 are not implemented — see the
Roadmap section of the main README.

## 2. How a streamline is judged "affected" by a lesion

The original pipeline does not classify a streamline as affected via its own geometry code —
it delegates that to the external tractography tool (see §4), passing the lesion mask as a
region-of-interest (ROI) filter against a pre-existing, static tractogram (the reference
atlas's streamlines are not regenerated per patient; the *same* atlas streamlines are
filtered per patient). The operational definition of "affected" is: **a streamline counts as
disconnected by the lesion if it passes through any voxel of the binary lesion mask.** There
is no partial-credit weighting by how much of the streamline intersects the lesion, and no
minimum-overlap-length threshold visible in the source — it is a binary per-streamline test
(intersects / does not intersect), aggregated by counting. This is exactly the whole-streamline,
binary rule `lqtpy.api.tract_disconnection` implements (equivalent to
`dipy.tracking.utils.target(..., include=True)`).

**Threshold values that do exist**, all supplied via configuration and used at other stages
of the pipeline (not at the single-streamline decision):
- A "spared" threshold applied to the connectivity matrix, not to individual streamlines: an
  edge (parcel pair) is treated as "structurally present" in the shortest-path graph only if
  at least a configured percentage of its original (atlas) streamline count survives the
  lesion. Two different default values for this threshold appear in different places in the
  original source (50 vs. 10) — see the correctness-trap note in §6.
- Several other thresholds are accepted as configuration (a delta-SSPL threshold, a parcel
  damage threshold, a tract disconnection threshold) but no code was found that actually
  reads and applies them to gate or flag anything — **unclear from the source** whether they
  are used by logic outside the files present, are vestigial, or the wiring was lost in
  translation. A reimplementer should not assume a specific "significance" cutoff exists
  without checking the original R package or the paper's methods section directly.

## 3. How per-tract and per-parcel percentages are computed

**Per-tract disconnection (numerator/denominator):**
- Numerator: the number of streamlines belonging to a given named tract (e.g., left IFOF)
  in the reference tractography atlas that pass through the patient's lesion mask.
- Denominator: the total number of streamlines that make up that same named tract in the
  reference atlas (a fixed, precomputed count per tract, not recomputed per patient).
- Result, times 100: percent of that tract's reference streamline population that is cut by
  this lesion. This is exactly what `lqtpy.api.tract_disconnection` returns as
  `percent_disconnected`.

**Per-edge (parcel-to-parcel) disconnection** *(not implemented in `lqtpy` yet)*:
- Numerator: streamline count between parcel *i* and parcel *j* in the reference tractogram
  that also intersects the lesion mask (i.e., the subset of the baseline i–j connection that
  passes through damaged tissue).
- Denominator: the total baseline streamline count between parcel *i* and parcel *j* in the
  undamaged reference connectivity matrix.
- Result, times 100: percent of that pairwise connection disconnected by the lesion. The
  complementary "percent spared" is `100 - percent disconnected`, computed directly rather
  than re-derived (with care taken for divide-by-zero / undefined edges, which are zeroed
  rather than left as NaN or Inf before being written out).

**Per-parcel overall disconnection** *(not implemented)*:
- Numerator: sum, over every other parcel *j*, of (baseline streamline count between parcel
  *i* and *j*) × (percent disconnected for that edge) — i.e., the total number of streamlines
  lost across all of parcel *i*'s connections.
- Denominator: sum, over every other parcel *j*, of the baseline streamline count between
  parcel *i* and *j* — i.e., parcel *i*'s total baseline structural degree (by streamline
  count, not edge count).
- This is a **streamline-count-weighted average** of the per-edge percentages, not a simple
  mean of percentages — a parcel with a few heavily-trafficked connections that are disrupted
  will score differently than one with many light connections disrupted equally. A
  reimplementer must weight by baseline streamline count, not just average the per-edge
  percentages unweighted.

**Network-level and network-to-network disconnection** *(not implemented)*: the same
numerator/denominator weighting, but with parcels first pooled into their assigned
functional-network group (per the region-group table) before summing, so the ratio is (total
lost streamlines within/between network groups) / (total baseline streamlines within/between
network groups).

**Parcel damage (a distinct, non-tractography metric, implemented as `lqtpy.api.atlas_overlap`):**
for each parcel, the percentage of that parcel's own voxels that are also lesion voxels — a
spatial-overlap fraction, computed directly from the two masks with no tractography involved.
This is reported alongside, not combined with, the disconnection measures.

## 4. Where and how the original shelled out to `dsi_studio.exe`

DSI Studio is invoked as an external command-line process (not a library call) at two
distinct points in the original pipeline, and its job in both cases is streamline
*selection/filtering against a fixed atlas tractogram*, not streamline *generation* — the
reference tractography is precomputed once and reused across all patients:

1. **Per-tract disconnection pass** (once per named tract, per patient): DSI Studio runs its
   "ana" (analysis) action against the precomputed group tractography file, scoped to one
   named tract's streamline subset, using the patient's lesion mask as the ROI, and exports a
   "stat" text file. The pipeline reads a single number out of that stat file — a streamline
   count — treats it as the numerator described in §3, and divides by a precomputed, fixed
   total-streamline-count for that tract (read from a small reference table shipped with the
   toolkit, not recomputed from the atlas at runtime).
2. **Parcel connectivity pass** (once per patient, and once overall for the reference atlas):
   DSI Studio runs the same "ana" action against the full group tractogram, this time
   exporting a full parcel-by-parcel connectivity matrix (streamline counts between every
   pair of regions in the supplied parcellation) plus a track-density-imaging (TDI) volume,
   using the lesion mask as the ROI filter for the patient-level pass, and no ROI (the whole
   atlas) for the one-time reference pass. The connectivity export is read back from a `.mat`
   file DSI Studio writes, and combined with the reference matrix as described in §3.

**What `lqtpy` reproduces, semantically, without DSI Studio (for the tract-disconnection
measure it implements):** a way to determine, per streamline in a reference/normative
tractography atlas, whether it intersects a given binary lesion mask (voxel-level test in the
shared coordinate space) — `lqtpy._discon_index` plus `tools/build_discon_index.py`. The
parcel-pair connectivity and TDI-volume pieces above are not implemented.

## 5. Coordinate-space / affine handling

- The original pipeline assumes, rather than verifies from each file's own NIfTI header in
  most places, that the lesion mask, the parcellation, and the reference tractography atlas
  are all already co-registered to the same standard space grid. Where an affine is read, it
  is read from the *lesion* image and applied to reorient/resample the lesion data itself
  before masking — but the parcellation and reference atlas are read and used as-is, without
  an explicit registration/resampling step to guarantee grid agreement with the lesion. Treat
  "inputs must already share a grid" as an implicit precondition, not something the pipeline
  enforces or corrects for you. `lqtpy` follows the same convention (see the "Conventions"
  section of the main README) but does explicitly resample the lesion mask onto each atlas's
  own grid before comparing.
- One utility function in the original (used as a fallback when a parcellation's own
  coordinate table is not supplied) computes each parcel's representative MNI coordinate by
  taking the mean voxel location of that parcel's mask and then adding a **fixed, hardcoded
  offset** to convert voxel indices to MNI millimeter coordinates, rather than reading the
  actual affine transform out of the parcellation image's own header. That hardcoded offset
  corresponds to the corner-of-volume convention of one specific standard grid (a 1mm,
  roughly 182×218×182-voxel MNI152 template). **This is a correctness trap** — see §6.
- A patient-level TDI probability/percent map is later masked against a separate
  quality-assurance (QA) volume associated with the reference tractography atlas, and voxels
  below a small value are zeroed and the map is optionally Gaussian-smoothed — this happens
  in the same standard space as everything else, again without an explicit
  cross-checking/resampling step.

## 6. Correctness traps for a reimplementer (most important first)

1. **Hardcoded voxel→MNI offset instead of reading the affine (§5).** The coordinate-fallback
   utility adds a fixed constant to voxel indices to get MNI millimeters. This is only
   correct if every input is on the exact same grid/orientation the original toolkit assumed.
   Any reimplementation that accepts parcellations on a different grid, a different voxel
   size, or a different corner convention (e.g., a different MNI variant, or non-FSL
   orientation) will silently produce wrong coordinates with no error raised. **A
   reimplementation should always derive world coordinates from each image's own affine, never
   from a hardcoded constant.** This is the single highest-risk trap because it fails
   silently — it produces plausible-looking but wrong numbers rather than crashing. `lqtpy`'s
   own tract-atlas geometry has an analogous, independently-diagnosed trap — see
   `docs/VALIDATION.md`.
2. **Streamline-count-weighted aggregation, not a simple average (§3).** It is tempting to
   compute a parcel's or network's overall disconnection as the mean of its edges' percent
   values. That is not what the reference method does — it weights by each edge's baseline
   streamline count. Get this wrong and heavily-connected regions will be systematically
   mis-scored relative to sparsely-connected ones.
3. **The two building blocks — "percent disconnected" and "percent spared" — must both be
   computed from the same reference denominator and handled for zero/undefined edges the
   same way** (edges with zero baseline streamlines produce divide-by-zero; the source
   pipeline zeroes these rather than propagating NaN/Inf into downstream sums). An
   inconsistent NaN policy between the disconnected-percent and spared-percent matrices will
   silently corrupt the shortest-path-length step, which consumes both.
4. **The "edge considered present" threshold for shortest-path-length has two different
   default values in different parts of the source** (one place defaults to 50%, another to
   10% spared-connection-strength as the cutoff for whether an edge exists in the
   graph-distance calculation). This is very likely a bug introduced during the R→Python
   port (a default that should have been threaded through from configuration was
   re-hardcoded locally instead), not an intentional design choice. **Do not port either
   number without first checking the original paper's stated default**, and do not assume the
   two are supposed to differ.
5. **A "spared connectivity" input variable that the shortest-path-length computation depends
   on is used before it is ever defined/loaded** in the corresponding original source file —
   i.e., that part of the pipeline as it exists in the original tree today does not actually
   run end-to-end; several required imports and an intermediate load-from-disk step are
   missing. Do not treat the original SSPL code as a working reference implementation to
   test against; treat only its formulas (as described in §1 item 7 and this section) as the
   intended method, and verify the actual arithmetic against the original paper or R package
   rather than against the original tree's current (broken) state.
6. **Streamline-to-tract and streamline-to-parcel-pair assignment happens inside the
   black-box tractography step (§4), not in the described pipeline's own code.** A
   reimplementation must make an explicit, documented decision about how a streamline is
   attributed to a tract or credited to a parcel-pair edge (e.g., by endpoints landing inside
   parcel masks, vs. any voxel intersection) — the original method's exact rule here is
   unclear from the source and matters for matching its numbers.
7. **Per-tract reference totals are precomputed constants, not recomputed from the atlas at
   runtime.** If a reimplementation regenerates its own reference tract streamline counts
   (e.g., from a different tractography run or a different seeding density) rather than using
   the original toolkit's fixed per-tract totals, percentages will not be comparable to
   results produced by the original tool, even if the "is this streamline lesioned" logic is
   otherwise faithful. `lqtpy` follows this convention: reference counts come from the
   HCP-842 atlas's own `tract_info.csv`, not recomputed.
8. **Binary, not weighted, streamline-lesion intersection test (§2).** There is no
   partial-overlap weighting visible anywhere in the original source. If a reimplementer is
   tempted to add a more "sophisticated" partial-credit or overlap-length-weighted scoring
   scheme, that would be a deliberate deviation from the original method, not a faithful
   port — worth flagging explicitly to whoever signs off on the reimplementation rather than
   doing silently. `lqtpy.api.tract_disconnection` intentionally keeps the binary rule.

## Licence background

The original pipeline these notes describe is a near line-for-line Python port of the
`jdwor/LQT` R package (GPL-2), which is itself an R implementation of the algorithm in
Griffis et al. (2021). That port was never copied into this repository — see
`CONTRIBUTING.md` for the clean-room rule this implies. This document was written by
reading that port for the purpose of an independent, from-scratch reimplementation; it
describes the *method*, not the code, precisely so that `lqtpy.api.tract_disconnection`
(written from this document alone, with no reference to the original port's code, names,
structure, or comments) is not a derivative work of GPL-licensed code.

## Citation

Griffis, J. C., Metcalf, N. V., Corbetta, M., & Shulman, G. L. (2021). Lesion Quantification
Toolkit: A MATLAB software tool for estimating grey matter damage and white matter
disconnections in patients with focal brain lesions. *NeuroImage: Clinical*, 30, 102639.
https://doi.org/10.1016/j.nicl.2021.102639

R implementation referenced during this analysis: Dworkin, J. D., `LQT`
(https://github.com/jdwor/LQT), an R port of the above method.

*This document is an independent description of the method, written from reading the
original project's now-deleted `src/` tree for the purpose of a from-scratch
reimplementation. It does not reproduce source code, comments, or docstrings from that
codebase or from the R package it was ported from.*
