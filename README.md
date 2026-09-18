<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/banner-dark.svg">
  <img src="docs/assets/banner-light.svg" alt="Lcutiepie (lqtpy)" width="720">
</picture>

`lqtpy` measures what a brain lesion has damaged. Give it a lesion mask in MNI space and it
returns lesion morphometry, how much of each atlas region is destroyed, and what fraction of
each white-matter tract is disconnected — as records you can use directly, or as a
self-contained HTML report.

It is a lean reimplementation of the ideas behind the Lesion Quantification Toolkit (Griffis
et al. 2021), written from the method rather than ported from its code. Core needs only
numpy, scipy and nibabel; importing it pulls in no plotting stack.

What it does **not** do: lesion segmentation, registration to MNI (bring your own MNI-space
mask), retinotopy, or the original method's parcel-to-parcel and shortest-path measures.

## Install

```bash
pip install .            # core
pip install ".[all]"     # + pandas frames, matplotlib/nilearn figures, HTML reports
```

Atlases ship inside the package, so everything except tract disconnection works offline.

## Usage

```python
import nibabel as nib
import lqtpy

lesion = nib.load("lesion_MNI.nii.gz")

lqtpy.lesion_stats(lesion)
# {'n_voxels': 7123, 'volume_mm3': 7123.0, 'volume_cc': 7.12,
#  'n_components': 1, 'laterality_index': -1.0, ...}

lqtpy.atlas_overlap(lesion, "aal")[:1]
# [{'LabelID': 12, 'RegionName': 'Frontal_Inf_Oper_R', 'Group': 'Right',
#   'RegionVoxels': 11174, 'LesionVoxels': 1314,
#   'PercentDamage': 11.76, 'PercentOfLesion': 18.45}]

lqtpy.lobe_overlap(lesion, "aal")[:1]
# [{'Lobe': 'Frontal', 'LobeVoxels': 453225, 'LesionVoxels': 1314,
#   'PercentDamage': 0.29, 'PercentOfLesion': 18.45}]
```

A lesion can be a path, a nibabel image, or an `(array, affine)` tuple. Every function takes
`threshold=0.5`: a voxel counts as lesioned when its value is **strictly greater** than the
threshold, and the binarised mask is then resampled onto the atlas grid with
nearest-neighbour interpolation.

| Function | Returns |
|---|---|
| `lesion_stats(lesion)` | volume, centre of mass, components, laterality, bounding box |
| `atlas_overlap(lesion, atlas)` | one record per damaged region |
| `network_rollup_for(lesion, atlas)` | the same, grouped by the atlas's region groups |
| `lobe_overlap(lesion, atlas)` | the same, grouped by anatomical lobe (frontal/parietal/temporal/...) |
| `tract_proxy_overlap(lesion)` | voxel overlap with tract masks — a coarse proxy, **not** disconnection |
| `tract_disconnection(lesion)` | percent of each tract's streamlines cut by the lesion |

Bundled atlases: `aal` (116 regions), `schaefer` (100 parcels, Yeo-7 networks), `dk`
(Mindboggle DKT31 + subcortical). Register your own with
`lqtpy.register_atlas(key, name, img, labels)`.

`network_rollup_for` groups by each atlas's own region `Group` (hemisphere for `aal`/`dk`, Yeo-7
network for `schaefer`); `lobe_overlap` instead groups by anatomical lobe, independent of
hemisphere. Every `aal`/`dk` region has a lobe; `schaefer` doesn't define one, and
`lobe_overlap` raises `NoLobeMappingError` for it.

### Reports

```python
lqtpy.analyze_single("lesion_MNI.nii.gz", out_dir="out", plots=True)
```

writes a standalone HTML report. To embed sections in your own page instead, use
`lqtpy.report.fragments` — every rule is scoped to `lqt-` classes, so the fragments survive
being dropped into a page that has its own CSS:

```python
from lqtpy.report import fragments
html = fragments.stylesheet() + fragments.parcel_damage_fragment(records, "AAL")
```

`ReportStyle` controls theme colours, which sections appear and in what order, units and
precision, and a detail-level preset.

### Command line

```bash
lqtpy single lesion_MNI.nii.gz -o out
lqtpy group lesions/*.nii.gz -o out
```

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
