"""Data dictionary: what each output column means, and its unit.

Every run that writes tables also writes ``data_dictionary.csv`` next to
them (columns ``File, Column, Unit, Meaning``), covering exactly the files
it wrote.
"""
from __future__ import annotations

import csv
import os

__all__ = ["COLUMNS", "write_dictionary"]

#: column -> (unit, meaning). Shared by every table that uses the column.
COLUMNS = {
    # region tables
    "RegionName": ("", "Atlas region (atlas label name)."),
    "LabelID": ("", "Integer label of the region in the atlas volume."),
    "Group": ("", "The atlas's region group: hemisphere (AAL, DKT), Yeo-7 network (Schaefer), "
                  "major territory (arterial)."),
    "RegionVoxels": ("voxels", "Size of the region on the atlas grid."),
    "LesionVoxels": ("voxels", "Lesion voxels inside the region (atlas grid)."),
    "PercentDamage": ("%", "Percent of the region covered by the lesion."),
    "PercentOfLesion": ("%", "Percent of the lesion that lies in the region (whole lesion on "
                             "the atlas grid as denominator)."),
    "LobeVoxels": ("voxels", "Size of the lobe on the atlas grid."),
    "NetworkVoxels": ("voxels", "Size of the network or territory on the atlas grid."),
    "Lobe": ("", "Anatomical lobe; 'Subcortical' is shown as deep grey matter in reports."),
    "Network": ("", "Yeo-7 network (Schaefer) or major vascular territory (arterial)."),
    # group tables
    "NSubjects": ("subjects", "Subjects with any lesion in the row's structure."),
    "NSubjectsTotal": ("subjects", "Subjects in the group."),
    "CohortMeanPercentDamage": ("%", "Mean percent damaged over all subjects (unaffected = 0)."),
    "MeanPercentDamageAffected": ("%", "Mean percent damaged over affected subjects only."),
    "MaxPercentDamage": ("%", "Largest percent damaged in any subject."),
    "CohortMeanPercentOverlap": ("%", "Mean tract overlap over all subjects (unaffected = 0)."),
    "MeanPercentOverlapAffected": ("%", "Mean tract overlap over affected subjects only."),
    "MaxPercentOverlap": ("%", "Largest tract overlap in any subject."),
    "Subject": ("", "Subject name (file name without extension), or its ID when anonymised."),
    "QC": ("", "Worst mask-quality result: OK, Note, Check or Problem."),
    "QCFlags": ("", "Quality flags needing a look (semicolon-separated ids)."),
    "GroupFlags": ("", "Group-level flags: duplicate, outlier (semicolon-separated)."),
    # tracts
    "Tract": ("", "White-matter tract (HCP842 atlas)."),
    "Hemi": ("", "Hemisphere of the tract: L, R or blank for commissural/midline."),
    "TractVoxels": ("voxels", "Tract size on its grid."),
    "PercentOverlap": ("%", "Percent of the tract's voxels the lesion touches (a proxy, not "
                            "disconnection)."),
    "MaxCrossSectionLoad": ("%", "Largest share of one 1 mm cross-section of the tract covered "
                                 "by the lesion."),
    "BottleneckLoad": ("%", "Share of the tract's narrowest cross-section covered by the lesion."),
    "BottleneckAreaMM2": ("mm²", "Area of that narrowest cross-section."),
    "tract": ("", "Tract short code (disconnection index)."),
    "name": ("", "Tract name."),
    "n_streamlines": ("streamlines", "Reference streamlines in the tract."),
    "n_disconnected": ("streamlines", "Streamlines passing through the lesion."),
    "percent_disconnected": ("%", "Percent of the tract's streamlines passing through the lesion."),
    # key structures
    "Structure": ("", "Key structure (tract or cortical area)."),
    "Hemisphere": ("", "left or right."),
    "DistanceMM": ("mm", "Shortest distance from the lesion to the structure (0 = reaches it)."),
    "Reaches": ("", "True when the lesion reaches the structure."),
    # parts
    "Part": ("", "Separate part number (1 = largest; gap-drawn slabs merged)."),
    "Voxels": ("voxels", "Size of the part on the lesion's grid."),
    "VolumeMM3": ("mm³", "Volume of the part."),
    "Location": ("", "Location class: periventricular, juxtacortical, cortical, deep white "
                     "matter, deep grey matter, infratentorial, outside the brain."),
    "TissueClass": ("", "cortical, subcortical, cortico-subcortical or infratentorial."),
    "PctCortex": ("%", "Share of the lesion (or part) in cortex."),
    "PctWhiteMatter": ("%", "Share in cerebral white matter."),
    "PctDeepGrey": ("%", "Share in deep grey matter (basal ganglia, thalamus)."),
    "PctCerebellum": ("%", "Share in the cerebellum."),
    "PctBrainstem": ("%", "Share in the brainstem."),
    "PctInfratentorial": ("%", "Share in cerebellum + brainstem."),
    "PctVentricles": ("%", "Share in the ventricles."),
    "PctCSF": ("%", "Share in sulcal/cisternal CSF."),
    "PctOutside": ("%", "Share outside the brain template."),
    "DistVentricleMM": ("mm", "Shortest distance to the lateral ventricle (template)."),
    "DistCortexMM": ("mm", "Shortest distance to the cortex (template)."),
    "MaxAxialMM": ("mm", "Largest extent across an axial plane."),
    "NPeriventricular": ("parts", "Separate parts classed periventricular."),
    "NJuxtacortical": ("parts", "Parts classed juxtacortical."),
    "NCortical": ("parts", "Parts classed cortical."),
    "NDeepWhiteMatter": ("parts", "Parts classed deep white matter."),
    "NDeepGrey": ("parts", "Parts classed deep grey matter."),
    "NInfratentorial": ("parts", "Parts classed infratentorial."),
    "NOutsideBrain": ("parts", "Parts mostly outside the brain."),
    # morphometry
    "n_voxels": ("voxels", "Lesion voxels on its own grid."),
    "voxel_volume_mm3": ("mm³", "Volume of one voxel (from the affine)."),
    "volume_mm3": ("mm³", "Lesion volume."),
    "volume_cc": ("cm³", "Lesion volume."),
    "voxel_size_mm": ("mm", "Voxel edge lengths."),
    "shape": ("voxels", "Image grid size."),
    "center_of_mass_mm": ("mm", "Centre of mass, world (MNI) coordinates x, y, z."),
    "bbox_size_mm": ("mm", "Bounding-box size."),
    "bbox_size_vox": ("voxels", "Bounding-box size."),
    "n_components": ("", "Connected pieces (26-connectivity)."),
    "largest_component_voxels": ("voxels", "Size of the largest piece."),
    "largest_component_pct": ("%", "Largest piece as a share of the lesion."),
    "component_sizes": ("voxels", "Sizes of the 10 largest pieces."),
    "pct_left": ("%", "Share of the lesion left of the midline (x < 0)."),
    "pct_right": ("%", "Share right of the midline."),
    "pct_midline": ("%", "Share on the midline plane."),
    "laterality_index": ("", "(left - right) / (left + right): +1 fully left, -1 fully right."),
    "bilateral": ("", "True when each hemisphere holds at least 5%."),
    "header_zooms_mismatch": ("", "True when the header voxel size disagrees with the affine."),
    # comparison
    "Name": ("", "Region, lobe, territory or tract."),
    "NAffectedA": ("subjects", "Affected subjects in group A."),
    "NA": ("subjects", "Subjects in group A."),
    "PctAffectedA": ("%", "Share of group A affected."),
    "MeanA": ("%", "Mean percent damaged in group A (unaffected = 0)."),
    "NAffectedB": ("subjects", "Affected subjects in group B."),
    "NB": ("subjects", "Subjects in group B."),
    "PctAffectedB": ("%", "Share of group B affected."),
    "MeanB": ("%", "Mean percent damaged in group B."),
    "DiffMean": ("% points", "MeanA - MeanB."),
    "PFisher": ("", "Fisher's exact test p, affected vs not, A vs B."),
    "QFisher": ("", "Benjamini-Hochberg q for PFisher."),
    "PMannWhitney": ("", "Mann-Whitney U test p on percent damaged, A vs B."),
    "QMannWhitney": ("", "Benjamini-Hochberg q for PMannWhitney."),
}


def write_dictionary(out_dir: str, files: "dict[str, list[str]]",
                     extra: "dict[str, tuple[str, str]] | None" = None) -> str:
    """Write ``data_dictionary.csv`` for ``{file name: [columns]}``;
    ``extra`` describes columns not in :data:`COLUMNS` (metadata)."""
    extra = extra or {}
    path = os.path.join(out_dir, "data_dictionary.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["File", "Column", "Unit", "Meaning"])
        for f, cols in files.items():
            for c in cols:
                unit, meaning = COLUMNS.get(c) or extra.get(c) or ("", "")
                w.writerow([f, c, unit, meaning])
    return path
