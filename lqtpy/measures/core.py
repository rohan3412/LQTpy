"""Core morphometry as a registered measure."""
from __future__ import annotations

from ..registry import Field, MeasureSpec


def _compute(prepared) -> dict:
    from ..morphometry import compute_morphometry
    return compute_morphometry(prepared)


MORPHOMETRY = MeasureSpec(
    name="morphometry",
    label="Core morphometry",
    tier="core",
    version="2",
    compute=_compute,
    description="Volume, centre of mass, extent, connected components and laterality.",
    fields=(
        Field("volume_mm3", "mm³", "Total lesion volume."),
        Field("volume_cc", "cm³", "Total lesion volume in cubic centimetres (= mL)."),
        Field("n_components", "", "Number of separate (26-connected) lesion parts."),
        Field("laterality_index", "", "+1 entirely left, -1 entirely right, near 0 bilateral."),
        Field("center_of_mass_mm", "mm", "Centre of mass in world (MNI) coordinates."),
        Field("bbox_size_mm", "mm", "Size of the box enclosing the lesion, per axis."),
        Field("pct_left", "%", "Share of lesion voxels in the left hemisphere."),
        Field("pct_right", "%", "Share of lesion voxels in the right hemisphere."),
        Field("pct_midline", "%", "Share of lesion voxels exactly on the midline (x = 0)."),
    ),
)
