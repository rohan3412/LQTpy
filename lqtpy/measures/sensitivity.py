"""Outline sensitivity (extra tier): how much the results would change if the
lesion's outline were drawn one voxel differently.

Every hand-drawn outline is uncertain by about a voxel. This measure redraws
the mask three ways on its own grid and recomputes the numbers:

- ``eroded``: one voxel peeled off everywhere (6-connected erosion);
- ``dilated``: one voxel added everywhere (6-connected dilation);
- ``smoothed``: blurred (Gaussian, one voxel) and thresholded at 0.5, which
  rounds corners and fills one-voxel notches.

For each it reports the volume change and the largest change in any AAL
region's percent damage, and whether the region holding most of the lesion
stays the same. It is not a quality finding: a perfectly drawn thin lesion
along a region border is sensitive by nature. It says how much weight the
exact numbers can bear.

Words (``interpret``) are given separately for the volume and for the
regions, and ``level`` is the worse of the two:

- volume: *stable* when every redraw changes it by 20% or less, *somewhat
  sensitive* up to 40%, *sensitive* beyond (small and thin lesions are
  always sensitive here: one voxel is a large share of them);
- regions: *stable* under 5 percentage points of damage with the main
  region unchanged, *somewhat sensitive* 5-15 points, *sensitive* beyond 15
  points or when the main region changes.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..registry import Descriptor, Field, MeasureSpec

MIN_VOXELS = 30
ATLAS = "aal"
STABLE_PP, SENSITIVE_PP = 5.0, 15.0
STABLE_VOL, SENSITIVE_VOL = 20.0, 40.0
_LEVELS = ("stable", "somewhat sensitive", "sensitive")
VARIANTS = ("eroded", "dilated", "smoothed")


def _variants(mask: np.ndarray) -> "dict[str, np.ndarray]":
    pad = np.pad(mask, 2)
    out = {
        "eroded": ndimage.binary_erosion(pad),
        "dilated": ndimage.binary_dilation(pad),
        "smoothed": ndimage.gaussian_filter(pad.astype(np.float32), 1.0) > 0.5,
    }
    return {k: v[2:-2, 2:-2, 2:-2] for k, v in out.items()}


def _overlap(mask: np.ndarray, affine) -> "dict[str, tuple[float, float]]":
    import nibabel as nib
    from ..api import atlas_overlap
    if not mask.any():
        return {}
    rows = atlas_overlap(nib.Nifti1Image(mask.astype(np.uint8), affine), ATLAS)
    return {r["RegionName"]: (float(r["PercentDamage"]), float(r["PercentOfLesion"])) for r in rows}


def _main(ov: dict) -> "str | None":
    return max(ov, key=lambda k: ov[k][1]) if ov else None


def compute(prepared) -> dict:
    n = int(prepared.n_voxels)
    if n < MIN_VOXELS:
        return {"characterised": False, "n_voxels": n}
    base = _overlap(prepared.mask, prepared.affine)
    main = _main(base)
    out = {"characterised": True, "atlas": ATLAS, "main_region": main, "variants": {}}
    worst_pp, main_stable = 0.0, True
    for name, m in _variants(prepared.mask).items():
        ov = _overlap(m, prepared.affine)
        keys = set(base) | set(ov)
        pp = max((abs(ov.get(k, (0.0, 0.0))[0] - base.get(k, (0.0, 0.0))[0]) for k in keys), default=0.0)
        vmain = _main(ov)
        out["variants"][name] = {"volume_change_pct": round(100.0 * (int(m.sum()) - n) / n, 1),
                                 "max_damage_change_pp": round(pp, 1), "main_region": vmain}
        worst_pp = max(worst_pp, pp)
        main_stable &= vmain == main
    vols = [v["volume_change_pct"] for v in out["variants"].values()]
    out.update(volume_change_min_pct=min(vols), volume_change_max_pct=max(vols),
               max_damage_change_pp=round(worst_pp, 1), main_region_stable=bool(main_stable))
    vol = max(abs(v) for v in vols)
    out["volume_level"] = _LEVELS[0 if vol <= STABLE_VOL else 1 if vol <= SENSITIVE_VOL else 2]
    out["region_level"] = _LEVELS[2 if worst_pp > SENSITIVE_PP or not main_stable
                                  else 1 if worst_pp > STABLE_PP else 0]
    out["level"] = max(out["volume_level"], out["region_level"], key=_LEVELS.index)
    return out


def interpret(values: dict, context: dict) -> "list[Descriptor]":
    if not values or not values.get("characterised"):
        return [Descriptor("sensitivity", "too small to test how the outline affects the results",
                           level="too-small")]
    lo, hi = values["volume_change_min_pct"], values["volume_change_max_pct"]
    pp = values["max_damage_change_pp"]
    how = {"stable": "is stable to", "somewhat sensitive": "is somewhat sensitive to",
           "sensitive": "depends strongly on"}
    vol = (f"the volume {how[values['volume_level']]} the exact outline (one voxel smaller, larger "
           f"or smoother: {lo:+.0f}% to {hi:+.0f}%)")
    reg = (f"region damage {how[values['region_level']]} it (up to {pp:.0f} percentage points"
           + ("" if values["main_region_stable"] else ", and the main region changes") + ")")
    return [Descriptor("sensitivity", vol, level="volume-" + values["volume_level"].replace(" ", "-")),
            Descriptor("sensitivity", reg, level="regions-" + values["region_level"].replace(" ", "-"))]


SENSITIVITY = MeasureSpec(
    name="sensitivity",
    label="Outline sensitivity",
    tier="extra",
    version="1",
    compute=compute,
    interpret=interpret,
    min_voxels=MIN_VOXELS,
    description="How much volume and region damage change if the outline is one voxel smaller, "
                "larger or smoother.",
    fields=(
        Field("volume_change_min_pct", "%", "Smallest volume change across the three redraws "
              "(usually the one-voxel-smaller outline)."),
        Field("volume_change_max_pct", "%", "Largest volume change across the three redraws "
              "(usually the one-voxel-larger outline)."),
        Field("max_damage_change_pp", "percentage points", "Largest change in any AAL region's "
              "percent damage across the three redraws."),
        Field("main_region_stable", "yes/no", "Whether the region holding most of the lesion stays "
              "the same in all three redraws."),
        Field("volume_level", "", "stable / somewhat sensitive / sensitive, for the volume."),
        Field("region_level", "", "stable / somewhat sensitive / sensitive, for region damage."),
        Field("level", "", "The worse of the two."),
    ),
)
