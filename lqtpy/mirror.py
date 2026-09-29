"""The lesion's mirror image in the other hemisphere.

Why: the same place on the other side is the lesion's **homologous
region**. It is used

- as an intact **control region** for the damaged one (perfusion,
  cortical thickness, fMRI or DTI measured in the mirror region of the
  healthy hemisphere), and
- in **recovery research**, where the undamaged homologue of a damaged area
  can take over part of its work (for example the right-hemisphere
  homologue of Broca's area in aphasia recovery).

:func:`mirror` flips a lesion across the midline (world x -> -x, which is
exact in MNI space). :func:`homologue` names the regions the mirror image
covers and says how symmetric the lesion itself is (how much of it overlaps
its own mirror image: near 0 for a one-sided lesion, high for a midline or
bilateral one).
"""
from __future__ import annotations

import numpy as np
import nibabel as nib

__all__ = ["mirror", "homologue"]


def mirror(lesion, *, threshold: float = 0.5) -> nib.Nifti1Image:
    """The lesion reflected across the midline plane x = 0 (world mm)."""
    from .prepared import prepare
    p = prepare(lesion, threshold)
    flip = np.diag([-1.0, 1.0, 1.0, 1.0])
    return nib.Nifti1Image(p.mask.astype(np.uint8), flip @ p.affine)


def homologue(lesion, *, threshold: float = 0.5, atlas: str = "dk", top: int = 6) -> dict:
    """Regions covered by the lesion's mirror image, and its symmetry.

    Returns ``{"regions": [{"name", "hemisphere", "pct_of_mirror",
    "function"}], "self_overlap_pct", "side"}``. ``side`` is the hemisphere
    the mirror image lies in ("right" for a left lesion)."""
    from .atlases import load_atlas
    from .overlap import region_overlap_from_mask
    from .prepared import prepare
    p = prepare(lesion, threshold)

    def build():
        at = load_atlas(atlas)
        m = prepare(mirror(p), threshold=0.0)
        gm = m.on_grid(at.img)
        recs = region_overlap_from_mask(gm.mask, at)
        total = gm.n_voxels or 1
        regions = []
        for r in recs[:top * 3]:
            meta = at.labels[r["LabelID"]]
            regions.append({"name": meta.display(), "hemisphere": meta.Hemisphere,
                            "function": meta.Function, "region": meta.RegionName,
                            "pct_of_mirror": round(100.0 * r["LesionVoxels"] / total, 1)})
        regions.sort(key=lambda d: -d["pct_of_mirror"])
        own = p.on_grid(at.img).mask
        inter = int(np.count_nonzero(own & gm.mask))
        from .morphometry import compute_morphometry
        mo = compute_morphometry(p)
        side = "right" if mo["pct_left"] > mo["pct_right"] else "left"
        return {"regions": [r for r in regions if r["pct_of_mirror"] >= 1][:top],
                "self_overlap_pct": round(100.0 * inter / max(int(own.sum()), 1), 1),
                "side": side}
    return dict(p.cached(("homologue", atlas, top), build))
