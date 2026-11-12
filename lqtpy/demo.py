"""``lqtpy demo``: a report for a made-up lesion, to see what lqtpy produces.

No patient data: the lesion is two overlapping ellipsoids in the left
hemisphere on the 1 mm MNI152 grid.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

__all__ = ["demo_lesion", "make_demo"]

_SHAPE = (182, 218, 182)
_AFFINE = np.array([[-1.0, 0, 0, 90], [0, 1, 0, -126], [0, 0, 1, -72], [0, 0, 0, 1]])


def demo_lesion():
    """A synthetic stroke-like lesion (left insula and the white matter around
    it, about 23 cc) as a NIfTI image in MNI space."""
    import nibabel as nib
    ijk = np.indices(_SHAPE).reshape(3, -1).T.astype(float)
    world = ijk @ _AFFINE[:3, :3].T + _AFFINE[:3, 3]
    m = np.zeros(len(world), bool)
    for centre, radii in (((-36, 2, 4), (11, 24, 17)), ((-30, -24, 20), (9, 12, 10))):
        d = (world - np.asarray(centre, float)) / np.asarray(radii, float)
        m |= (d ** 2).sum(1) <= 1.0
    return nib.Nifti1Image(m.reshape(_SHAPE).astype(np.uint8), _AFFINE)


def make_demo(out_dir: "str | None" = None, *, open_browser: bool = True, verbose: bool = True) -> str:
    """Write the demo lesion and its report; returns the report path."""
    import nibabel as nib
    from .analysis import analyze_single
    out_dir = out_dir or tempfile.mkdtemp(prefix="lqtpy_demo_")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "demo_lesion.nii.gz")
    nib.save(demo_lesion(), path)
    res = analyze_single(path, out_dir=out_dir, plots=True, verbose=verbose)
    report = res["report_path"]
    if open_browser:
        import webbrowser
        webbrowser.open(report if "://" in report else "file:///" + os.path.abspath(report).replace("\\", "/"))
    return report
