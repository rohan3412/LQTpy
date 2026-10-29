"""Synthetic library of mask-quality problems ("QC zoo").

Every case is a mask with a known truth: which finding must appear (or must
not), at what extent, and where. Used by ``tests/test_qc_zoo.py`` (a fast
subset in CI) and ``tools/qc_zoo_report.py`` (the full grid: detection rate
by extent, false alarms, extent error, grade agreement, location hits).

No patient data: everything is built with numpy on the 1 mm FSL MNI152 grid
(``synthetic.MNI_*``), or resampled from it to the lab's 0.74 mm and 2 mm
MNI-aligned grids.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import nibabel as nib
from scipy import ndimage as nd

import synthetic as S

#: The lab's MNI-aligned grids besides FSL 1 mm (from the Lesion Registry).
LAB_GRIDS = {
    "0.74mm": ((207, 256, 215), 0.7374631, (-75.76, -110.76, -71.76)),
    "2mm": ((99, 117, 95), 2.0, (-98.0, -134.0, -72.0)),
}


@dataclass
class Case:
    name: str
    mask: np.ndarray
    affine: np.ndarray = field(default_factory=lambda: S.MNI_AFFINE.copy())
    #: finding ids that must be present (any one of a tuple counts)
    expect: tuple = ()
    #: finding ids that must NOT be present at Check or above
    forbid: tuple = ()
    #: the expected minimum grade of the expected finding ("minor"/"moderate"/"major")
    grade: "str | None" = None
    #: true extent, for extent-error measurements: {metric: value}
    truth: dict = field(default_factory=dict)
    #: a world point (mm) the finding's first location should be near
    near_mm: "tuple | None" = None
    #: known limitation: not expected to be detectable from the mask alone
    undetectable: bool = False
    kwargs: dict = field(default_factory=dict)
    group: str = ""
    fast: bool = True   # include in the CI subset

    def img(self):
        data = self.mask
        if data.dtype == bool:
            data = data.astype(np.uint8)
        return nib.Nifti1Image(data, self.affine)


def _w():
    return S.mni_world()


C_BASE, R_BASE = (-40, 5, 15), (14, 12, 11)
DRAWING = ("sparse_slices", "flat", "hollow", "holes", "strokes", "box", "comb", "blocky",
           "partial_gaps")
EXTRA = ("outside_brain", "edge_artifact", "specks", "strokes")


def on_lab_grid(mask: np.ndarray, grid: str) -> "tuple[np.ndarray, np.ndarray]":
    """Resample a 1 mm MNI mask onto one of the lab's MNI-aligned grids."""
    from lqtpy._resample import resample_to_img
    shape, vs, origin = LAB_GRIDS[grid]
    aff = np.diag([vs, vs, vs, 1.0])
    aff[:3, 3] = origin
    ref = nib.Nifti1Image(np.zeros(shape, np.uint8), aff)
    out = resample_to_img(nib.Nifti1Image(mask.astype(np.float32), S.MNI_AFFINE), ref,
                          interpolation="linear")
    return np.asarray(out.dataobj) > 0.5, aff


# --------------------------------------------------------------------------- #
# Clean lesions (must stay OK or Note)
# --------------------------------------------------------------------------- #
def clean_cases() -> "list[Case]":
    w = _w()
    rng = np.random.default_rng(7)
    cases = [
        Case("clean sphere r3 (lacune)", S.sphere((-24, 4, 8), 3)),
        Case("clean sphere r8", S.sphere((-40, 10, 10), 8)),
        Case("clean ellipsoid MCA 12 cm3", S.ellipsoid(C_BASE, R_BASE)),
        Case("clean large MCA", S.blobs([((-40, 10, 10), 18), ((-35, -12, 18), 14)])),
        Case("clean thin but real (8 mm)", S.ellipsoid((-30, -20, 20), (4, 18, 12))),
        Case("clean warped native 5 mm", S.native_drawing(C_BASE, R_BASE, slice_mm=5.0)),
        Case("clean warped native 3 mm", S.native_drawing(C_BASE, R_BASE, slice_mm=3.0)),
        Case("clean warped native 1 mm", S.native_drawing(C_BASE, R_BASE, slice_mm=1.0), fast=False),
    ]
    # Multi-focal small-vessel disease: 12 small lesions in white matter.
    centres = [(-24, 8, 22), (26, -6, 24), (-28, -30, 26), (18, 28, 12), (-20, -12, 30),
               (22, 14, 28), (-26, 20, 18), (28, -24, 20), (-18, -40, 24), (20, -38, 26),
               (-30, 0, 32), (30, 2, 30)]
    cases.append(Case("clean multi-focal small vessel", S.blobs([(c, 2.5 + (i % 3)) for i, c in
                                                                  enumerate(centres)])))
    # Cortical ribbon: cortex voxels in a patch (thin, curved, real).
    from lqtpy.atlases import load_atlas
    tl = load_atlas("tissue").label_array()
    patch = S.sphere((-50, -20, 30), 14)
    ribbon = patch & np.isin(tl, (1, 2)) if tl.shape == patch.shape else patch
    cases.append(Case("clean cortical ribbon", ribbon))
    # Jagged slice-by-slice drawing (outlines jump between slices): comb-like
    # since QC rules 1.2 (the user's call: report it when it is true).
    m = np.zeros(S.MNI_SHAPE, bool)
    for z in range(0, 30):
        rx, ry = rng.uniform(8, 22), rng.uniform(8, 22)
        m |= (w[..., 2] == z) & (((w[..., 0] + 40) / rx) ** 2 + ((w[..., 1] - 5) / ry) ** 2 <= 1)
    jagged = Case("jagged slice-by-slice drawing (comb-like)", m)
    cases.append(jagged)
    for grid in LAB_GRIDS:
        mm, aff = on_lab_grid(S.ellipsoid(C_BASE, R_BASE), grid)
        cases.append(Case(f"clean ellipsoid on {grid} grid", mm, aff))
    for c in cases:
        c.forbid = DRAWING + EXTRA + ("not_mni", "misregistered", "inverted")
        c.group = "clean"
    # Its slices differ wildly, so some are partly drawn too.
    jagged.expect = ("comb",)
    jagged.forbid = tuple(f for f in jagged.forbid if f not in ("comb", "partial_gaps"))
    jagged.group = "drawing"
    return cases


# --------------------------------------------------------------------------- #
# Problems at increasing extent
# --------------------------------------------------------------------------- #
def problem_cases() -> "list[Case]":
    w = _w()
    rng = np.random.default_rng(0)
    base = S.ellipsoid(C_BASE, R_BASE)
    zs = np.flatnonzero(base.any((0, 1)))
    cases: "list[Case]" = []

    def add(c: Case, group: str):
        c.group = group
        cases.append(c)

    # -- some slices only ---------------------------------------------------- #
    for k, th in ((2, 5.0), (3, 5.0), (4, 5.0), (2, 3.0), (2, 7.0)):
        m = S.native_drawing(C_BASE, R_BASE, slice_mm=th, draw_every=k)
        add(Case(f"native {th:g} mm, every {k}th slice", m, expect=("sparse_slices",),
                 forbid=("holes",), grade="moderate", truth={"missing_frac": 1 - 1 / k}, fast=k in (2, 4)), "sparse")
    for k in (2, 3, 5, 8):
        m = base.copy()
        m[:, :, [z for i, z in enumerate(zs) if i % k]] = False
        add(Case(f"MNI grid, every {k}th slice", m, expect=("sparse_slices",), forbid=("holes",),
                 grade="major",
                 truth={"missing_frac": 1 - 1 / k}, fast=k in (2, 8)), "sparse")
    for n in (1, 2, 4):
        m = base.copy()
        mid = zs[len(zs) // 2]
        gone = [mid - 2 * j for j in range(n)]
        m[:, :, gone] = False
        add(Case(f"{n} interior slice(s) left empty", m, expect=("sparse_slices",),
                 grade="minor", truth={"missing_frac": n / len(zs)},
                 near_mm=(C_BASE[0], C_BASE[1], float(S.MNI_AFFINE[2, 3] + mid))), "sparse")

    # -- single slice ---------------------------------------------------------- #
    for th in (3.0, 5.0, 7.0):
        m = S.native_drawing(C_BASE, R_BASE, slice_mm=th, only_slice=0)
        add(Case(f"native single {th:g} mm slice, warped", m, expect=("flat",), grade="major",
                 fast=th == 5.0), "flat")
    add(Case("MNI 1-voxel slice", base & (np.abs(w[..., 2] - 15) < 0.5), expect=("flat",),
             grade="major"), "flat")
    add(Case("MNI 3-voxel slab", base & (np.abs(w[..., 2] - 15) < 1.5), expect=("flat",),
             grade="major"), "flat")
    # one flat part next to a real lesion: flat share ~ small
    sheet = (np.abs(w[..., 2] - 40) < 0.5) & S.ellipsoid((-30, 30, 40), (10, 8, 30))
    add(Case("small flat part beside a real lesion", base | sheet, expect=("flat",),
             grade=None), "flat")

    # -- outline ---------------------------------------------------------------- #
    add(Case("native outline, warped", S.native_drawing(C_BASE, R_BASE, outline=True),
             expect=("hollow",), grade="major"), "hollow")
    add(Case("MNI shell outline", base & ~nd.binary_erosion(base), expect=("hollow",),
             grade="major"), "hollow")

    # -- holes --------------------------------------------------------------------- #
    for frac in (0.05, 0.2, 0.4):
        m = base & ~(rng.random(base.shape) < frac)
        add(Case(f"{int(frac * 100)}% random voxels missing", m, expect=("holes",),
                 grade={0.05: "minor", 0.2: "moderate", 0.4: "major"}[frac],
                 truth={"missing_frac": frac}, fast=frac != 0.2), "holes")
    for r in (4, 6, 9):
        m = base & ~S.sphere(C_BASE, r)
        add(Case(f"one {r} mm cavity inside", m, expect=("holes",),
                 grade="minor" if r < 9 else "moderate", near_mm=C_BASE, fast=r == 9), "holes")

    # -- specks ------------------------------------------------------------------------ #
    for n in (1, 5, 30, 200):
        m = base.copy()
        idx = np.argwhere(S.sphere((0, -20, 20), 60) & ~S.sphere(C_BASE, 20))
        for i in rng.choice(len(idx), n, replace=False):
            m[tuple(idx[i])] = True
        add(Case(f"{n} single-voxel specks", m, expect=("specks",),
                 grade=None if n < 5 else ("minor" if n < 100 else "moderate"),
                 truth={"n_specks": n}, fast=n in (5, 200)), "specks")

    # -- outside the brain ------------------------------------------------------------- #
    for mm3 in (30, 200, 1000, 5000):
        r = (3 * mm3 / (4 * np.pi)) ** (1 / 3)
        blob = S.sphere((-82, 5, 15), r + 2) & (w[..., 0] < -76)
        # (the largest blob reaches the image edge: reported once, as an edge artefact)
        add(Case(f"extra {mm3} mm3 blob outside the skull", base | blob,
                 expect=("outside_brain", "edge_artifact"),
                 grade={30: "minor", 200: "minor", 1000: "moderate", 5000: "major"}[mm3],
                 near_mm=(-82, 5, 15), fast=mm3 in (200, 5000)), "outside")
    for d in (5, 10, 20):
        m = S.ellipsoid((C_BASE[0] - 25 - d, C_BASE[1], C_BASE[2]), R_BASE)
        add(Case(f"lesion shifted {d} mm past the cortex", m,
                 expect=("outside_brain", "misregistered", "not_mni"),
                 grade="moderate" if d < 20 else "major", fast=d == 10), "outside")

    # -- image edge ------------------------------------------------------------------------- #
    for size in (1, 5, 12):
        m = base.copy()
        m[0, 100:100 + size, 80:80 + size] = True
        add(Case(f"fragment on the image edge ({size * size} mm3)", m, expect=("edge_artifact",),
                 grade="minor", near_mm=(90, -126 + 100 + size / 2, -72 + 80 + size / 2),
                 fast=size == 5), "edge")

    # -- strokes and boxes -------------------------------------------------------------------- #
    sheet = (np.abs(w[..., 0] + 40) < 0.5) & (np.abs(w[..., 1] - 5) < 25) & (np.abs(w[..., 2] - 15) < 25)
    add(Case("sagittal stroke through the lesion", base | sheet, expect=("strokes",), grade="minor"),
        "strokes")
    disc = (np.abs(w[..., 1] + 51) <= 3.5) & (((w[..., 0] + 12) / 26) ** 2 + ((w[..., 2] - 5) / 30) ** 2 <= 1)
    sag = (np.abs(w[..., 0] + 10) <= 0.5) & (w[..., 1] >= -75) & (w[..., 1] <= -30) & (w[..., 2] >= -30) & (w[..., 2] <= 35)
    ax = (np.abs(w[..., 2] - 2) <= 0.5) & (w[..., 1] >= -60) & (w[..., 1] <= -40) & (w[..., 0] >= -35) & (w[..., 0] <= 15)
    add(Case("crosshair drawing (coronal disc + two stroke planes)", disc | sag | ax,
             expect=("strokes",), grade="minor"), "strokes")
    box = (np.abs(w[..., 0] + 40) <= 12) & (np.abs(w[..., 1] - 5) <= 15) & (np.abs(w[..., 2] - 15) <= 10)
    add(Case("rectangle-tool box", box, expect=("box",), grade="moderate"), "box")

    # -- values ---------------------------------------------------------------------------------- #
    add(Case("probability map (0-1)", nd.gaussian_filter(base.astype(np.float32), 3),
             expect=("values",)), "values")
    lab = base.astype(np.uint8)
    lab[S.sphere((-40, 5, 20), 5)] = 2
    add(Case("two label values (1, 2)", lab, expect=("values",), grade="minor"), "values")
    nanm = base.astype(np.float32)
    nanm[0, 0, 0] = np.nan
    add(Case("NaN voxels", nanm, expect=("input",)), "values")

    # -- space ------------------------------------------------------------------------------------- #
    add(Case("inverted mask (background = 1)", ~base, expect=("inverted",), grade="major"), "space")
    add(Case("native grid, identity affine", base, np.eye(4), expect=("not_mni",), grade="major"),
        "space")
    bad = S.MNI_AFFINE.copy()
    bad[:3, 3] += [0, 60, 0]
    add(Case("affine shifted 60 mm (bad origin)", base, bad, expect=("not_mni", "misregistered"),
             grade="major"), "space")
    add(Case("left-right flipped, expected left", base[::-1], expect=("side_mismatch",),
             grade="major", kwargs={"expected_side": "left"}), "space")
    add(Case("front-back flipped (no expected position)", base[:, ::-1], undetectable=True),
        "space")

    # -- plausibility ------------------------------------------------------------------------------ #
    add(Case("tiny lesion 30 mm3", S.sphere(C_BASE, 1.9), expect=("plausibility",)), "plausibility")
    add(Case("whole hemisphere", (w[..., 0] < -2) & S.ellipsoid((-35, -15, 10), (35, 70, 55)),
             expect=("plausibility",)), "plausibility")
    add(Case("only in the lateral ventricle", _in_ventricle(), expect=("plausibility",)),
        "plausibility")
    add(Case("mirror copy on both sides", base | base[::-1], expect=("plausibility",)),
        "plausibility")

    # -- on the lab grids ------------------------------------------------------------------------- #
    for grid in LAB_GRIDS:
        mm, aff = on_lab_grid(S.native_drawing(C_BASE, R_BASE, draw_every=2), grid)
        add(Case(f"every 2nd slice on the {grid} grid", mm, aff, expect=("sparse_slices",),
                 grade="moderate", fast=grid == "0.74mm"), "grids")
    return cases


def _in_ventricle() -> np.ndarray:
    """A small blob inside the left lateral ventricle of the template."""
    from lqtpy.atlases import load_atlas
    tl = load_atlas("tissue").label_array()
    vent = tl == 7
    c = np.argwhere(vent).mean(0)
    w = S.mni_world()
    centre = S.MNI_AFFINE[:3, :3] @ c + S.MNI_AFFINE[:3, 3]
    # the ventricle voxel closest to its centroid, then a sphere there
    idx = np.argwhere(vent)
    k = idx[np.argmin(((idx - c) ** 2).sum(1))]
    centre = S.MNI_AFFINE[:3, :3] @ k + S.MNI_AFFINE[:3, 3]
    return S.sphere(tuple(centre), 4) & vent if vent.shape == S.MNI_SHAPE else S.sphere(tuple(centre), 3)


def all_cases() -> "list[Case]":
    return clean_cases() + problem_cases()
