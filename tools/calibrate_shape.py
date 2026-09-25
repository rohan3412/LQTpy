"""Check the shape bands in lqtpy.measures.shape against synthetic reference
shapes at several sizes. Prints a table; exits non-zero if a reference shape
gets the wrong descriptor.

Run: python tools/calibrate_shape.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import nibabel as nib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import lqtpy  # noqa: E402
from lqtpy.measures import shape  # noqa: E402

SHAPE = (120, 120, 120)
AFF = np.diag([1.0, 1, 1, 1]); AFF[:3, 3] = -60
G = np.indices(SHAPE).astype(float) - 60  # world coords, x first


def ellipsoid(a, b, c, rot=0.0):
    x, y, z = G
    t = np.deg2rad(rot)
    xr, yr = x * np.cos(t) + y * np.sin(t), -x * np.sin(t) + y * np.cos(t)
    return (xr / a) ** 2 + (yr / b) ** 2 + (z / c) ** 2 <= 1


def tube(r, length, axis=1):
    x, y, z = G
    coords = [x, y, z]
    along = coords.pop(axis)
    return (coords[0] ** 2 + coords[1] ** 2 <= r * r) & (np.abs(along) <= length / 2)


def lobulated(r, k=5, spread=None):  # lobes overlap so it stays one component
    spread = spread or 0.75 * r
    m = np.zeros(SHAPE, bool)
    for i in range(k):
        ang = 2 * np.pi * i / k
        c = np.array([spread * np.cos(ang), spread * np.sin(ang), 0.3 * r * (-1) ** i])
        m |= (G[0] - c[0]) ** 2 + (G[1] - c[1]) ** 2 + (G[2] - c[2]) ** 2 <= (0.6 * r) ** 2
    return m


def branching(r, length):
    x, y, z = G
    stem = (x ** 2 + z ** 2 <= r * r) & (y >= -length / 2) & (y <= 0)
    b1 = ((x - y * 0.8) ** 2 + z ** 2 <= r * r) & (y > 0) & (y <= length / 2)
    b2 = ((x + y * 0.8) ** 2 + z ** 2 <= r * r) & (y > 0) & (y <= length / 2)
    return stem | b1 | b2


def c_shape(R, r):
    x, y, z = G
    ring = (np.sqrt(x ** 2 + y ** 2) - R) ** 2 + z ** 2 <= r * r
    return ring & ~((x > 0) & (np.abs(y) < R * 0.6))


CASES = []
for s in (5, 8, 12, 20):
    CASES += [(f"sphere r{s}", ellipsoid(s, s, s), "compact"),
              (f"ellipsoid 2:1:1 ({s})", ellipsoid(s, 2 * s, s), "elongated"),
              (f"ellipsoid 3:1:1 ({s})", ellipsoid(s, 3 * s, s), "elongated-marked"),
              (f"plate 4:4:1 ({s})", ellipsoid(2 * s, 2 * s, max(1.5, s / 2)), "flat"),
              (f"tube r{max(2, s // 3)} L{4 * s}", tube(max(2, s // 3), 4 * s), "elongated-marked"),
              (f"lobulated ({s})", lobulated(s), "lobulated"),
              (f"branching ({s})", branching(max(2, s // 3), 4 * s), "lobulated"),
              (f"C-shape ({s})", c_shape(1.5 * s, max(2, s // 3)), "lobulated")]
CASES.append(("oblique ellipsoid 3:1:1 (10)", ellipsoid(10, 30, 10, rot=40), "elongated-marked"))

bad = 0
print(f"{'case':28s} {'vox':>7s} {'sph':>6s} {'sol':>6s} {'elong':>6s} {'flat':>6s}  descriptors")
for name, m, expect in CASES:
    p = lqtpy.prepare(nib.Nifti1Image(m.astype(np.uint8), AFF))
    v = shape.compute(p)
    descs = shape.describe_shape(v)
    levels = [d.level for d in descs]
    ok = expect in levels or (expect == "lobulated" and "lobulated-mild" in levels
                              and v["n_voxels"] < 1000)
    bad += not ok
    print(f"{name:28s} {v['n_voxels']:7d} {v.get('sphericity', 0):6.3f} {v.get('solidity', 0):6.3f} "
          f"{v.get('elongation', 0):6.2f} {v.get('flatness', 0):6.2f}  "
          f"{'ok ' if ok else 'BAD'} {shape.sentence(descs, 'It')}")
print(f"\n{len(CASES) - bad}/{len(CASES)} reference shapes described as expected")
sys.exit(1 if bad else 0)
