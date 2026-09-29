"""Plugin starter template (``lqtpy new-plugin NAME``).

Writes a ready-to-install package with one extra measure wired to the
``lqtpy.measures`` entry point, an interpretation that turns the number into
words, and a test. Fill in ``compute()`` (for example with the
fractal-dimension code from your own repository), then::

    pip install -e ./lqtpy-NAME
    lqtpy single lesion.nii.gz --extras NAME

and the measure appears in the report's Extras section, in the run
manifest, and (through ``interpret``) in plain words.
"""
from __future__ import annotations

import os
import re

__all__ = ["new_plugin"]

_PYPROJECT = '''[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "lqtpy-{dash}"
version = "0.1.0"
description = "{title} measure for LQTpy"
requires-python = ">=3.9"
dependencies = ["lqtpy>=0.5", "numpy"]

[project.entry-points."lqtpy.measures"]
{name} = "lqtpy_{name}:MEASURES"

[tool.hatch.build.targets.wheel]
packages = ["lqtpy_{name}"]
'''

_INIT = '''"""{title}: an extra measure for LQTpy.

Registered through the ``lqtpy.measures`` entry point (see pyproject.toml),
so ``lqtpy.analyze_single(..., extras=["{name}"])`` and
``lqtpy single lesion.nii.gz --extras {name}`` run it.
"""
from __future__ import annotations

import numpy as np

from lqtpy.registry import Descriptor, Field, MeasureSpec

__all__ = ["MEASURES", "compute", "interpret"]

#: Below this many voxels the measure isn't meaningful.
MIN_VOXELS = 50


def compute(prepared) -> dict:
    """``prepared`` is an lqtpy.PreparedLesion: ``.mask`` (bool array),
    ``.affine``, ``.n_voxels``, ``.cropped(pad)`` -> (sub_mask, offset),
    ``.world_coords()``. Return one value per declared Field; ``None``
    when the lesion is too small."""
    if prepared.n_voxels < MIN_VOXELS:
        return {{"{name}": None}}
    sub, _ = prepared.cropped(pad=1)
    # TODO: replace this placeholder with the real measure.
    # (It returns the fraction of the bounding box the lesion fills.)
    value = float(sub.sum()) / float(sub.size)
    return {{"{name}": round(value, 4)}}


def interpret(values: dict, context: dict) -> "list[Descriptor]":
    """Numbers into words for reports (keep the full number in the CSV).
    ``context`` holds e.g. ``volume_mm3`` for size-aware wording."""
    v = values.get("{name}")
    if v is None:
        return [Descriptor("{name}", "too small to measure", level="na")]
    # TODO: calibrate these cut-offs on reference lesions.
    if v < 0.3:
        return [Descriptor("{name}", "low {title_lower}", level="low", detail=f"{{v:.3f}}")]
    if v > 0.6:
        return [Descriptor("{name}", "high {title_lower}", level="high", detail=f"{{v:.3f}}")]
    return [Descriptor("{name}", "typical {title_lower}", level="typical", detail=f"{{v:.3f}}")]


MEASURES = [
    MeasureSpec(
        name="{name}",
        label="{title}",
        compute=compute,
        interpret=interpret,
        fields=(Field("{name}", "", "{title} of the lesion (describe what the number means)."),),
        tier="extra",
        version="1",
        min_voxels=MIN_VOXELS,
        description="{title} (plugin).",
    )
]
'''

_TEST = '''import numpy as np
import nibabel as nib

import lqtpy
from lqtpy_{name} import MEASURES, compute


def _sphere(r=8):
    z, y, x = np.mgrid[-20:21, -20:21, -20:21]
    m = (x ** 2 + y ** 2 + z ** 2) <= r ** 2
    return nib.Nifti1Image(m.astype(np.uint8), np.eye(4))


def test_compute_returns_declared_fields():
    out = compute(lqtpy.prepare(_sphere()))
    assert set(out) == {{f.name for f in MEASURES[0].fields}}
    assert out["{name}"] is not None


def test_too_small_gives_none():
    m = np.zeros((10, 10, 10), np.uint8)
    m[5, 5, 5] = 1
    assert compute(lqtpy.prepare(nib.Nifti1Image(m, np.eye(4))))["{name}"] is None


def test_registered_through_entry_point():
    # Needs `pip install -e .` so the entry point exists.
    names = [m.name for m in lqtpy.registry.list_measures("extra", available_only=False)]
    assert "{name}" in names
'''

_README = '''# lqtpy-{dash}

{title}, an extra measure for [LQTpy](https://github.com/rohan3412/LQTpy).

```bash
pip install -e .
lqtpy single lesion.nii.gz --extras {name}
```

- `lqtpy_{name}/__init__.py`: `compute()` (the number) and `interpret()` (the words).
- `tests/`: `pytest` checks the fields, the small-lesion case and registration.
- The entry point in `pyproject.toml` is what makes LQTpy find it.

Keep `compute()` fast and deterministic; return `None` values (don't raise)
when the lesion is too small. Record the method's source and licence here.
'''


def new_plugin(name: str, dest: str = ".") -> str:
    """Write the template for a measure called ``name`` into
    ``dest/lqtpy-<name>/`` and return that path. Refuses to overwrite."""
    if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
        raise ValueError("plugin name: lowercase letters, digits and underscores, starting "
                         "with a letter (e.g. 'fractal')")
    dash = name.replace("_", "-")
    title = name.replace("_", " ").capitalize()
    root = os.path.join(dest, f"lqtpy-{dash}")
    if os.path.exists(root):
        raise FileExistsError(f"{root} already exists")
    fmt = dict(name=name, dash=dash, title=title, title_lower=title.lower())
    files = {
        "pyproject.toml": _PYPROJECT.format(**fmt),
        os.path.join(f"lqtpy_{name}", "__init__.py"): _INIT.format(**fmt),
        os.path.join("tests", f"test_{name}.py"): _TEST.format(**fmt),
        "README.md": _README.format(**fmt),
    }
    for rel, text in files.items():
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
    return root
