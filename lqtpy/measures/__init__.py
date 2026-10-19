"""Built-in measures, registered with :mod:`lqtpy.registry`.

- ``morphometry`` (core): volume, centre, extent, components, laterality --
  :func:`lqtpy.morphometry.compute_morphometry`.
- ``shape`` (extra): surface area, sphericity, principal-axis elongation and
  flatness, solidity, distances -- :mod:`lqtpy.measures.shape`.
- ``sensitivity`` (extra): how much volume and region damage change if the
  outline is one voxel smaller, larger or smoother --
  :mod:`lqtpy.measures.sensitivity`.

Third-party measures (e.g. fractal dimension) arrive through the
``lqtpy.measures`` entry point group instead; see :mod:`lqtpy.registry`.
"""
from __future__ import annotations

_registered = False


def register_builtin() -> None:
    global _registered
    if _registered:
        return
    _registered = True
    from ..registry import register_measure
    from . import core, sensitivity, shape
    register_measure(core.MORPHOMETRY, overwrite=True)
    register_measure(shape.SHAPE, overwrite=True)
    register_measure(sensitivity.SENSITIVITY, overwrite=True)
