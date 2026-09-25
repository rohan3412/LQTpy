"""Built-in measures, registered with :mod:`lqtpy.registry`.

- ``morphometry`` (core): volume, centre, extent, components, laterality --
  :func:`lqtpy.morphometry.compute_morphometry`.
- ``shape`` (extra): surface area, sphericity, principal-axis elongation and
  flatness, solidity, distances -- :mod:`lqtpy.measures.shape`.

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
    from . import core, shape
    register_measure(core.MORPHOMETRY, overwrite=True)
    register_measure(shape.SHAPE, overwrite=True)
