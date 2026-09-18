"""LQTpy - Lesion Quantification Toolkit (lean rewrite).

Fast morphometric and atlas-overlap analysis of lesion NIfTI files, with
self-contained HTML reports for single lesions and groups. ``import lqtpy``
stays free of matplotlib/nilearn/pandas -- see ``lqtpy.api`` for the small,
stable numeric surface meant to be called in-process from a web handler.
"""
from .analysis import analyze_single, analyze_group
from .api import (atlas_overlap, lesion_stats, lobe_overlap,
                  network_rollup_for, tract_disconnection, tract_proxy_overlap)
from .atlases import (RegionMeta, atlas_from_arrays, cache_clear,
                      list_atlases, load_atlas, preload, register_atlas)
from .errors import (DataUnavailableError, GeometryError, InvalidLesionError,
                     LQTpyError, MissingDependencyError, NoLobeMappingError,
                     UnknownAtlasError)
from .io import as_lesion_image

__version__ = "0.4.0"

__all__ = [
    # Orchestrators
    "analyze_single", "analyze_group",
    # Numeric API (lqtpy.api)
    "lesion_stats", "atlas_overlap", "network_rollup_for", "lobe_overlap",
    "tract_proxy_overlap", "tract_disconnection",
    # Atlas registry + host-application atlas bridging
    "list_atlases", "load_atlas", "register_atlas", "atlas_from_arrays",
    "RegionMeta",
    # Cache lifecycle (FastAPI startup)
    "preload", "cache_clear",
    # Image coercion
    "as_lesion_image",
    # Typed exceptions
    "LQTpyError", "InvalidLesionError", "UnknownAtlasError", "GeometryError",
    "MissingDependencyError", "DataUnavailableError", "NoLobeMappingError",
]
