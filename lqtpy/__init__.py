"""LQTpy - Lesion Quantification Toolkit (lean rewrite).

Fast morphometry, atlas overlap, mask quality checks and plain-language
descriptions of lesion NIfTI files, with self-contained HTML reports for
single lesions and groups. ``import lqtpy`` stays free of
matplotlib/nilearn/pandas -- see ``lqtpy.api`` for the small, stable numeric
surface meant to be called in-process from a web handler, and
``lqtpy.prepare`` to load a lesion once and reuse it across calls.
"""
from .analysis import analyze_single, analyze_group
from .api import (atlas_coverage, atlas_overlap, lesion_stats, lobe_overlap,
                  network_rollup_for, tract_disconnection, tract_proxy_overlap)
from .prepared import PreparedLesion, prepare
from . import (cohort, formats, metadata, mirror, native, qc, registry, repro, structures,
               tissue, watershed)
from .describe import Description, describe
from .atlases import (RegionMeta, atlas_from_arrays, cache_clear,
                      list_atlases, load_atlas, preload, register_atlas)
from .errors import (DataUnavailableError, GeometryError, IndexFormatError,
                     InvalidAtlasError, InvalidLesionError, LQTpyError,
                     MissingDependencyError, NoLobeMappingError,
                     UnknownAtlasError)
from .io import as_lesion_image

from ._version import __version__  # noqa: E402

__all__ = [
    # Orchestrators
    "analyze_single", "analyze_group",
    # Numeric API (lqtpy.api)
    "prepare", "PreparedLesion",
    # Quality control, descriptions, plugin registry
    "qc", "tissue", "describe", "Description", "registry",
    # Locations, cohorts, inputs, reproducibility
    "watershed", "structures", "mirror", "cohort", "metadata", "formats", "native", "repro",
    "lesion_stats", "atlas_overlap", "atlas_coverage", "network_rollup_for",
    "lobe_overlap", "tract_proxy_overlap", "tract_disconnection",
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
    "IndexFormatError", "InvalidAtlasError",
]
