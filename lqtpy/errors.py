"""Typed exception hierarchy for lqtpy.

lqtpy is meant to be called **in-process** from a web server (e.g. a FastAPI
app serving concurrent requests on a threadpool). A web handler needs to map
a failure to an HTTP status code without parsing message strings, so every
lqtpy-specific error is one of two kinds:

- a **caller mistake** -- bad input the caller can fix and retry. Maps to a
  4xx response.
- an **internal/environment failure** -- something wrong with the server
  itself (missing data file, missing optional dependency). Maps to a 5xx
  response.

Each exception below documents which HTTP class it maps to. Several also
subclass a stdlib exception type for backward compatibility, so code written
against the previous ad-hoc ``raise ValueError(...)``/``raise KeyError(...)``
still catches these.
"""
from __future__ import annotations


class LQTpyError(Exception):
    """Base class for every lqtpy-specific exception."""


class InvalidLesionError(LQTpyError, ValueError):
    """Caller mistake -- maps to HTTP 4xx.

    Raised for an unreadable or wrong-dimensional lesion image, or for an
    empty lesion mask where a non-empty one is required. Also a
    ``ValueError``, so existing ``except ValueError`` callers keep working
    unchanged.
    """


class UnknownAtlasError(LQTpyError, KeyError):
    """Caller mistake -- maps to HTTP 4xx.

    Raised for an atlas key that isn't registered (built in or via
    :func:`lqtpy.atlases.register_atlas`). Also a ``KeyError``, so existing
    ``except KeyError`` callers keep working unchanged.
    """


class GeometryError(LQTpyError):
    """Caller mistake -- maps to HTTP 4xx.

    Raised when an image's affine is degenerate (non-finite or singular) or
    otherwise geometrically incompatible with the operation being performed.
    """


class NoLobeMappingError(LQTpyError, ValueError):
    """Caller mistake -- maps to HTTP 4xx.

    Raised by :func:`lqtpy.api.lobe_overlap`/:func:`lqtpy.overlap.lobe_rollup`
    when the requested atlas doesn't define a lobe for any of its regions
    (``RegionMeta.Lobe`` is ``None`` throughout) -- currently ``schaefer``,
    which only has Yeo-7 functional networks. Also a ``ValueError``, so
    existing ``except ValueError`` callers keep working unchanged.
    """


class MissingDependencyError(LQTpyError, ImportError):
    """Internal/environment failure -- maps to HTTP 5xx.

    Raised when an optional extra is required for the code path being
    called but isn't installed in this environment. Also an
    ``ImportError``, so existing ``except ImportError`` callers keep working
    unchanged.
    """

    def __init__(self, extra: str, message: "str | None" = None):
        #: the optional-dependency extra (from ``pyproject.toml``) that
        #: needs installing, e.g. ``"tables"`` or ``"plots"``.
        self.extra = extra
        if message is None:
            message = (
                f"This requires the optional '{extra}' extra: "
                f"pip install 'lqtpy[{extra}]'"
            )
        super().__init__(message)


class DataUnavailableError(LQTpyError, FileNotFoundError):
    """Internal/environment failure -- maps to HTTP 5xx.

    Raised when a required data file is missing on disk -- a bundled atlas
    volume that failed to install correctly, or fetch-on-demand data that
    hasn't been downloaded yet. Also a ``FileNotFoundError``, so existing
    ``except FileNotFoundError``/``except OSError`` callers keep working
    unchanged.
    """
