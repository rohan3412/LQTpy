"""Plugin registry: core and extra *measures* (and extra atlases).

lqtpy is split into a fast **core** and optional **extras**. Both are
described the same way, as :class:`MeasureSpec` objects, so the rest of the
package (analysis, manifests, reports, the natural-language description)
treats a measure from a separate pip package exactly like a built-in one.

Writing a plugin
----------------
A plugin package exposes an entry point in the ``lqtpy.measures`` group
(or ``lqtpy.atlases`` for atlases) pointing at a :class:`MeasureSpec`, a
list of them, or a zero-argument callable returning either::

    # pyproject.toml of the plugin package
    [project.entry-points."lqtpy.measures"]
    fractal = "lqtpy_fractal:MEASURES"

Discovery is lazy: nothing is imported until a measure is first listed or
used, so ``import lqtpy`` stays light. A plugin that fails to import, or
whose ``requires`` modules are missing, is listed as *unavailable* with the
reason; it never breaks core.

A measure
---------
``compute(prepared) -> dict`` receives a :class:`~lqtpy.prepared.PreparedLesion`
and returns ``{field_name: value}`` for the fields it declares. It should
return ``None`` values (not raise) when the lesion is below ``min_voxels``.
``interpret(values, context) -> list[Descriptor]`` turns numbers into words
for reports and descriptions; see :mod:`lqtpy.interpret`.
"""
from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "Field", "Descriptor", "MeasureSpec", "AtlasSpec",
    "register_measure", "get_measure", "list_measures", "unavailable_measures",
    "run_measures", "plugin_summary", "discover",
]


@dataclass(frozen=True)
class Field:
    """One output value of a measure."""

    name: str
    unit: str = ""
    #: One plain-language sentence: what the number means to a reader.
    meaning: str = ""


@dataclass(frozen=True)
class Descriptor:
    """A word-level interpretation of one or more measure values.

    ``text`` is the phrase a report shows ("compact, roughly round");
    ``measure`` names the source measure; ``level`` is a short machine key
    ("compact") for tests and styling; ``detail`` explains the basis
    ("sphericity 0.82: top 20% of reference lesions").
    """

    measure: str
    text: str
    level: str = ""
    detail: str = ""


@dataclass(frozen=True)
class MeasureSpec:
    name: str
    label: str
    compute: Callable
    fields: "tuple[Field, ...]" = ()
    tier: str = "extra"
    version: str = "1"
    description: str = ""
    interpret: "Callable | None" = None
    #: Below this many lesion voxels the measure isn't meaningful; compute
    #: should return None values and interpret should say so.
    min_voxels: int = 0
    #: Importable module names this measure needs (checked before use).
    requires: "tuple[str, ...]" = ()
    #: Where it came from: "builtin" or the entry point's name.
    source: str = "builtin"

    def __post_init__(self):
        if self.tier not in ("core", "extra"):
            raise ValueError(f"MeasureSpec {self.name!r}: tier must be 'core' or 'extra'")


@dataclass(frozen=True)
class AtlasSpec:
    """An atlas contributed by a plugin. ``load()`` returns ``(img, labels)``
    in any form :func:`lqtpy.atlases.register_atlas` accepts."""

    key: str
    name: str
    load: Callable
    tier: str = "extra"
    license: str = ""
    citation: str = ""
    source: str = "builtin"


_lock = threading.RLock()
_MEASURES: "dict[str, MeasureSpec]" = {}
_UNAVAILABLE: "dict[str, dict]" = {}
_discovered = False


def register_measure(spec: MeasureSpec, *, overwrite: bool = False) -> None:
    """Register a measure. Raises ``ValueError`` on a duplicate name unless
    ``overwrite=True``."""
    with _lock:
        if spec.name in _MEASURES and not overwrite:
            raise ValueError(f"measure {spec.name!r} is already registered")
        _MEASURES[spec.name] = spec
        _UNAVAILABLE.pop(spec.name, None)


def _missing_requirements(spec: MeasureSpec) -> "list[str]":
    missing = []
    for mod in spec.requires:
        try:
            importlib.import_module(mod)
        except ImportError:
            missing.append(mod)
    return missing


def _entry_points(group: str):
    from importlib import metadata
    try:
        eps = metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=group))
        return list(eps.get(group, []))  # Python 3.9
    except Exception:
        return []


def _as_list(obj):
    if callable(obj) and not isinstance(obj, (MeasureSpec, AtlasSpec)):
        obj = obj()
    if isinstance(obj, (MeasureSpec, AtlasSpec)):
        return [obj]
    return list(obj or [])


def discover(force: bool = False) -> None:
    """Load built-in measures and every installed plugin (once)."""
    global _discovered
    with _lock:
        if _discovered and not force:
            return
        _discovered = True
        from . import measures as _builtin  # registers core + built-in extras
        _builtin.register_builtin()
        for ep in _entry_points("lqtpy.measures"):
            try:
                for spec in _as_list(ep.load()):
                    if spec.source == "builtin":
                        spec = _replace(spec, source=f"plugin:{ep.name}")
                    register_measure(spec, overwrite=True)
            except Exception as exc:
                _UNAVAILABLE[f"plugin:{ep.name}"] = {
                    "name": ep.name, "kind": "measure", "tier": "extra",
                    "reason": f"failed to load: {type(exc).__name__}: {exc}"}
        from . import atlases as _atlases
        for ep in _entry_points("lqtpy.atlases"):
            try:
                for spec in _as_list(ep.load()):
                    _atlases.register_atlas_spec(spec, source=f"plugin:{ep.name}")
            except Exception as exc:
                _UNAVAILABLE[f"atlas-plugin:{ep.name}"] = {
                    "name": ep.name, "kind": "atlas", "tier": "extra",
                    "reason": f"failed to load: {type(exc).__name__}: {exc}"}


def _replace(spec, **kw):
    import dataclasses
    return dataclasses.replace(spec, **kw)


def list_measures(tier: "str | None" = None, *, available_only: bool = True) -> "list[MeasureSpec]":
    """Registered measures, core first. ``tier`` filters to "core"/"extra".
    With ``available_only`` (default), measures whose ``requires`` are
    missing are left out (see :func:`unavailable_measures`)."""
    discover()
    out = []
    for spec in _MEASURES.values():
        if tier and spec.tier != tier:
            continue
        if available_only and _missing_requirements(spec):
            continue
        out.append(spec)
    return sorted(out, key=lambda s: (s.tier != "core", s.name))


def get_measure(name: str) -> MeasureSpec:
    discover()
    try:
        return _MEASURES[name]
    except KeyError:
        raise KeyError(f"unknown measure {name!r}; registered: {sorted(_MEASURES)}") from None


def unavailable_measures() -> "list[dict]":
    """Measures/plugins that can't run here, with the reason."""
    discover()
    out = list(_UNAVAILABLE.values())
    for spec in _MEASURES.values():
        missing = _missing_requirements(spec)
        if missing:
            out.append({"name": spec.name, "kind": "measure", "tier": spec.tier,
                        "reason": f"needs {', '.join(missing)} (not installed)"})
    return out


def run_measures(prepared, names: "list[str] | None" = None, *, tier: "str | None" = None,
                 manifest=None, interpret: bool = True) -> dict:
    """Compute measures on a prepared lesion.

    Returns ``{measure_name: {"values": {...}, "descriptors": [Descriptor...]}}``.
    Each measure's outcome is recorded in ``manifest`` (if given): ran,
    unavailable (missing requirements) or failed (raised). A failing extra
    never stops the others.
    """
    from .manifest import FAILED, RAN, UNAVAILABLE

    discover()
    specs = ([get_measure(n) for n in names] if names is not None
             else list_measures(tier, available_only=False))
    results = {}
    for spec in specs:
        missing = _missing_requirements(spec)
        if missing:
            if manifest is not None:
                manifest.add(f"measure:{spec.name}", spec.label, UNAVAILABLE, tier=spec.tier,
                             reason=f"needs {', '.join(missing)} (not installed)")
            continue
        try:
            values = spec.compute(prepared)
            descs = []
            if interpret and spec.interpret is not None:
                descs = list(spec.interpret(values, {"prepared": prepared}))
            results[spec.name] = {"values": values, "descriptors": descs}
            if manifest is not None:
                manifest.add(f"measure:{spec.name}", spec.label, RAN, tier=spec.tier,
                             version=spec.version, source=spec.source)
        except Exception as exc:  # an extra must never break the run
            if manifest is not None:
                manifest.add(f"measure:{spec.name}", spec.label, FAILED, tier=spec.tier,
                             reason=f"{type(exc).__name__}: {exc}")
    return results


def plugin_summary() -> "list[dict]":
    """Every registered measure and plugin atlas, with availability."""
    discover()
    rows = []
    for spec in list_measures(available_only=False):
        missing = _missing_requirements(spec)
        rows.append({"name": spec.name, "kind": "measure", "tier": spec.tier,
                     "version": spec.version, "source": spec.source,
                     "available": not missing,
                     "reason": f"needs {', '.join(missing)}" if missing else None})
    from . import atlases as _atlases
    for key, info in _atlases.atlas_catalog().items():
        rows.append({"name": key, "kind": "atlas", "tier": info["tier"],
                     "version": "", "source": info["source"], "available": True,
                     "reason": None})
    for u in _UNAVAILABLE.values():
        rows.append({**u, "version": "", "source": u.get("name"), "available": False})
    return rows
