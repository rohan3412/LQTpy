"""Run manifests: a record of what an analysis did and didn't do.

Every ``analyze_single``/``analyze_group`` result carries a ``manifest``.
Reports render their scope block straight from it, so a reader can always
tell a core-only report from one with extras, and why a section is missing.

A module entry is::

    {"id": "disconnection", "label": "Streamline disconnection",
     "tier": "core" | "extra", "status": "ran" | "skipped" | "unavailable" | "failed",
     "reason": "no disconnection index installed" | None}

``skipped`` means the caller turned it off; ``unavailable`` means it
couldn't run in this environment (missing data or plugin); ``failed``
means it raised, and ``reason`` carries the error.
"""
from __future__ import annotations

import datetime

from ._version import __version__

__all__ = ["Manifest", "RAN", "SKIPPED", "UNAVAILABLE", "FAILED"]

RAN, SKIPPED, UNAVAILABLE, FAILED = "ran", "skipped", "unavailable", "failed"
_STATUSES = (RAN, SKIPPED, UNAVAILABLE, FAILED)


class Manifest:
    """Builder for a manifest dict. ``to_dict()`` is what results carry."""

    def __init__(self, kind: str, *, settings: "dict | None" = None):
        if kind not in ("single", "group", "group-member"):
            raise ValueError(f"unknown report kind {kind!r}")
        self.kind = kind
        self.settings = dict(settings or {})
        self.modules: "list[dict]" = []
        self.created = datetime.datetime.now().isoformat(timespec="seconds")

    def add(self, id: str, label: str, status: str, *, tier: str = "core",
            reason: "str | None" = None, **extra) -> dict:
        if status not in _STATUSES:
            raise ValueError(f"unknown module status {status!r}")
        if tier not in ("core", "extra"):
            raise ValueError(f"unknown tier {tier!r}")
        entry = {"id": id, "label": label, "tier": tier, "status": status, "reason": reason}
        entry.update(extra)
        self.modules.append(entry)
        return entry

    def status(self, id: str) -> "str | None":
        for m in self.modules:
            if m["id"] == id:
                return m["status"]
        return None

    def to_dict(self) -> dict:
        from . import registry
        ran_extra = any(m["tier"] == "extra" and m["status"] == RAN for m in self.modules)
        return {
            "kind": self.kind,
            "scope": "core + extras" if ran_extra else "core",
            "lqtpy_version": __version__,
            "created": self.created,
            "settings": self.settings,
            "modules": list(self.modules),
            "plugins": registry.plugin_summary(),
        }
