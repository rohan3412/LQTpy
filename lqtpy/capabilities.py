"""What this lqtpy can do, in one call, for programs that bundle it
(a drawing app, a registry) and must work with whichever version is
installed.

    caps = lqtpy.capabilities()
    if caps["features"].get("qc.fix_plan"): ...
    if caps["qc_version"] >= "1.2": ...          # QC rules (compare as version tuples)

Feature names are only ever added. A feature is listed True only when it
works right now (an optional package it needs is installed). Cheap: no
image is read and nothing heavy is imported.
"""
from __future__ import annotations

import importlib.util

__all__ = ["capabilities", "CAPABILITIES_VERSION"]

CAPABILITIES_VERSION = "1"


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def capabilities() -> dict:
    """``{"lqtpy_version", "qc_version", "summary_version",
    "capabilities_version", "features": {name: bool}, "actions": [...],
    "transform_formats": [...], "data": {...}}``."""
    from . import qc, qc_tools
    from ._version import __version__
    from .qc_advice import ADVICE
    from .qc_model import QC_VERSION
    from .summary import SUMMARY_VERSION
    from . import datasets

    features = {
        "qc.assess": hasattr(qc, "assess"),
        "qc.tiers": True,                                   # header / fast / full
        "qc.context": True,                                 # DrawingContext
        "qc.fix": hasattr(qc, "fix"),
        "qc.fix_plan": hasattr(qc_tools, "fix_plan"),
        "qc.problem_mask": hasattr(qc_tools, "problem_mask"),
        "qc.export_problems": hasattr(qc_tools, "export_problems"),
        "qc.advice": True,                                  # Finding.advice / .action
        "qc.session": _has("lqtpy.qc_session"),
        "summary": True,
        "compare": _has("lqtpy.compare"),
        "labels.per_label": True,
        "bids": True,
        "jobs.auto": True,
        "report": True,
        "report.viewer": _has("nilearn"),
        "report.flagged_voxels": _has("nilearn"),
        "report.qc_examples": True,
        "xlsx": _has("openpyxl"),
        "dicom_seg": _has("highdicom"),
        "notebook.cards": True,
        "warp.matrix": True,
        "warp.deformation_field": True,
        "warp.ants": _has("lqtpy.ants"),
    }
    from .native import TRANSFORM_FORMATS
    return {
        "capabilities_version": CAPABILITIES_VERSION,
        "lqtpy_version": __version__,
        "qc_version": QC_VERSION,
        "summary_version": SUMMARY_VERSION,
        "features": features,
        "actions": sorted({a for a, _ in ADVICE.values()}),
        "transform_formats": list(TRANSFORM_FORMATS),
        "data": {"disconnection_index": datasets.discon_index_available(),
                 "suit_atlas": datasets.suit_available()},
    }
