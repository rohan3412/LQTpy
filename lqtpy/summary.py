"""One documented, versioned JSON summary of a lesion, for programs that
store or show lqtpy's results (a lesion registry, a drawing app).

    s = lqtpy.summary("lesion.nii.gz")          # runs the analysis (no files written)
    s = lqtpy.summary(result)                   # or reuses analyze_single(...)'s result
    json.dumps(s)                               # plain JSON types only

The shape is fixed per :data:`SUMMARY_VERSION` ("major.minor"): a minor bump
only *adds* keys, a major bump renames or removes one. Programs should check
``schema == "lqtpy.summary"`` and the major version. Every key is always
present; a part that didn't run is ``null`` (never missing). Lists are
sorted, largest first, and cut to ``top`` entries (default 10).

Keys (version 1.0)
------------------
``schema``, ``summary_version``, ``lqtpy_version``, ``qc_version``, ``created``
``input``         ``{name, sha256}`` (no path: summaries are safe to share)
``settings``      threshold, atlases, profile, extras
``volume``        ``{cc, mm3, voxels, parts, largest_part_pct}``
``location``      ``{centre_mm, side ("left"|"right"|"bilateral"|"midline"),
                  pct_left, pct_right, laterality_index}``
``quality``       ``{verdict, score, needs_attention, findings: [{id, label, grade,
                  message, where}]}``
``description``   the plain-language text, or null
``tissue``        ``{class, composition_pct: {cortex, white_matter, ...}}``
``regions``       ``{atlas key: [{name, pct_damage, pct_of_lesion}]}``
``coverage_pct``  ``{atlas key: % of the lesion inside a labelled region}``
``tracts``        ``[{name, pct_overlap}]`` (overlap proxy), or null
``disconnection`` ``[{tract, name, pct_disconnected}]``, or null
``key_structures`` ``[{name, distance_mm, reaches}]``, or null
``border_zones``  ``{pct_in_border_zones, fits}``, or null
"""
from __future__ import annotations

import datetime

__all__ = ["summary", "SUMMARY_VERSION", "SCHEMA"]

SCHEMA = "lqtpy.summary"
SUMMARY_VERSION = "1.0"


def _side(m: dict) -> str:
    left, right = m.get("pct_left", 0.0), m.get("pct_right", 0.0)
    if m.get("bilateral"):
        return "bilateral"
    if left == right:
        return "midline"
    return "left" if left > right else "right"


def _rows(df) -> list:
    if df is None:
        return []
    if hasattr(df, "to_dict"):
        return df.to_dict("records")
    return list(df)


def _num(v):
    try:
        import numpy as np
        if isinstance(v, np.generic):
            return v.item()
    except ImportError:
        pass
    return v


def _from_result(r: dict, top: int) -> dict:
    from ._version import __version__
    m = r["morphometry"]
    man = r.get("manifest") or {}
    settings = man.get("settings") or {}
    findings = r.get("qc") or []
    regions, coverage = {}, {}
    for a in r.get("atlases") or []:
        rows = sorted(_rows(a.get("df")), key=lambda x: -float(x.get("PercentOfLesion") or 0))
        regions[a["key"]] = [{"name": x["RegionName"], "pct_damage": _num(x["PercentDamage"]),
                              "pct_of_lesion": _num(x["PercentOfLesion"])}
                             for x in rows if (x.get("LesionVoxels") or 0) > 0][:top]
        coverage[a["key"]] = _num((a.get("coverage") or {}).get("PercentLabelled"))
    tr = r.get("tracts")
    tracts = None if tr is None else [
        {"name": x["Tract"], "pct_overlap": _num(x["PercentOverlap"])}
        for x in sorted(_rows(tr.get("df")), key=lambda x: -float(x["PercentOverlap"] or 0))
        if (x.get("PercentOverlap") or 0) > 0][:top]
    dc = r.get("disconnection")
    discon = None if dc is None else [
        {"tract": x["tract"], "name": x["name"], "pct_disconnected": _num(x["percent_disconnected"])}
        for x in sorted(_rows(dc), key=lambda x: -float(x["percent_disconnected"] or 0))
        if (x.get("percent_disconnected") or 0) > 0][:top]
    ks = r.get("key_structures")
    keys = None if ks is None else [
        {"name": x["Structure"], "distance_mm": _num(x["DistanceMM"]), "reaches": bool(x["Reaches"])}
        for x in sorted(_rows(ks), key=lambda x: float(x["DistanceMM"]))][:top]
    t = r.get("tissue")
    bz = r.get("border_zones")
    desc = r.get("description")
    return {
        "schema": SCHEMA,
        "summary_version": SUMMARY_VERSION,
        "lqtpy_version": __version__,
        "qc_version": r.get("qc_version"),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": {"name": r.get("name"), "sha256": (man.get("input") or {}).get("sha256")},
        "settings": {k: settings.get(k) for k in ("threshold", "atlases", "profile", "extras")},
        "volume": {"cc": m["volume_cc"], "mm3": m["volume_mm3"], "voxels": m["n_voxels"],
                   "parts": m["n_components"], "largest_part_pct": m["largest_component_pct"]},
        "location": {"centre_mm": [_num(v) for v in m["center_of_mass_mm"]], "side": _side(m),
                     "pct_left": m["pct_left"], "pct_right": m["pct_right"],
                     "laterality_index": m["laterality_index"]},
        "quality": None if r.get("qc_version") is None else {
            "verdict": r.get("qc_verdict"), "score": r.get("qc_score"),
            "needs_attention": any(f.get("severity") != "info" for f in findings),
            "findings": [{k: f.get(k) for k in ("id", "label", "grade", "message", "where")}
                         for f in findings]},
        "description": desc.get("text") if desc else None,
        "tissue": None if not t else {"class": t.get("class"),
                                      "composition_pct": dict(t.get("composition") or {})},
        "regions": regions,
        "coverage_pct": coverage,
        "tracts": tracts,
        "disconnection": discon,
        "key_structures": keys,
        "border_zones": None if not bz else {"pct_in_border_zones": bz.get("pct_in_border_zones"),
                                             "fits": bool(bz.get("fits"))},
    }


def summary(lesion_or_result, *, top: int = 10, **analysis_options) -> dict:
    """The summary of a lesion (anything :func:`lqtpy.prepare` takes; the
    analysis runs in memory, writing nothing) or of an existing
    :func:`lqtpy.analyze_single` result. ``analysis_options`` go to
    ``analyze_single`` (atlases, threshold, profile, ...)."""
    r = lesion_or_result
    if not (isinstance(r, dict) and "morphometry" in r):
        from .analysis import analyze_single
        r = analyze_single(r, write=False, plots=False, **analysis_options)
    return _from_result(r, top)
