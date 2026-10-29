"""QC result model: graded findings, caller context, and the mask verdict.

A **finding** says what is wrong (``id``), how serious that kind of problem
is (``category``), how much of it there is (``extent``, ``affected_pct``),
how bad that makes it (``grade``: minor / moderate / major), where it is
(``locations``) and how it could be fixed (``fix``). ``severity`` keeps
the four display words used everywhere in lqtpy:

========  ==================================================
Note      informational, or below the minor band
Check     graded minor or moderate
Problem   graded major, or a space/file failure
========  ==================================================

The mask as a whole gets a **verdict** (usable / usable with caution / fix
before use / unusable) and a **score** from 0 (unusable) to 100 (no
findings), for ranking copies of the same lesion. Checks add up: several
moderate or minor findings together (:data:`FIX_POINTS`) make a mask "fix
before use", as one major finding does.

The JSON shape (``QCResult.to_dict``) is versioned by :data:`QC_VERSION`,
which changes only when QC rules or thresholds change. Key caches on it,
not on the package version. See ``docs/QC.md``.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

__all__ = ["QC_VERSION", "Finding", "DrawingContext", "QCResult", "GRADE_BANDS", "CATEGORY",
           "VERDICTS", "FIX_POINTS", "grade_of", "severity_for", "verdict_and_score"]

#: Changes whenever QC rules, thresholds or the JSON shape change.
QC_VERSION = "1.2"

INFO, WARN, ERROR = "info", "warn", "error"
GRADES = ("minor", "moderate", "major")

#: finding id -> category
CATEGORY = {
    "not_mni": "space", "misregistered": "space", "inverted": "space", "side_mismatch": "space",
    "header": "space",
    "input": "values", "values": "values",
    "sparse_slices": "drawing", "flat": "drawing", "hollow": "drawing", "holes": "drawing",
    "strokes": "drawing", "box": "drawing", "comb": "drawing", "blocky": "drawing",
    "partial_gaps": "drawing",
    "outside_brain": "extra", "edge_artifact": "extra", "specks": "extra",
    "fragmentation": "plausibility", "plausibility": "plausibility",
}

#: Categories whose findings never go above this grade (suspicion, not proof).
GRADE_CEILING = {"plausibility": "moderate"}

#: finding id -> (metric in ``extent``, (minor, moderate, major) lower bounds).
#: Below the minor bound a finding is a Note. Bounds are on the metric's own
#: unit; most are "% of the lesion affected". Tuned on the synthetic library
#: and on labelled real masks (docs/QC.md records the evidence).
GRADE_BANDS = {
    "sparse_slices": ("missing_pct", (1.0, 10.0, 25.0)),
    "flat": ("flat_pct", (5.0, 25.0, 50.0)),
    "hollow": ("missing_pct", (5.0, 15.0, 30.0)),
    "holes": ("missing_pct", (2.0, 10.0, 25.0)),
    "strokes": ("affected_pct", (0.5, 5.0, 15.0)),
    "box": ("affected_pct", (5.0, 25.0, 60.0)),
    # comb-like / blocky: a Note by default, a Check when the outline clearly changes
    "comb": ("comb_ratio", (2.5, 3.5, 1e9)),
    "blocky": ("blocky_pct", (35.0, 65.0, 1e9)),
    "partial_gaps": ("missing_pct", (0.0, 10.0, 25.0)),
    "outside_brain": ("outside_pct", (1.0, 5.0, 15.0)),
    "edge_artifact": ("affected_pct", (0.0, 5.0, 20.0)),
    "specks": ("n_specks", (4.0, 100.0, 1e9)),
    "values": ("volume_spread_pct", (0.0, 20.0, 1e9)),
    "fragmentation": ("n_parts", (149.0, 400.0, 1e9)),
}

VERDICTS = ("usable", "usable with caution", "fix before use", "unusable")
#: Checks add up: 2 points per moderate and 1 per minor finding; this many
#: points make a mask "fix before use" even without a major finding.
FIX_POINTS = 4
#: Findings that make a mask unusable outright (every number would be wrong).
FATAL = {"not_mni", "inverted"}
_PENALTY = {"major": 40, "moderate": 15, "minor": 5, None: 1}


@dataclass
class Finding:
    """One QC finding. The first five fields are the original ``Flag``
    fields (unchanged, so older callers keep working)."""

    id: str
    severity: str           # "info" | "warn" | "error"  (Note / Check / Problem)
    message: str            # one plain sentence for reports
    evidence: dict = field(default_factory=dict)
    suggestion: str = ""
    category: str = ""
    grade: "str | None" = None
    #: named measurements, units in the name (mm3, mm, pct, n_...)
    extent: dict = field(default_factory=dict)
    #: estimated share of the lesion (0-100) this finding affects
    affected_pct: "float | None" = None
    #: where to look, most important first: {"kind": "slice"|"point"|"region",
    #: "axis", "plane", "index", "ijk", "mm", "n_voxels"}
    locations: list = field(default_factory=list)
    #: machine-readable fix hint, e.g. {"call": "qc.repair_gaps", "kwargs": {...}}
    fix: "dict | None" = None

    def to_dict(self) -> dict:
        """The fields, plus ready-made wording: ``label`` ("Check, moderate")
        and ``where`` ("axial slice 107", "at -40, 10, 5 mm", or "")."""
        return {**asdict(self), "label": self.label, "where": self.where}

    @property
    def where(self) -> str:
        """The first location in words, or "" when the finding has none."""
        if not self.locations:
            return ""
        loc = self.locations[0]
        if loc.get("kind") == "slice":
            return f"{loc['plane']} slice {loc['index']}"
        return "at " + ", ".join(f"{v:.0f}" for v in loc["mm"]) + " mm"

    @property
    def label(self) -> str:
        """'Check, moderate' / 'Problem, major' / 'Note'."""
        from . import _format as fmt
        word = fmt.quality_label(self.severity)
        return f"{word}, {self.grade}" if self.grade else word


#: Backward-compatible name.
Flag = Finding


@dataclass
class DrawingContext:
    """What a drawing tool knows that a saved mask doesn't show. Every field
    is optional; anything missing is inferred from the mask.

    ``drawn_slices``: ``{axis: [slice indices]}`` on the mask's own grid
    (axis 0/1/2 = array axis) where the user actually drew. ``interpolated_slices``:
    the same for slices filled by interpolation. ``drawing_axis``: the grid
    axis the user drew on (0/1/2). ``base``: ``"mni_template"``, ``"subject"``
    or a free-text name; ``space``: ``"mni"`` or ``"native"`` when known.
    ``expected_side``: ``"left"``/``"right"``. ``source``: ``"drawn"`` or
    ``"loaded"``. ``labels_are_colours``: True when the mask's integer values
    are pen colours of one lesion (a drawing tool's palette), so several
    label values are not a finding."""

    drawn_slices: "dict | None" = None
    interpolated_slices: "dict | None" = None
    drawing_axis: "int | None" = None
    base: "str | None" = None
    space: "str | None" = None
    expected_side: "str | None" = None
    source: "str | None" = None
    labels_are_colours: "bool | None" = None

    @classmethod
    def from_any(cls, ctx) -> "DrawingContext | None":
        """Accept a DrawingContext, a dict (JSON), or None. Unknown keys are
        ignored and bad values dropped, never raised."""
        if ctx is None or isinstance(ctx, cls):
            return ctx
        if not isinstance(ctx, dict):
            return None
        out = cls()

        def axes(v):
            if not isinstance(v, dict):
                return None
            res = {}
            for k, idx in v.items():
                try:
                    a = int(k)
                    if a in (0, 1, 2):
                        res[a] = sorted({int(i) for i in idx})
                except (TypeError, ValueError):
                    continue
            return res or None

        out.drawn_slices = axes(ctx.get("drawn_slices"))
        out.interpolated_slices = axes(ctx.get("interpolated_slices"))
        try:
            ax = ctx.get("drawing_axis")
            out.drawing_axis = int(ax) if ax is not None and int(ax) in (0, 1, 2) else None
        except (TypeError, ValueError):
            out.drawing_axis = None
        for k in ("base", "space", "source"):
            v = ctx.get(k)
            setattr(out, k, str(v) if v is not None else None)
        out.labels_are_colours = True if ctx.get("labels_are_colours") is True else None
        side = str(ctx.get("expected_side") or "").strip().lower()
        out.expected_side = side if side in ("left", "right") else None
        return out

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class QCResult:
    """Everything :func:`lqtpy.qc.assess` found about one mask."""

    findings: "list[Finding]"
    verdict: str
    score: int
    worst: str
    tier: str
    qc_version: str = QC_VERSION
    context: dict = field(default_factory=dict)
    options: dict = field(default_factory=dict)
    seconds: float = 0.0
    source: "str | None" = None

    @property
    def flags(self) -> "list[Finding]":
        return self.findings

    @property
    def needs_attention(self) -> bool:
        """True when any finding is a Check or a Problem: the one question a
        tool asks before interrupting its user. Notes alone never do."""
        return any(f.severity != INFO for f in self.findings)

    def to_dict(self) -> dict:
        return {"schema": "lqtpy.qc", "qc_version": self.qc_version, "tier": self.tier,
                "verdict": self.verdict, "score": self.score, "worst": self.worst,
                "needs_attention": self.needs_attention,
                "findings": [f.to_dict() for f in self.findings], "context": self.context,
                "options": self.options, "seconds": round(self.seconds, 3), "source": self.source}

    def to_json(self, **kw) -> str:
        return json.dumps(self.to_dict(), default=_jsonable, **kw)


def _jsonable(o):
    """``json.dump`` default: numpy values to plain ones, sets to lists, the
    rest to text."""
    import numpy as np
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    return str(o)


def grade_of(fid: str, value) -> "str | None":
    """The grade for a finding's metric value, from :data:`GRADE_BANDS`."""
    band = GRADE_BANDS.get(fid)
    if band is None or value is None:
        return None
    lo_minor, lo_mod, lo_major = band[1]
    g = None
    if value >= lo_major:
        g = "major"
    elif value >= lo_mod:
        g = "moderate"
    elif value > lo_minor or (lo_minor == 0 and value > 0):
        g = "minor"
    cap = GRADE_CEILING.get(CATEGORY.get(fid, ""))
    if g and cap and GRADES.index(g) > GRADES.index(cap):
        g = cap
    return g


def severity_for(grade: "str | None", category: str = "", fatal: bool = False) -> str:
    if fatal:
        return ERROR
    if grade == "major":
        return ERROR
    if grade in ("minor", "moderate"):
        return WARN
    return INFO


def verdict_and_score(findings: "list[Finding]") -> "tuple[str, int, str]":
    """(verdict, score 0-100, worst severity) for a mask's findings."""
    worst = "ok"
    order = {"ok": 0, INFO: 1, WARN: 2, ERROR: 3}
    for f in findings:
        if order[f.severity] > order[worst]:
            worst = f.severity
    if any(f.id in FATAL for f in findings):
        return "unusable", 0, worst
    score = 100
    for f in findings:
        score -= _PENALTY.get(f.grade if f.severity != INFO else None, 1)
    score = max(1, min(100, score))
    points = sum({"moderate": 2, "minor": 1}.get(f.grade, 0) for f in findings if f.severity == WARN)
    if any(f.severity == ERROR for f in findings) or points >= FIX_POINTS:
        v = "fix before use"
    elif any(f.grade == "moderate" for f in findings) or sum(f.grade == "minor" for f in findings) >= 3:
        v = "usable with caution"
    else:
        v = "usable"
    return v, int(score), worst
