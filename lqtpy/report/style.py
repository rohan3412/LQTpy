"""Report style knobs (``ReportStyle``): the small set of decisions a host
application or the standalone report can override -- colours, which
sections render and in what order, units/precision, and a detail-level
preset. Pure data + validation; no HTML/CSS-string building here (that's
``fragments.stylesheet``), and no plotting imports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["SeverityScale", "ReportStyle", "DEFAULT_STYLE", "DARK_STYLE", "VALID_SECTIONS"]

#: Section ids ``ReportStyle.sections``/``fragments.render_sections`` know
#: about, in the order they'd typically appear.
VALID_SECTIONS = (
    "morphometry", "parcel_damage", "network_rollup",
    "disconnection", "tract_proxy",
)

_VALID_UNITS = ("mm3", "cm3", "mL")
_VALID_DETAIL = ("summary", "standard", "full")

# Accepts #rgb/#rgba/#rrggbb/#rrggbbaa, func(...) notations (rgb/rgba/hsl/
# hsla/oklch/...), and bare CSS colour keywords (e.g. "red", "transparent",
# "currentColor"). Not a full CSS-colour grammar -- just enough to catch
# obviously-wrong values (empty strings, stray HTML, etc.) early.
_COLOR_RE = re.compile(r"^(#[0-9a-fA-F]{3,8}|[a-zA-Z][a-zA-Z-]*\([^)]*\)|[a-zA-Z][a-zA-Z-]*)$")


def _check_color(label: str, value) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty CSS colour string, got {value!r}")
    if not _COLOR_RE.match(value.strip()):
        raise ValueError(f"{label}={value!r} doesn't look like a plausible CSS colour")


def _check_nonneg_int(label: str, value) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative int, got {value!r}")


@dataclass(frozen=True)
class SeverityScale:
    """Colour scale for percentage-damage values, bucketed into three tiers.

    Thresholds are inclusive upper bounds: a value ``<= low_max`` is "low",
    ``<= medium_max`` is "medium", anything higher is "high". Colour is
    always paired with the printed percentage -- never the only signal.
    """

    low: str = "#2f9e44"
    medium: str = "#e8a800"
    high: str = "#c1443c"
    low_max: float = 20.0
    medium_max: float = 60.0

    def __post_init__(self) -> None:
        for name in ("low", "medium", "high"):
            _check_color(f"SeverityScale.{name}", getattr(self, name))
        if not (0 <= self.low_max <= self.medium_max <= 100):
            raise ValueError(
                "SeverityScale thresholds must satisfy "
                f"0 <= low_max <= medium_max <= 100, got low_max={self.low_max!r}, "
                f"medium_max={self.medium_max!r}"
            )

    def bucket(self, percent: float) -> str:
        """CSS colour for a percentage-damage value."""
        if percent <= self.low_max:
            return self.low
        if percent <= self.medium_max:
            return self.medium
        return self.high


@dataclass(frozen=True)
class ReportStyle:
    """Style knobs for LQTpy's HTML fragments and standalone report.

    Covers exactly four families: theme/colours, section selection+order,
    units/precision, and a detail-level preset. Everything else (layout,
    fonts) is fixed in ``fragments.stylesheet`` / the standalone report's
    own page chrome.
    """

    # 1. Theme/colours. A host overrides individual colours by constructing
    #    a new ReportStyle (frozen dataclass -- use dataclasses.replace).
    accent: str = "#c1443c"
    text: str = "#1b1f24"
    muted_text: str = "#5b6673"
    border: str = "#e3e7ec"
    surface: str = "#f7f9fb"
    row_stripe: str = "#f2f4f7"
    severity: SeverityScale = field(default_factory=SeverityScale)

    # 2. Section selection and order.
    sections: "tuple[str, ...]" = VALID_SECTIONS

    # 3. Units and precision.
    volume_unit: str = "cm3"        # "mm3" | "cm3" | "mL"
    percent_decimals: int = 1
    volume_decimals: int = 2

    # 4. Detail-level preset.
    detail: str = "standard"        # "summary" | "standard" | "full"

    def __post_init__(self) -> None:
        for name in ("accent", "text", "muted_text", "border", "surface", "row_stripe"):
            _check_color(f"ReportStyle.{name}", getattr(self, name))
        if not isinstance(self.severity, SeverityScale):
            raise ValueError(
                f"ReportStyle.severity must be a SeverityScale, got {type(self.severity)!r}")

        unknown = [s for s in self.sections if s not in VALID_SECTIONS]
        if unknown:
            raise ValueError(
                f"Unknown section id(s) {unknown!r}; valid ids are {VALID_SECTIONS}")

        if self.volume_unit not in _VALID_UNITS:
            raise ValueError(
                f"volume_unit must be one of {_VALID_UNITS}, got {self.volume_unit!r}")
        if self.detail not in _VALID_DETAIL:
            raise ValueError(f"detail must be one of {_VALID_DETAIL}, got {self.detail!r}")

        _check_nonneg_int("percent_decimals", self.percent_decimals)
        _check_nonneg_int("volume_decimals", self.volume_decimals)

    # ------------------------------------------------------------------ #
    # Derived helpers used by fragments.py.
    # ------------------------------------------------------------------ #
    def volume_from_mm3(self, mm3: float) -> float:
        """Convert an ``mm3`` volume to the style's chosen unit, rounded to
        ``volume_decimals``. 1 cm3 == 1 mL == 1000 mm3."""
        factor = 1.0 if self.volume_unit == "mm3" else 1.0 / 1000.0
        return round(float(mm3) * factor, self.volume_decimals)

    def volume_unit_label(self) -> str:
        return {"mm3": "mm³", "cm3": "cm³", "mL": "mL"}[self.volume_unit]

    def top_n(self) -> "int | None":
        """Max table rows for the current detail preset, or ``None`` for
        every row."""
        return {"summary": 5, "standard": 20, "full": None}[self.detail]

    def show_secondary(self) -> bool:
        """Whether secondary/auxiliary table columns should render."""
        return self.detail != "summary"

    def show_zero_rows(self) -> bool:
        """Whether zero-valued rows (e.g. 0%-disconnected tracts) should
        render, rather than being dropped for brevity."""
        return self.detail == "full"


DEFAULT_STYLE = ReportStyle()

#: A dark-background preset. Hosts needing something between light/dark
#: construct their own ``ReportStyle`` (or ``dataclasses.replace`` one of
#: these two).
DARK_STYLE = ReportStyle(
    accent="#ff8a75",
    text="#e8eaed",
    muted_text="#9aa4b2",
    border="#3a4048",
    surface="#20242b",
    row_stripe="#262b33",
    severity=SeverityScale(low="#5cd66c", medium="#f2c14e", high="#ff6b5e"),
)
