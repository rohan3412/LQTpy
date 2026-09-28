"""One wording and one number format for every output.

Reports, descriptions, the CLI and CSVs all take their labels from here,
so the same concept never appears under two names:

- **Volume**: always cm³, rounded by :func:`volume` (>= 100: whole numbers;
  >= 1: one decimal; >= 0.1: two; below that, three).
- **Quality**: ``OK / Note / Check / Problem`` (internal severities
  ``ok / info / warn / error``).
- **Deep grey matter**: the basal ganglia and thalamus. The atlases' lobe
  value ``Subcortical`` is shown as "deep grey matter" (short: "deep GM").
- **Vascular territories**: short form ``L MCA`` / ``R PCA`` /
  ``bilateral VB`` in tiles, tables and charts; long form "the left middle
  cerebral artery (MCA) territory" in running text.
"""
from __future__ import annotations

__all__ = ["volume", "volume_num", "quality_label", "QUALITY", "territory_short",
           "territory_long", "TERRITORY_NAMES", "lobe_word", "DEEP_GREY", "DEEP_GREY_SHORT",
           "side_short"]

UNIT = "cm³"


def volume_num(mm3: float) -> str:
    """The number part of :func:`volume` (no unit): tables with a cm³ header."""
    cc = float(mm3) / 1000.0
    if cc >= 100:
        return f"{cc:,.0f}"
    if cc >= 1:
        return f"{cc:.1f}"
    if cc >= 0.1:
        return f"{cc:.2f}"
    return f"{cc:.3f}"


def volume(mm3: float) -> str:
    """'15.2 cm³', '0.42 cm³', '0.035 cm³': the one volume format."""
    return f"{volume_num(mm3)} {UNIT}"


#: severity -> display word (same everywhere: tiles, tables, CLI, PDF).
QUALITY = {"ok": "OK", "info": "Note", "warn": "Check", "error": "Problem"}


def quality_label(severity: "str | None") -> str:
    return QUALITY.get(severity or "ok", str(severity))


DEEP_GREY = "deep grey matter"
DEEP_GREY_SHORT = "deep GM"

_LOBE_WORD = {"Frontal": "frontal lobe", "Parietal": "parietal lobe", "Temporal": "temporal lobe",
              "Occipital": "occipital lobe", "Insula": "insula", "Limbic": "limbic lobe",
              "Subcortical": DEEP_GREY, "Cerebellum": "cerebellum",
              "Brainstem": "brainstem", "Ventricle": "ventricles", "Unassigned": "unassigned"}
_LOBE_SHORT = {"Subcortical": DEEP_GREY, "Ventricle": "ventricles"}


def lobe_word(lobe: str, short: bool = False, capital: bool = False) -> str:
    """'Frontal' -> 'frontal lobe' (short: 'frontal'); 'Subcortical' ->
    'deep grey matter' in both forms."""
    if short:
        w = _LOBE_SHORT.get(lobe, lobe.lower())
    else:
        w = _LOBE_WORD.get(lobe, lobe.lower())
    return w[:1].upper() + w[1:] if capital else w


TERRITORY_NAMES = {"ACA": "anterior cerebral artery", "MCA": "middle cerebral artery",
                   "PCA": "posterior cerebral artery", "VB": "vertebrobasilar"}


def side_short(hemisphere: str) -> str:
    return {"left": "L", "right": "R", "bilateral": "bilateral", "midline": "midline"}.get(
        hemisphere or "", "")


def territory_short(hemisphere: str, territory: str) -> str:
    """'L MCA', 'R PCA', 'bilateral VB'."""
    s = side_short(hemisphere)
    return f"{s} {territory}".strip()


def territory_long(hemisphere: str, territory: str) -> str:
    """'the left middle cerebral artery (MCA) territory'; 'the vertebrobasilar
    (VB) territory on both sides'."""
    full = TERRITORY_NAMES.get(territory)
    name = f"{full} ({territory})" if full else territory
    if hemisphere in ("left", "right"):
        return f"the {hemisphere} {name} territory"
    if hemisphere == "bilateral":
        return f"the {name} territory on both sides"
    return f"the {name} territory"
