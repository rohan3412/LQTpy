"""Natural-language lesion descriptions, built from structured facts.

Two steps, kept separate so the text is deterministic and testable and the
same facts can feed reports, tables or (later) an optional LLM rewrite:

1. :func:`facts` computes a nested structure: whole lesion -> hemisphere ->
   lobe -> region, then clusters (separate parts), with volumes, shares,
   percent of each structure damaged, functional labels, QC caveats and the
   share of the lesion the atlas doesn't label.
2. :func:`describe` renders those facts with templates chosen by a
   **profile**:

   ``stroke``        one or a few lesions; lobe and region detail, each
                     separate part described on its own.
   ``small_vessel``  many small lesions (MS, small-vessel disease); a count
                     and distribution summary instead of a paragraph per
                     lesion, with only the largest described individually.
   ``tumour``        one dominant mass; where it's centred, what it extends
                     into, midline crossing, ventricle involvement.
   ``research``      neutral anatomical wording with explicit percentages.

Functional labels (e.g. "Broca's area") are location facts from the atlas
label table, never predictions of deficits.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import nibabel as nib

from .atlases import load_atlas
from .overlap import coverage as _coverage
from .overlap import region_overlap_from_mask
from . import _format as _fmt
from .prepared import prepare

__all__ = ["facts", "describe", "Description", "PROFILES"]


@dataclass(frozen=True)
class Profile:
    name: str
    #: A region is named if it holds >= this % of the (part of the) lesion ...
    min_pct_of_lesion: float = 5.0
    #: ... or has >= this % of itself damaged.
    min_pct_damaged: float = 10.0
    max_regions_per_lobe: int = 3
    max_lobes: int = 4
    #: Parts described individually (the rest are summarised).
    max_clusters: int = 5
    #: Parts smaller than this share of the lesion aren't described individually.
    min_cluster_share: float = 2.0
    show_percent_damaged: bool = True


PROFILES = {
    "stroke": Profile("stroke"),
    "small_vessel": Profile("small_vessel", min_pct_of_lesion=10.0, min_pct_damaged=50.0,
                            max_regions_per_lobe=2, max_lobes=3, max_clusters=3,
                            show_percent_damaged=False),
    "tumour": Profile("tumour", max_clusters=3),
    "research": Profile("research", min_pct_of_lesion=2.0, min_pct_damaged=5.0,
                        max_regions_per_lobe=5, max_lobes=6, max_clusters=10),
}


@dataclass
class Description:
    """``text``: the whole description as plain text (paragraphs separated
    by blank lines). ``sections``: ``[(heading, paragraph), ...]`` in reading
    order, plain. ``marked``: the same sections with the key findings wrapped
    in ``**`` (the main side, lobe, territory, dominant parts, functional
    areas); :meth:`html` renders them bold. ``facts``: the structured facts
    the text was rendered from."""

    text: str
    sections: "list[tuple[str, str]]" = field(default_factory=list)
    facts: dict = field(default_factory=dict)
    profile: str = "stroke"
    marked: "list[tuple[str, str]]" = field(default_factory=list)

    def __str__(self) -> str:
        return self.text

    def html(self, heading_tag: "str | None" = None) -> str:
        """Escaped HTML, one ``<p>`` per section, key findings in ``<strong>``."""
        import html as _html
        out = []
        for head, para in self.marked or self.sections:
            body = emphasis_html(para)
            if heading_tag:
                out.append(f"<{heading_tag}>{_html.escape(head)}</{heading_tag}><p>{body}</p>")
            else:
                out.append(f"<p>{body}</p>")
        return "".join(out)


def _b(text: str) -> str:
    """Mark a key finding (rendered bold in reports, stripped in plain text)."""
    return f"**{text}**"


def strip_emphasis(text: str) -> str:
    return text.replace("**", "")


def emphasis_html(text: str) -> str:
    """Escape ``text`` and turn ``**...**`` spans into ``<strong>``."""
    import html as _html
    parts = text.split("**")
    return "".join(_html.escape(p) if i % 2 == 0 else f"<strong>{_html.escape(p)}</strong>"
                   for i, p in enumerate(parts))


# --------------------------------------------------------------------------- #
# Facts
# --------------------------------------------------------------------------- #
def _vol_text(mm3: float) -> str:
    return _fmt.volume(mm3)


def _hemi_word(h):
    return {"L": "left", "R": "right", "Midline": "midline"}.get(h, "unlateralised")


def _hierarchy(mask, atlas, lesion_total: int, grid_affine, wm=None) -> dict:
    """Hemisphere -> lobe -> region breakdown of a mask on ``atlas``'s grid.

    With ``wm`` (a white-matter atlas on the same grid), voxels the main
    atlas doesn't label are looked up there: named white-matter structures
    per hemisphere, and only what neither atlas labels stays "unlabelled"."""
    records = region_overlap_from_mask(mask, atlas)
    by_id = atlas.labels
    hemis: "dict[str, dict]" = {}
    for r in records:
        m = by_id[r["LabelID"]]
        h = m.Hemisphere or "Midline"
        lobe = m.Lobe or "Unassigned"
        hd = hemis.setdefault(h, {"voxels": 0, "lobes": {}})
        hd["voxels"] += r["LesionVoxels"]
        ld = hd["lobes"].setdefault(lobe, {"voxels": 0, "regions": []})
        ld["voxels"] += r["LesionVoxels"]
        ld["regions"].append({
            "name": m.DisplayName or m.RegionName, "region": m.RegionName,
            "function": m.Function, "tissue": m.Tissue,
            "pct_of_lesion": round(100.0 * r["LesionVoxels"] / lesion_total, 1) if lesion_total else 0.0,
            "pct_damaged": r["PercentDamage"], "voxels": r["LesionVoxels"],
        })
    # Unlabelled voxels split by hemisphere (world x of the grid voxels).
    labels = atlas.label_array()
    known = np.zeros(int(labels.max()) + 1, bool)
    known[[k for k in by_id if 0 <= k < known.size]] = True
    lab_at = labels[mask]
    unl = ~known[np.clip(lab_at, 0, None)] | (lab_at <= 0)
    wm_ok = (wm is not None and wm.label_array().shape == labels.shape
             and np.allclose(np.asarray(wm.img.affine, float), grid_affine, atol=1e-3))
    wm_damage = {}
    if wm_ok:
        wrecs = region_overlap_from_mask(mask, wm)
        wm_damage = {r["LabelID"]: r["PercentDamage"] for r in wrecs}
    if unl.any():
        ijk = np.argwhere(mask)[unl]
        wl = wm.label_array()[mask][unl] if wm_ok else np.zeros(len(ijk), int)
        x = ijk @ grid_affine[0, :3] + grid_affine[0, 3]
        for side, sel in (("L", x < -1e-3), ("R", x > 1e-3), ("Midline", np.abs(x) <= 1e-3)):
            n = int(sel.sum())
            if not n:
                continue
            hd = hemis.setdefault(side, {"voxels": 0, "lobes": {}})
            hd["voxels"] += n
            named = wl[sel]
            hd["unlabelled_voxels"] = int(np.count_nonzero(named <= 0))
            if wm_ok:
                ids, cnt = np.unique(named[named > 0], return_counts=True)
                hd["wm_voxels"] = int(cnt.sum())
                hd["wm"] = [{"name": wm.labels[int(i)].DisplayName or wm.labels[int(i)].RegionName,
                             "voxels": int(c),
                             "pct_of_lesion": round(100.0 * int(c) / lesion_total, 1),
                             "pct_damaged": wm_damage.get(int(i), 0.0)}
                            for i, c in sorted(zip(ids, cnt), key=lambda t: -t[1])
                            if int(i) in wm.labels]
    out = []
    for h, hd in hemis.items():
        lobes = []
        for lobe, ld in hd["lobes"].items():
            regions = sorted(ld["regions"], key=lambda r: -r["voxels"])
            lobes.append({"lobe": lobe, "pct_of_lesion": round(100.0 * ld["voxels"] / lesion_total, 1),
                          "voxels": ld["voxels"], "regions": regions})
        lobes.sort(key=lambda d: -d["voxels"])
        out.append({"hemisphere": _hemi_word(h), "code": h,
                    "pct_of_lesion": round(100.0 * hd["voxels"] / lesion_total, 1) if lesion_total else 0.0,
                    "unlabelled_pct": round(100.0 * hd.get("unlabelled_voxels", 0) / lesion_total, 1)
                    if lesion_total else 0.0,
                    "wm_pct": round(100.0 * hd.get("wm_voxels", 0) / lesion_total, 1)
                    if lesion_total else 0.0,
                    "wm": hd.get("wm", []),
                    "lobes": lobes})
    out.sort(key=lambda d: -d["pct_of_lesion"])
    return {"hemispheres": out}


def facts(lesion, *, atlas: str = "dk", threshold: float = 0.5, profile: str = "stroke",
          qc_flags=None, shape: bool = False, networks: bool = True,
          white_matter: "str | None" = "jhu", territories: "str | None" = "arterial",
          tissue: bool = True) -> dict:
    """Structured facts for a description (see module docstring).

    ``atlas`` is the anatomical atlas for the hemisphere/lobe/region
    hierarchy (needs ``Lobe``; ``dk`` or ``aal``). ``qc_flags`` defaults to
    :func:`lqtpy.qc.check`. ``shape`` adds shape descriptors (extra tier).
    ``networks`` adds the Yeo-7 network involvement from ``schaefer``.
    ``white_matter`` names the white-matter atlas used for the part the main
    atlas doesn't label (``None`` to skip); ``territories`` the vascular
    territory atlas (``None`` to skip). ``tissue`` adds tissue composition
    and a location class per part (periventricular, juxtacortical, ...; see
    :mod:`lqtpy.tissue`).
    """
    from . import qc as _qc
    from .morphometry import compute_morphometry

    p = prepare(lesion, threshold)
    prof = PROFILES[profile]
    at = load_atlas(atlas)
    gm = p.on_grid(at.img)
    morph = compute_morphometry(p)
    cov = _coverage(p, at)
    flags = _qc.check(p) if qc_flags is None else qc_flags
    vox_mm3 = float(abs(np.linalg.det(p.affine[:3, :3])))

    wm_at = load_atlas(white_matter) if white_matter else None
    whole = _hierarchy(gm.mask, at, gm.n_voxels, np.asarray(at.img.affine, float), wm_at)
    out = {
        "profile": profile,
        "atlas": at.key,
        "atlas_name": at.name,
        "volume_mm3": morph["volume_mm3"],
        "n_components": morph["n_components"],
        "pct_left": morph["pct_left"], "pct_right": morph["pct_right"],
        "pct_midline": morph.get("pct_midline", 0.0),
        "bilateral": morph["bilateral"],
        "center_of_mass_mm": morph["center_of_mass_mm"],
        "coverage": cov,
        "hemispheres": whole["hemispheres"],
        "qc": [f.to_dict() if hasattr(f, "to_dict") else dict(f) for f in flags],
    }

    loc = None
    if tissue:
        from . import tissue as _tissue
        loc = _tissue.locate(p)
        out["tissue"] = {k: loc[k] for k in ("composition", "class", "counts", "contact_mm")}
        ut = _unlabelled_tissue(gm.mask, at, wm_at, by_side=True)
        out["unlabelled_tissue"] = ut.get("all", {})
        out["unlabelled_tissue_by_side"] = {k: v for k, v in ut.items() if k != "all"}

    # Clusters (gap-aware).
    cl = _qc.clusters(p)
    sizes = cl["sizes"]
    total = int(sizes.sum()) if len(sizes) else 0
    from scipy import ndimage as _nd
    cents = (_nd.center_of_mass(np.ones(cl["labels"].shape), cl["labels"],
                                index=np.arange(1, len(sizes) + 1)) if len(sizes) else [])
    clusters = []
    for i, n in enumerate(sizes):
        share = 100.0 * int(n) / total if total else 0.0
        cw = p.affine[:3, :3] @ (np.asarray(cents[i]) + cl["offset"]) + p.affine[:3, 3]
        entry = {"id": i + 1, "voxels": int(n), "volume_mm3": round(int(n) * vox_mm3, 1),
                 "share": round(share, 1), "described": False,
                 "center_mm": [round(float(v), 1) for v in cw],
                 "hemisphere": "left" if cw[0] < -1 else ("right" if cw[0] > 1 else "midline")}
        if loc is not None and i < len(loc["parts"]):
            lp = loc["parts"][i]
            entry.update({"location": lp["class"], "tissue_class": lp["tissue_class"],
                          "tissue": lp["composition"],
                          "dist_ventricle_mm": lp["dist_ventricle_mm"],
                          "dist_cortex_mm": lp["dist_cortex_mm"],
                          "max_axial_mm": lp["max_axial_mm"]})
        if i < prof.max_clusters and (share >= prof.min_cluster_share or i == 0) and len(sizes) > 1:
            cmask = np.zeros(p.shape, bool)
            sub = cl["labels"] == i + 1
            lo = cl["offset"]
            cmask[lo[0]:lo[0] + sub.shape[0], lo[1]:lo[1] + sub.shape[1],
                  lo[2]:lo[2] + sub.shape[2]] = sub
            cp = prepare(nib.Nifti1Image(cmask.astype(np.uint8), p.affine), threshold=0.0)
            cgm = cp.on_grid(at.img)
            entry["hemispheres"] = _hierarchy(cgm.mask, at, cgm.n_voxels,
                                              np.asarray(at.img.affine, float), wm_at)["hemispheres"]
            if territories:
                entry["territories"] = _territories(cp, territories)
            entry["described"] = True
            if shape:
                from .measures import shape as _shape
                entry["shape"] = [d.text for d in _shape.describe_shape(_shape.compute(cp))]
                entry["shape_levels"] = [d.level for d in _shape.describe_shape(_shape.compute(cp))]
        clusters.append(entry)
    out["clusters"] = clusters
    out["n_clusters"] = len(clusters)
    out["merged_gap_slabs"] = cl["gap_plane"] is not None and len(sizes) < morph["n_components"]

    if shape:
        from .measures import shape as _shape
        sv = _shape.compute(p)
        descs = _shape.describe_shape(sv)
        out["shape"] = {"values": {k: v for k, v in sv.items() if k != "per_component"},
                        "descriptors": [d.text for d in descs],
                        "levels": [d.level for d in descs],
                        "sentence": _shape.sentence(descs, "It" if len(sizes) <= 1 else "The largest part")}

    if territories:
        out["territories"] = _territories(p, territories)

    if networks:
        try:
            from .overlap import network_rollup
            sch = load_atlas("schaefer")
            sgm = p.on_grid(sch.img)
            recs = region_overlap_from_mask(sgm.mask, sch)
            out["networks"] = network_rollup(recs, sch, lesion_total=sgm.n_voxels)
        except Exception:
            out["networks"] = []

    fn = {}
    for h in out["hemispheres"]:
        for lobe in h["lobes"]:
            for r in lobe["regions"]:
                if r["function"]:
                    key = (h["hemisphere"], r["function"])
                    fn[key] = max(fn.get(key, 0.0), r["pct_damaged"])
    out["functional"] = [{"hemisphere": h, "function": f, "pct_damaged": v}
                         for (h, f), v in sorted(fn.items(), key=lambda kv: -kv[1])]
    return out


def _unlabelled_tissue(mask, atlas, wm_atlas, by_side: bool = False) -> dict:
    """Tissue classes of the lesion voxels neither ``atlas`` nor the
    white-matter atlas names, as % of the lesion (on ``atlas``'s grid).
    ``by_side``: ``{"all": {...}, "left": {...}, "right": {...}, "midline":
    {...}}`` instead (each still a share of the whole lesion)."""
    from . import tissue as _tissue
    from .atlases import load_atlas as _load
    ts = _load("tissue")
    grid = np.asarray(atlas.img.affine, float)
    lab = atlas.label_array()
    if ts.label_array().shape != lab.shape or not np.allclose(
            np.asarray(ts.img.affine, float), grid, atol=1e-3):
        return {}
    total = int(np.count_nonzero(mask))
    if not total:
        return {}
    lab_m = lab[mask]
    known = np.isin(lab_m, list(atlas.labels)) & (lab_m > 0)
    if wm_atlas is not None and wm_atlas.label_array().shape == lab.shape:
        wl = wm_atlas.label_array()[mask]
        known |= np.isin(wl, list(wm_atlas.labels)) & (wl > 0)
    if known.all():
        return {} if not by_side else {"all": {}}
    cls = _tissue._LUT[ts.label_array()[mask][~known]]
    counts = np.bincount(cls, minlength=len(_tissue._KEYS))
    whole = {k: round(100.0 * int(c) / total, 1) for k, c in zip(_tissue._KEYS, counts) if c}
    if not by_side:
        return whole
    out = {"all": whole}
    x = np.argwhere(mask)[~known] @ grid[0, :3] + grid[0, 3]
    for side, sel in (("left", x < -1e-3), ("right", x > 1e-3), ("midline", np.abs(x) <= 1e-3)):
        if sel.any():
            c = np.bincount(cls[sel], minlength=len(_tissue._KEYS))
            out[side] = {k: round(100.0 * int(v) / total, 1) for k, v in zip(_tissue._KEYS, c) if v}
    return out


#: Both halves of one territory hold at least this share of the lesion: it is
#: reported as one bilateral territory.
BILATERAL_TERRITORY_MIN = 0.2
_DEEP_WORDS = ("lenticulostriate", "thalamoperforator", "basilar")


def _territories(p, key: str) -> "list[dict]":
    """Major territory per hemisphere with its branch/perforator parts."""
    at = load_atlas(key)
    gm = p.on_grid(at.img)
    total = gm.n_voxels
    if not total:
        return []
    recs = region_overlap_from_mask(gm.mask, at)
    groups: "dict[tuple, dict]" = {}
    for r in recs:
        m = at.labels[r["LabelID"]]
        terr = m.Territory or m.Group
        if terr == "Ventricle":
            continue
        g = groups.setdefault((m.Hemisphere, terr), {"voxels": 0, "parts": []})
        g["voxels"] += r["LesionVoxels"]
        name = m.DisplayName or m.RegionName
        g["parts"].append({"name": name, "deep": any(w in name for w in _DEEP_WORDS),
                           "pct_of_lesion": round(100.0 * r["LesionVoxels"] / total, 1),
                           "pct_damaged": r["PercentDamage"]})
    # A lesion straddling the midline inside one territory (a pontine lesion in
    # the basilar territory, bilateral ACA) is one bilateral territory, not a
    # "left" one with a "right" remainder.
    for terr in {t for _, t in groups}:
        lft, rgt = groups.get(("L", terr)), groups.get(("R", terr))
        if (lft and rgt and min(lft["voxels"], rgt["voxels"]) >= BILATERAL_TERRITORY_MIN * total):
            merged = {"voxels": lft["voxels"] + rgt["voxels"], "parts": []}
            by_name: "dict[str, dict]" = {}
            for pp in lft["parts"] + rgt["parts"]:
                q = by_name.setdefault(pp["name"], {**pp, "pct_of_lesion": 0.0})
                q["pct_of_lesion"] = round(q["pct_of_lesion"] + pp["pct_of_lesion"], 1)
                q["pct_damaged"] = max(q["pct_damaged"], pp["pct_damaged"])
            merged["parts"] = list(by_name.values())
            del groups[("L", terr)], groups[("R", terr)]
            groups[("bilateral", terr)] = merged
    out = []
    for (hemi, terr), g in groups.items():
        out.append({"hemisphere": "bilateral" if hemi == "bilateral" else _hemi_word(hemi),
                    "territory": terr,
                    "pct_of_lesion": round(100.0 * g["voxels"] / total, 1),
                    "parts": sorted(g["parts"], key=lambda d: -d["pct_of_lesion"])})
    return sorted(out, key=lambda d: -d["pct_of_lesion"])


def _part_word(name: str, terr: str) -> str:
    """'middle cerebral artery territory, insular part' -> 'insular part';
    'lateral lenticulostriate territory' -> 'lateral lenticulostriate (deep)'."""
    if ", " in name:
        return name.split(", ", 1)[1]
    word = name.replace(" territory", "")
    if any(w in word for w in _DEEP_WORDS):
        return f"{word} (deep)"
    return word


def _territory_sentence(terrs: "list[dict]", subject: str = "It") -> str:
    """One sentence per major territory, key parts emphasised:
    'It lies in the **left middle cerebral artery (MCA) territory (100%)**:
    mainly its **frontal part (52%)** and insular part (37%), with 6% in the
    lateral lenticulostriate (deep perforator) territory.'"""
    main = [t for t in terrs if t["pct_of_lesion"] >= 5][:3]
    out = []
    for i, t in enumerate(main):
        name = _fmt.territory_long(t["hemisphere"], t["territory"])
        verb = "lies" if subject == "It" else "lie"
        head = f"{name} ({_pct(t['pct_of_lesion'])})"
        sent = (f"{subject} {verb} in {_b(head)}" if i == 0
                else f"{_pct(t['pct_of_lesion'])} is in {name}")
        branch = [pp for pp in t["parts"] if not pp["deep"] and pp["pct_of_lesion"] >= 5][:3]
        deep = [pp for pp in t["parts"] if pp["deep"] and pp["pct_of_lesion"] >= 2]
        top = t["parts"][0] if t["parts"] else None
        details = []
        if top and top["pct_of_lesion"] >= 0.9 * t["pct_of_lesion"]:
            # One part is (nearly) the whole territory: say so, with both numbers.
            what = (f"its {top['name'].replace(' territory', '')} (deep perforator) territory"
                    if top["deep"] else f"its {_part_word(top['name'], t['territory'])}")
            what = f"{what} ({_pct(top['pct_of_lesion'])})"
            details.append(f"almost entirely {_b(what) if i == 0 else what}")
        else:
            if len(t["parts"]) > 1 and branch:
                items = [f"{_part_word(pp['name'], t['territory'])} ({_pct(pp['pct_of_lesion'])})"
                         for pp in branch]
                if i == 0:
                    items[0] = _b(items[0])
                details.append("mainly its " + _join(items))
            for pp in deep:
                deep_name = pp["name"].replace(" territory", "")
                details.append(f"{_pct(pp['pct_of_lesion'])} in the {deep_name} "
                               "(deep perforator) territory")
        if details:
            sent += ": " + ", with ".join(details)
        out.append(sent + ".")
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _lobe_word(lobe: str) -> str:
    return _fmt.lobe_word(lobe)


def _join(items: "list[str]") -> str:
    items = [i for i in items if i]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _pct(v: float) -> str:
    return f"{v:.0f}%" if v >= 1 else "<1%"


def _region_phrase(r: dict, prof: Profile) -> str:
    name = r["name"]
    extras = []
    if r["function"]:
        extras.append(_b(r["function"]))
    if prof.show_percent_damaged:
        extras.append(f"{_pct(r['pct_damaged'])} of it damaged")
    return f"the {name}" + (f" ({'; '.join(extras)})" if extras else "")


def _named_regions(lobe: dict, prof: Profile) -> "list[dict]":
    keep = [r for r in lobe["regions"]
            if r["pct_of_lesion"] >= prof.min_pct_of_lesion or r["pct_damaged"] >= prof.min_pct_damaged]
    return keep[:prof.max_regions_per_lobe]


def _where_sentence(hemis: "list[dict]", prof: Profile, subject: str, atlas_label: str,
                    regions: bool = True, unl_tissue: "dict | None" = None) -> "list[str]":
    """Lobes per hemisphere in one sentence, then the unlabelled share,
    then (optionally) one sentence of named regions per lobe."""
    main = [h for h in hemis if h["pct_of_lesion"] >= 1]
    # Emphasise the largest compartment of the main hemisphere: its top lobe,
    # or its white matter when that holds more of the lesion.
    wm_main = bool(main) and main[0].get("wm_pct", 0) > max(
        (lb["pct_of_lesion"] for lb in main[0]["lobes"]), default=0)
    clauses, lobe_list = [], []
    for h in main:
        lobes = [lb for lb in h["lobes"] if lb["pct_of_lesion"] >= 1][:prof.max_lobes]
        if not lobes:
            continue
        side = h["hemisphere"]
        parts = [f"{_lobe_word(lb['lobe'])} ({_pct(lb['pct_of_lesion'])})" for lb in lobes]
        if parts and not clauses and not wm_main:
            parts[0] = _b(parts[0])  # the main lobe of the main hemisphere
        prefix = f"the {side} " if side in ("left", "right") else "the midline "
        clauses.append(prefix + _join(parts))
        lobe_list += [(side, lb) for lb in lobes]
    sentences = []
    if clauses:
        sentences.append(f"{subject} involves " + ("; and ".join(clauses) if len(clauses) > 1
                                                    else clauses[0]) + ".")
    wm_clauses = []
    for h in main:
        wms = [w for w in h.get("wm", []) if w["pct_of_lesion"] >= prof.min_pct_of_lesion
               or w["pct_damaged"] >= prof.min_pct_damaged][:prof.max_regions_per_lobe]
        if h.get("wm_pct", 0) >= 1:
            side = f"{h['hemisphere']} " if h["hemisphere"] in ("left", "right") else ""
            names = _join([f"the {w['name']}" + (f" ({_pct(w['pct_damaged'])} of it damaged)"
                                                  if prof.show_percent_damaged else "")
                           for w in wms])
            clause = f"{side}white matter ({_pct(h['wm_pct'])})"
            if wm_main and not wm_clauses:
                clause = _b(clause)
                if wms:
                    names = _join([(_b(f"the {w['name']}") if k == 0 else f"the {w['name']}")
                                   + (f" ({_pct(w['pct_damaged'])} of it damaged)"
                                      if prof.show_percent_damaged else "")
                                   for k, w in enumerate(wms)])
            wm_clauses.append(clause + (f", mainly {names}" if names else ""))
    if wm_clauses:
        sentences.append(f"{subject} also involves " + "; and ".join(wm_clauses) + ".")
    unl = sum(h.get("unlabelled_pct", 0.0) for h in hemis)
    if unl >= 1 and unl_tissue:
        sentences.append(_unlabelled_sentence(unl, unl_tissue))
    elif unl >= 1:
        share = "Most" if unl >= 60 else ("About half" if 40 <= unl < 60 else None)
        tail = ("lies in tissue no bundled atlas names (unnamed white matter, CSF or the brain edge)."
                if any(h.get("wm") is not None and "wm_pct" in h for h in hemis)
                else f"lies in white matter or other tissue the {atlas_label} atlas doesn't label.")
        if share:
            sentences.append(f"{share} of it ({_pct(unl)}) {tail}")
        else:
            sentences.append(f"{_pct(unl)} of it {tail}")
    if regions:
        for side, lb in lobe_list:
            regs = _named_regions(lb, prof)
            if regs:
                where = (f"In the {side} {_lobe_word(lb['lobe'])}" if side in ("left", "right")
                         else f"In the {_lobe_word(lb['lobe'])}")
                sentences.append(f"{where} it mainly involves "
                                 f"{_join([_region_phrase(r, prof) for r in regs])}.")
    return sentences


_UNL_WORDS = (("white_matter", "in unnamed white matter"), ("csf", "in CSF spaces"),
              ("outside", "outside the brain"))


def _unlabelled_sentence(unl: float, ut: dict, whole: str = "it") -> str:
    """'17% of it lies outside the named structures: 12% in unnamed white
    matter and 3% in CSF spaces.'"""
    share = "Most" if unl >= 60 else ("About half" if 40 <= unl < 60 else None)
    head = (f"{share} of {whole} ({_pct(unl)})" if share else f"{_pct(unl)} of {whole}")
    items = [(w, ut.get(k, 0.0)) for k, w in _UNL_WORDS if ut.get(k, 0.0) >= 1]
    # The rest is almost all cortex: the template's cortical ribbon is wider
    # than the DKT regions (its outer edge in particular).
    rest = sum(v for k, v in ut.items() if k not in ("white_matter", "csf", "outside"))
    if rest >= 1:
        items.append(("in cortex just outside the named regions", rest))
    items.sort(key=lambda t: -t[1])
    if not items:
        return f"{head} lies at the edges of named structures."
    return (f"{head} lies outside the named structures: "
            + _join([f"{_pct(v)} {w}" for w, v in items]) + ".")


_TISSUE_ORDER = ("cortex", "white_matter", "deep_grey", "cerebellum", "brainstem",
                 "ventricles", "csf", "outside")
_LOC_PHRASE = {"infratentorial": "infratentorial", "cortical": "cortical",
               "deep grey matter": "in deep grey matter", "periventricular": "periventricular",
               "juxtacortical": "juxtacortical", "deep white matter": "in deep white matter",
               "outside the brain": "outside the brain"}


def _tissue_list(comp: dict, min_pct: float = 2.0) -> str:
    from .tissue import TISSUE_WORDS
    items = [(k, comp.get(k, 0.0)) for k in _TISSUE_ORDER if comp.get(k, 0.0) >= min_pct]
    items.sort(key=lambda t: -t[1])
    return _join([f"{TISSUE_WORDS[k]} ({_pct(v)})" for k, v in items])


#: "Lacunar-sized": one subcortical part at most this wide across an axial
#: plane (the STRIVE recent small subcortical infarct limit) ...
LACUNAR_MAX_AXIAL_MM = 20.0
#: ... and at most this volume. Width alone let a 3.5 cm³ lesion that is
#: 19 mm across but tall through: a 20 mm sphere is 4.2 cm³, a typical
#: lacune well under 1.5 cm³.
LACUNAR_MAX_MM3 = 1500.0


def _tissue_sentence(f: dict, prof: Profile) -> str:
    """'By tissue it is **cortico-subcortical**: cortex (53%), white matter
    (35%) and deep grey matter (9%).' A single small subcortical lesion is
    called lacunar-sized (at most 20 mm across and 1.5 cm³); a single lesion reaching the
    ventricle or cortex is called periventricular / juxtacortical."""
    t = f.get("tissue") or {}
    cls = t.get("class")
    if not cls:
        return ""
    lst = _tissue_list(t["composition"])
    n = f["n_clusters"]
    first = f["clusters"][0] if f.get("clusters") else {}
    what = _b(cls)
    if (cls == "subcortical" and n <= 1 and prof.name == "stroke"
            and first.get("max_axial_mm", 99) <= LACUNAR_MAX_AXIAL_MM
            and f["volume_mm3"] <= LACUNAR_MAX_MM3):
        what = (f"a {_b('small subcortical lesion')} ({first['max_axial_mm']:.0f} mm across, "
                "lacunar-sized)")
    sent = f"By tissue it is {what}" + (f": {lst}." if lst else ".")
    if n <= 1 and first.get("location") in ("periventricular", "juxtacortical"):
        where = "the lateral ventricle" if first["location"] == "periventricular" else "the cortex"
        sent += f" It reaches {where} ({first['location']})."
    return sent


def _count_words(counts: dict) -> str:
    """'**12 periventricular**, 5 cortical or juxtacortical and 4 in deep
    white matter'."""
    merged = dict(counts)
    cj = merged.pop("cortical", 0) + merged.pop("juxtacortical", 0)
    items = [(v, _LOC_PHRASE[k]) for k, v in merged.items() if v]
    if cj:
        items.append((cj, "cortical or juxtacortical"))
    items.sort(key=lambda t: -t[0])
    words = [f"{v} {w}" for v, w in items]
    if words:
        words[0] = _b(words[0])
    return _join(words)


def _midline_structures(f: dict) -> "tuple[float, str | None]":
    """Share of the lesion in structures the atlas puts on the midline
    (brainstem, vermis, ...), and the largest one's name."""
    for h in f.get("hemispheres", []):
        if h.get("code") == "Midline":
            top = h["lobes"][0]["lobe"] if h["lobes"] else None
            return h["pct_of_lesion"], (_lobe_word(top) if top else None)
    return 0.0, None


def _side_summary(f: dict) -> str:
    """Where the lesion sits across the hemispheres. The share on the
    midline plane is never dropped: 40% left + 40% right says so when the
    other 20% is on the midline."""
    pl, pr, pm = f["pct_left"], f["pct_right"], f.get("pct_midline", 0.0)
    mid_share, mid_name = _midline_structures(f)
    split = f"{_pct(pl)} left, {_pct(pr)} right" + (f", {_pct(pm)} on the midline" if pm >= 1 else "")
    if mid_share >= 50:
        where = f"on the midline ({mid_name})" if mid_name else "on the midline"
        return f"{where}, {split}"
    if pl >= 95:
        return "in the left hemisphere"
    if pr >= 95:
        return "in the right hemisphere"
    if max(pl, pr) >= 75:
        main, other, op = ("left", "right", pr) if pl > pr else ("right", "left", pl)
        tail = f"{_pct(op)} {other}" + (f", {_pct(pm)} on the midline" if pm >= 1 else "")
        return f"mainly in the {main} hemisphere, crossing the midline ({tail})"
    return f"in both hemispheres ({split})"


def _caveats(f: dict) -> str:
    flags = [q for q in f["qc"] if q["severity"] in ("warn", "error") and q["id"] != "input"]
    if not flags:
        return ""
    lines = [q["message"] for q in flags]
    return "Quality notes: " + " ".join(lines)


def _atlas_label(f):
    return {"dk": "DKT", "aal": "AAL"}.get(f["atlas"], f["atlas"])


def _render(f: dict) -> "list[tuple[str, str]]":
    prof = PROFILES[f["profile"]]
    sections: "list[tuple[str, str]]" = []
    vol = _vol_text(f["volume_mm3"])
    n = f["n_clusters"]
    atlas_label = _atlas_label(f)

    if prof.name == "small_vessel" and n > 3:
        # Distribution summary, not a paragraph per lesion.
        described = [c for c in f["clusters"] if c.get("described")]
        sizes = sorted((c["volume_mm3"] for c in f["clusters"]), reverse=True)
        med = float(np.median(sizes))
        nl = sum(1 for c in f["clusters"] if c.get("hemisphere") == "left")
        nr = sum(1 for c in f["clusters"] if c.get("hemisphere") == "right")
        nm = n - nl - nr
        split = f" ({nl} left, {nr} right" + (f", {nm} midline" if nm else "") + ")"
        head = (f"{_b(f'{n} separate lesions')}{split} totalling {_b(vol)}. "
                f"The largest is {_vol_text(sizes[0])}; the median is {_vol_text(med)}.")
        if f.get("tissue") and any(f["tissue"]["counts"].values()):
            head += f" By location: {_count_words(f['tissue']['counts'])}."
        sections.append(("Summary", head))
        dist = []
        for h in f["hemispheres"]:
            lobes = [lb for lb in h["lobes"] if lb["pct_of_lesion"] >= 5][:prof.max_lobes]
            if lobes and h["hemisphere"] in ("left", "right"):
                dist.append(f"{h['hemisphere']} {_join([_lobe_word(lb['lobe']) for lb in lobes])}")
        unl = sum(h.get("unlabelled_pct", 0.0) for h in f["hemispheres"])
        wm_total = sum(h.get("wm_pct", 0.0) for h in f["hemispheres"])
        wm_names: "dict[str, float]" = {}
        for h in f["hemispheres"]:
            for w in h.get("wm", []):
                wm_names[w["name"]] = wm_names.get(w["name"], 0.0) + w["pct_of_lesion"]
        txt = ""
        if dist:
            txt = f"Grey-matter involvement is mainly in the {'; '.join(dist)}."
        if wm_total >= 1:
            top = [k for k, v in sorted(wm_names.items(), key=lambda kv: -kv[1]) if v >= 5][:3]
            txt += (f" {_pct(wm_total)} of the total volume is in named white matter"
                    + (f", mainly the {_join(top)}" if top else "") + ".")
        if unl >= 1 and f.get("unlabelled_tissue"):
            txt += " " + _unlabelled_sentence(unl, f["unlabelled_tissue"], "the total volume")
        elif unl >= 1:
            txt += f" {_pct(unl)} lies in tissue no bundled atlas names."
        if txt:
            sections.append(("Distribution", txt.strip()))
        if described:
            parts = []
            for c in described:
                where = _cluster_where(c)
                loc = f", {c['location']}" if c.get("location") else ""
                parts.append(f"Lesion {c['id']} ({_vol_text(c['volume_mm3'])}{loc}) {where}.")
            sections.append(("Largest lesions", " ".join(parts)))
    else:
        if n <= 1:
            opener = f"A single {_b(vol)} lesion {_b(_side_summary(f))}."
        else:
            opener = (f"A {_b(vol)} lesion in {_b(f'{n} separate parts')}, {_side_summary(f)}.")
        if f.get("merged_gap_slabs"):
            opener += (" Pieces that look like slices of one gap-drawn lesion are counted"
                       " as one part.")
        sections.append(("Summary", opener))
        where = _where_sentence(f["hemispheres"], prof, "It" if n <= 1 else "Overall it",
                                atlas_label, regions=n <= 1,
                                unl_tissue=f.get("unlabelled_tissue"))
        if where:
            sections.append(("Location", " ".join(where)))
        ts = _tissue_sentence(f, prof)
        if ts:
            sections.append(("Tissue", ts))
        if prof.name == "tumour":
            extra = []
            vent = [lb for h in f["hemispheres"] for lb in h["lobes"] if lb["lobe"] == "Ventricle"]
            if vent:
                extra.append("It extends into the ventricular system.")
            if extra:
                sections.append(("Extent", " ".join(extra)))
        if n > 1:
            parts = []
            for c in f["clusters"]:
                if not c.get("described"):
                    continue
                txt = (f"Part {c['id']} ({_vol_text(c['volume_mm3'])}, {_pct(c['share'])} of the "
                       f"lesion) {_cluster_where(c)}")
                # Stroke/tumour parts: cortical/subcortical; other profiles: the
                # MS-style location class (periventricular, juxtacortical, ...).
                if prof.name in ("stroke", "tumour") and c.get("tissue_class"):
                    txt += f"; it is {c['tissue_class']}"
                elif c.get("location"):
                    txt += f"; it is {_LOC_PHRASE[c['location']]}"
                txt += "."
                blocked = any(q["id"] in _SHAPE_BLOCKERS and q["severity"] != "info"
                              for q in f["qc"])
                if c.get("shape") and not blocked:
                    txt += f" It is {_shape_predicate(c)}."
                parts.append(txt)
            rest = [c for c in f["clusters"] if not c.get("described")]
            if rest:
                rv = sum(c["volume_mm3"] for c in rest)
                parts.append(f"{len(rest)} smaller part{'s' if len(rest) != 1 else ''} "
                             f"total {_vol_text(rv)}.")
            sections.append(("Parts", " ".join(parts)))

    if f.get("territories"):
        ts = _territory_sentence(f["territories"],
                                 "The lesions" if prof.name == "small_vessel" and n > 3 else "It")
        if ts:
            sections.append(("Vascular territory", ts))

    if f.get("functional"):
        items = [_b(f"{x['hemisphere']} {x['function']}") if x["hemisphere"] in ("left", "right")
                 else _b(x["function"]) for x in f["functional"][:4]]
        sections.append(("Functional areas",
                         f"It involves structures labelled {_join(items)}. These are location "
                         "labels, not predictions of symptoms."))

    if f.get("networks"):
        nets = sorted((r for r in f["networks"] if r["PercentOfLesion"] >= 5),
                      key=lambda r: -r["PercentOfLesion"])[:3]
        if nets:
            from .atlases import _YEO7_NAMES
            names = [f"{_YEO7_NAMES.get(r['Network'], r['Network'])} "
                     f"({_pct(r['PercentOfLesion'])}{' of the lesion' if i == 0 else ''})"
                     for i, r in enumerate(nets)]
            sections.append(("Networks", "By Yeo-7 functional network (Schaefer parcels), it "
                             "overlaps the " + _join(names) + "."))

    if f.get("shape") and prof.name != "small_vessel" and n <= 1:
        bad = [q for q in f["qc"] if q["id"] in _SHAPE_BLOCKERS and q["severity"] != "info"]
        if bad:
            sections.append(("Shape", "Shape isn't described because the mask looks "
                             + _SHAPE_BLOCKERS[bad[0]["id"]] + "."))
        else:
            sections.append(("Shape", f["shape"]["sentence"]))

    cav = _caveats(f)
    if cav:
        sections.append(("Quality", cav))
    return sections


_SHAPE_BLOCKERS = {"sparse_slices": "drawn on some slices only",
                   "hollow": "like an unfilled outline",
                   "flat": "drawn on a single slice"}


def _cluster_where(c: dict) -> str:
    hemis = c.get("hemispheres") or []
    if not hemis:
        return "lies outside the labelled atlas regions"
    h = hemis[0]
    lobes = [lb for lb in h["lobes"] if lb["pct_of_lesion"] >= 10][:2]
    side = h["hemisphere"]
    if not lobes:
        wms = [w for w in h.get("wm", []) if w["pct_of_lesion"] >= 10][:2]
        if wms:
            return f"lies in {side} white matter, mainly the {_join([w['name'] for w in wms])}"
        if h.get("unlabelled_pct", 0) >= 50:
            return f"lies in {side} white matter or other unlabelled tissue"
        return f"is in the {side} hemisphere"
    top = lobes[0]["regions"][0]["name"] if lobes[0]["regions"] else None
    where = f"is in the {side} {_join([_lobe_word(lb['lobe']) for lb in lobes])}"
    if top:
        where += f", mainly the {top}"
    return where


def _shape_predicate(c: dict) -> str:
    from .measures.shape import _FEATURE_LEVELS
    head = [t for t, lv in zip(c["shape"], c["shape_levels"]) if lv not in _FEATURE_LEVELS]
    feats = [t for t, lv in zip(c["shape"], c["shape_levels"]) if lv in _FEATURE_LEVELS]
    out = ", ".join(head)
    if feats:
        out += (", with " if out else "shaped with ") + " and ".join(feats)
    return out


def describe(lesion, *, profile: str = "stroke", atlas: str = "dk", threshold: float = 0.5,
             qc_flags=None, shape: bool = False, networks: bool = True,
             white_matter: "str | None" = "jhu", territories: "str | None" = "arterial",
             tissue: bool = True) -> Description:
    """A plain-language description of a lesion.

    ``profile``: ``stroke`` (default), ``small_vessel``, ``tumour`` or
    ``research`` -- see the module docstring. ``shape=True`` adds shape
    descriptors (extra tier); ``tissue=False`` drops the tissue composition
    and location classes. Returns a :class:`Description` whose ``text``
    is ready to show and whose ``facts`` hold every number behind it.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    f = facts(lesion, atlas=atlas, threshold=threshold, profile=profile, qc_flags=qc_flags,
              shape=shape, networks=networks, white_matter=white_matter, territories=territories,
              tissue=tissue)
    marked = _render(f)
    sections = [(h, strip_emphasis(p)) for h, p in marked]
    text = "\n\n".join(p for _, p in sections)
    return Description(text=text, sections=sections, facts=f, profile=profile, marked=marked)
