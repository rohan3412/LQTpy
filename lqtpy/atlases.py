"""Atlas registry.

Six bundled atlases, all resolved lazily to NIfTI + a label table. ``aal``
is the default/first atlas:

- ``aal``      : AAL 116 anatomical regions (labels derived by spatial overlap
                 with nilearn's AAL SPM12, stored in ``data/aal_labels.csv``).
- ``schaefer`` : Schaefer-2018 100 cortical parcels / 7 Yeo networks.
- ``dk``       : Desikan-Killiany-Tourville (DKT31) cortex + CMA subcortical
                 structures, from the Mindboggle-101 joint-fusion volume
                 atlas (CC BY 4.0; see ``data/atlases/SOURCE.md``), 96
                 regions total (62 cortical + 34 subcortical/ventricle/
                 cerebellar). The 62 cortical regions are renderable with
                 ggseg; the label table is ``data/dkt31_cma_labels.csv``.
- ``jhu``      : JHU ICBM-DTI-81, 48 white-matter structures (internal
                 capsule limbs, corona radiata, corpus callosum, ...).
- ``arterial`` : Johns Hopkins arterial territories (Liu et al. 2023),
                 30 territories + lateral ventricles; ``Territory`` is the
                 major territory (ACA/MCA/PCA/VB). Its released header has
                 no origin; see ``tools/build_extra_atlases.py``.
- ``tissue``   : one tissue class per voxel (cortex, cerebral white matter,
                 deep grey matter, ventricles, cerebellum, brainstem, CSF),
                 by hemisphere; built by ``tools/build_tissue_map.py``. Used
                 by :mod:`lqtpy.tissue` for location words (periventricular,
                 juxtacortical, ...); not a default table atlas.

A host application can add its own atlases at runtime with
:func:`register_atlas`/:func:`atlas_from_arrays`, and a plugin package can
contribute one through the ``lqtpy.atlases`` entry point group (see
:mod:`lqtpy.registry`); once registered they work through
:func:`load_atlas`, :func:`list_atlases`, and every function in
:mod:`lqtpy.api`.

Label-table schema (v2)
-----------------------
Every region is a :class:`RegionMeta`. ``RegionName`` and ``Group`` are
required; ``Lobe``, ``DisplayName``, ``Hemisphere``, ``Tissue``, ``Network``,
``Function`` and ``Territory`` are optional. The natural-language
description and the reports use whichever fields an atlas provides.

Thread safety
--------------
``lqtpy`` is meant to be called in-process from a web server handling
concurrent requests on a threadpool. Every atlas is built once (lazily, on
first use) and cached; concurrent first callers race on a lock but all end
up sharing the *same* built :class:`Atlas` object, never racing to build it
twice or seeing a half-built one. The label volume and per-label voxel
counts a built ``Atlas`` computes lazily are guarded the same way, and the
resulting cached arrays are made read-only so one request can't corrupt
another's data. Call :func:`preload` at process startup to build every cache
up front (so first-request latency doesn't include atlas loading), and
:func:`cache_clear` to drop everything and force a rebuild.
"""
from __future__ import annotations

import atexit
import csv
import gzip
import os
import threading
from contextlib import ExitStack
from dataclasses import dataclass, field
from importlib import resources
from typing import NamedTuple

import numpy as np
import nibabel as nib

from .errors import DataUnavailableError, InvalidAtlasError, UnknownAtlasError
from .io import as_lesion_image

_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data")

# Atlas volumes/labels are staged inside the package (``lqtpy/data/atlases/``)
# and must resolve from an installed wheel in site-packages, where there is no
# repo root to walk up to. ``as_file`` hands back the real filesystem path for
# a normal on-disk install; it only extracts to a temp file for a zipped
# package. Either way nibabel loads volumes lazily, so the path has to stay
# valid after this function returns -- the ExitStack keeps it alive for the
# life of the process and is cleaned up at exit.
_resource_stack = ExitStack()
atexit.register(_resource_stack.close)


def _atlas_path(*parts: str) -> str:
    ref = resources.files("lqtpy").joinpath("data", "atlases", *parts)
    return str(_resource_stack.enter_context(resources.as_file(ref)))


class RegionMeta(NamedTuple):
    """Per-region metadata (label-table schema v2).

    Only ``RegionName`` and ``Group`` are required; every other field is
    optional and ``None`` when an atlas doesn't define it. Fields are
    appended (never reordered), so positional construction from older code
    keeps working.
    """

    RegionName: str
    Group: str
    #: Anatomical lobe (Frontal/Parietal/Temporal/Occipital/Limbic/Insula/
    #: Subcortical/Cerebellum/...) -- independent of ``Group``, which is
    #: hemisphere for aal/dk and Yeo-7 network for schaefer. ``None`` when
    #: the atlas doesn't define a lobe for this region (currently schaefer).
    #: See ``data/aal_labels.csv``/``data/dkt31_cma_labels.csv`` for the
    #: mappings and their sourcing.
    Lobe: "str | None" = None
    #: Readable name without hemisphere ("inferior frontal gyrus, opercular part").
    DisplayName: "str | None" = None
    #: "L", "R" or "Midline"; derived from the region name when not given.
    Hemisphere: "str | None" = None
    #: Cortex / Deep grey matter / Cerebellum / White matter / CSF / Brainstem / Other.
    Tissue: "str | None" = None
    #: Functional network (e.g. Yeo-7 "Default").
    Network: "str | None" = None
    #: Conservative common functional name ("primary motor cortex", "Broca's area").
    Function: "str | None" = None
    #: Vascular territory (e.g. "MCA"), for territory atlases.
    Territory: "str | None" = None

    def display(self) -> str:
        """Readable name with hemisphere ("left inferior frontal gyrus,
        opercular part"). Falls back to ``RegionName``."""
        base = self.DisplayName or self.RegionName
        side = {"L": "left ", "R": "right "}.get(self.Hemisphere or "", "")
        return side + base


#: Optional schema-v2 fields a label table / dict label may carry.
OPTIONAL_FIELDS = ("Lobe", "DisplayName", "Hemisphere", "Tissue", "Network",
                   "Function", "Territory")


def _hemisphere_from_name(name: str) -> "str | None":
    """Best-effort hemisphere from common atlas naming conventions
    (``_L``/``_R`` suffix, ``left_``/``right_`` prefix, Schaefer ``_LH_``)."""
    n = str(name).lower()
    if n.endswith("_l") or n.startswith(("left_", "left-", "lh_")) or "_lh_" in n:
        return "L"
    if n.endswith("_r") or n.startswith(("right_", "right-", "rh_")) or "_rh_" in n:
        return "R"
    return None


def _meta(name: str, group: str, fields: dict) -> RegionMeta:
    extra = {f: (fields.get(f) or None) for f in OPTIONAL_FIELDS}
    if extra["Hemisphere"] is None:
        # Bundled aal/dk use Group for hemisphere, which also knows midline
        # structures; otherwise derive from the name, else leave unknown.
        extra["Hemisphere"] = ({"Left": "L", "Right": "R", "Midline": "Midline"}.get(group)
                               or _hemisphere_from_name(name))
    return RegionMeta(RegionName=name, Group=group, **extra)


def _smallest_signed_dtype(max_label: int) -> np.dtype:
    """Smallest signed integer numpy dtype that can represent ``max_label``."""
    for dt in (np.int8, np.int16, np.int32, np.int64):
        if max_label <= np.iinfo(dt).max:
            return np.dtype(dt)
    return np.dtype(np.int64)


@dataclass
class Atlas:
    """A loaded atlas: label volume plus a per-region metadata table."""

    key: str
    name: str
    img: nib.Nifti1Image
    #: dict mapping integer label id -> RegionMeta.
    labels: "dict[int, RegionMeta]"
    #: True when regions can be rendered as a ggseg Desikan-Killiany flat map.
    ggseg: bool = False
    #: Maps RegionName -> ggseg region key (only for ggseg atlases).
    ggseg_map: dict = field(default_factory=dict)
    #: "core" (bundled, always available) or "extra" (host/plugin).
    tier: str = "core"
    #: Lazily-cached label volume + per-label voxel counts (subject-invariant).
    _label_array: "np.ndarray | None" = field(default=None, repr=False)
    _region_voxels: "np.ndarray | None" = field(default=None, repr=False)
    #: Guards the double-checked lazy build of the two arrays above so
    #: concurrent first callers build exactly once and all see the same
    #: (read-only) array, never a half-built one.
    _build_lock: threading.Lock = field(default_factory=threading.Lock,
                                        repr=False, compare=False)

    def label_array(self) -> np.ndarray:
        """Integer label volume, loaded once and cached on the atlas.

        Uses the smallest signed integer dtype that can hold this atlas's
        max label id (computed per atlas, not hard-coded), so a
        cortex-only atlas with ~100 regions takes a fraction of the memory
        a generic int32 volume would. The returned array is read-only.
        """
        if self._label_array is None:
            with self._build_lock:
                if self._label_array is None:
                    raw = np.asanyarray(self.img.dataobj)
                    if raw.dtype.kind == "f":
                        raw = np.rint(raw)
                    arr32 = raw.astype(np.int32)
                    max_label = int(arr32.max()) if arr32.size else 0
                    dtype = _smallest_signed_dtype(max_label)
                    arr = arr32.astype(dtype)
                    arr.flags.writeable = False
                    self._label_array = arr
        return self._label_array

    def region_voxels(self) -> np.ndarray:
        """Voxel count per label id (index = label id), computed once.

        The returned array is read-only.
        """
        if self._region_voxels is None:
            with self._build_lock:
                if self._region_voxels is None:
                    counts = np.bincount(self.label_array().ravel())
                    counts.flags.writeable = False
                    self._region_voxels = counts
        return self._region_voxels


# --------------------------------------------------------------------------- #
# Label-table builders
# --------------------------------------------------------------------------- #
def _read_csv_rows(path: str):
    """Yield dict rows from a CSV (or gzipped CSV) file via csv.DictReader.

    Lines starting with ``#`` (leading whitespace allowed) are skipped
    before the header row -- ``aal_labels.csv``/``dkt31_cma_labels.csv`` use
    this for a provenance/citation comment ahead of the real header.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", newline="", encoding="utf-8") as fh:
        lines = (line for line in fh if not line.lstrip().startswith("#"))
        yield from csv.DictReader(lines)


_YEO7_NAMES = {"Vis": "visual network", "SomMot": "somatomotor network",
               "DorsAttn": "dorsal attention network",
               "SalVentAttn": "salience / ventral attention network",
               "Limbic": "limbic network", "Cont": "frontoparietal control network",
               "Default": "default mode network"}


def _schaefer_display(name: str) -> str:
    """'7Networks_LH_SalVentAttn_FrOperIns_1' -> 'salience / ventral
    attention network parcel (FrOperIns 1)'."""
    parts = name.split("_")
    net = _YEO7_NAMES.get(parts[2], parts[2]) if len(parts) > 2 else name
    rest = " ".join(parts[3:])
    return f"{net} parcel ({rest})" if rest else f"{net} parcel"


def _schaefer_labels() -> "dict[int, RegionMeta]":
    path = _atlas_path(
        "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.csv")
    # RegionName like "7Networks_LH_Vis_1" -> network "Vis", hemi "LH".
    def network(name: str) -> str:
        parts = name.split("_")
        return parts[2] if len(parts) > 2 else "NA"
    out: "dict[int, RegionMeta]" = {}
    for i, row in enumerate(_read_csv_rows(path), start=1):
        name = str(row["RegionName"])
        net = network(name)
        out[i] = _meta(name, net, {"Network": net, "Tissue": "Cortex",
                                   "DisplayName": _schaefer_display(name)})
    return out


def _aal_labels() -> "dict[int, RegionMeta]":
    path = os.path.join(DATA_DIR, "aal_labels.csv")
    def hemi(name: str) -> str:
        if name.endswith("_L"):
            return "Left"
        if name.endswith("_R"):
            return "Right"
        return "Midline"
    out: "dict[int, RegionMeta]" = {}
    for row in _read_csv_rows(path):
        lid = int(row["LabelID"])
        name = str(row["RegionName"])
        out[lid] = _meta(name, hemi(name), row)
    return out


def _fs_to_ggseg(name: str) -> str | None:
    """'left_caudalanteriorcingulate' -> 'caudalanteriorcingulate_left'
    (also handles inter-word underscores, e.g. the old vendor naming
    'left_caudal_anterior_cingulate' -> same result)."""
    for side in ("left_", "right_"):
        if name.startswith(side):
            return name[len(side):].replace("_", "") + "_" + side[:-1]
    return None


def _dk_labels() -> "tuple[dict[int, RegionMeta], dict]":
    """DKT31 cortex + CMA subcortical label table (see
    ``data/dkt31_cma_labels.csv`` and ``data/atlases/SOURCE.md`` for
    provenance). Region ids >= 1000 are the 62 cortical DKT31 regions
    (FreeSurfer aparc numbering); everything else is CMA subcortical/
    ventricle/cerebellar. Only the cortical regions get a ggseg mapping --
    ``lqtpy/data/ggseg_dk.csv.gz`` only has cortical polygon geometry."""
    path = os.path.join(DATA_DIR, "dkt31_cma_labels.csv")
    labels: "dict[int, RegionMeta]" = {}
    gmap: dict = {}
    for row in _read_csv_rows(path):
        lid = int(row["LabelID"])
        name = str(row["RegionName"])
        if name.startswith("left_"):
            group = "Left"
        elif name.startswith("right_"):
            group = "Right"
        else:
            group = "Midline"
        labels[lid] = _meta(name, group, row)
        if lid >= 1000:  # cortical DKT31 range -- see the docstring above.
            g = _fs_to_ggseg(name)
            if g is not None:
                gmap[name] = g
    return labels, gmap


def _table_labels(filename: str):
    """Generic schema-v2 label table (``LabelID``, ``RegionName``, ``Group``
    plus optional fields) from ``lqtpy/data``."""
    def load() -> "dict[int, RegionMeta]":
        out = {}
        for row in _read_csv_rows(os.path.join(DATA_DIR, filename)):
            out[int(row["LabelID"])] = _meta(str(row["RegionName"]), row.get("Group") or "", row)
        return out
    return load


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "aal": dict(
        name="AAL (116 anatomical regions)",
        img=lambda: _atlas_path("aal.nii.gz"),
        labels=_aal_labels,
        ggseg=False,
        license="Free for academic use with citation",
        citation="Tzourio-Mazoyer N, et al. NeuroImage 15(1):273-289, 2002.",
    ),
    "schaefer": dict(
        name="Schaefer-100 / 7 Networks",
        img=lambda: _atlas_path(
            "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.nii.gz"),
        labels=_schaefer_labels,
        ggseg=False,
        license="Free for academic use with citation",
        citation="Schaefer A, et al. Cerebral Cortex 28(9):3095-3114, 2018.",
    ),
    "dk": dict(
        name="Desikan-Killiany-Tourville (DKT31 + CMA)",
        img=lambda: _atlas_path("OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz"),
        labels=_dk_labels,
        ggseg=True,
        license="CC BY 4.0",
        citation="Klein A, Tourville J. Front Neurosci 6:171, 2012.",
    ),
    "jhu": dict(
        name="JHU ICBM-DTI-81 white matter",
        img=lambda: _atlas_path("JHU-ICBM-labels-1mm.nii.gz"),
        labels=_table_labels("jhu_labels.csv"),
        ggseg=False,
        license="FSL atlas terms (academic use with citation)",
        citation="Mori S, et al. NeuroImage 40(2):570-582, 2008.",
    ),
    "arterial": dict(
        name="Arterial territories (Johns Hopkins)",
        img=lambda: _atlas_path("ArterialAtlas_level1_MNI152_1mm.nii.gz"),
        labels=_table_labels("arterial_labels.csv"),
        ggseg=False,
        license="CC BY-SA 4.0",
        citation="Liu C-F, et al. Digital 3D Brain MRI Arterial Territories Atlas. "
                 "Sci Data 10:74, 2023.",
    ),
    "tissue": dict(
        name="Tissue classes (DKT31/CMA + ICBM 2009a white matter)",
        img=lambda: _atlas_path("tissue_MNI152_1mm.nii.gz"),
        labels=_table_labels("tissue_labels.csv"),
        ggseg=False,
        license="Derived from CC BY 4.0 (DKT) and ICBM 2009a (free with notice) data",
        citation="Klein A, Tourville J. Front Neurosci 6:171, 2012; Fonov V, et al. "
                 "NeuroImage 54(1):313-327, 2011.",
    ),
}

#: aal is the default/first atlas. ``jhu`` (white matter) and ``arterial``
#: (vascular territories) complete the picture the grey-matter atlases leave
#: out; both share dk's grid, so they add little cost.
DEFAULT_ATLASES = ["aal", "schaefer", "dk", "jhu", "arterial"]

#: Host- and plugin-registered atlases. Same shape of entry as _REGISTRY,
#: except "img" is already a validated nibabel image (not a callable
#: path-builder) and "labels" is already a plain dict[int, RegionMeta] --
#: or, for a plugin atlas not yet used, both are None and "loader" builds
#: them on first use.
_CUSTOM: "dict[str, dict]" = {}

#: Built Atlas objects, keyed by atlas key. Guarded by _cache_lock so
#: concurrent first callers build exactly once.
_CACHE: "dict[str, Atlas]" = {}
_cache_lock = threading.RLock()


def _discover_plugins() -> None:
    from . import registry
    registry.discover()
    _register_downloadable()


def _register_downloadable() -> None:
    """Atlases fetched on request (``lqtpy fetch suit``): listed always,
    loadable once downloaded."""
    if "suit" in _CUSTOM:
        return
    from . import datasets
    if not datasets.suit_available():
        return
    with _cache_lock:
        _CUSTOM.setdefault("suit", {
            "name": "SUIT cerebellar lobules (Diedrichsen 2009)", "img": None, "labels": None,
            "ggseg": False, "loader": datasets.load_suit, "tier": "extra",
            "source": "download", "license": "Academic use with citation",
            "citation": "Diedrichsen J, et al. A probabilistic MR atlas of the human cerebellum. "
                        "NeuroImage 46(1):39-46, 2009."})


def list_atlases() -> list[str]:
    """Every atlas key: bundled, host-registered, and plugin-provided."""
    _discover_plugins()
    return list(_REGISTRY) + list(_CUSTOM)


def atlas_catalog() -> "dict[str, dict]":
    """``{key: {name, tier, source, license, citation}}`` for every atlas.
    Doesn't load any volume."""
    out = {}
    for key, spec in _REGISTRY.items():
        out[key] = {"name": spec["name"], "tier": "core", "source": "builtin",
                    "license": spec.get("license", ""), "citation": spec.get("citation", "")}
    for key, spec in _CUSTOM.items():
        out[key] = {"name": spec["name"], "tier": spec.get("tier", "extra"),
                    "source": spec.get("source", "host"),
                    "license": spec.get("license", ""), "citation": spec.get("citation", "")}
    return out


def _resolve_img(img_spec):
    """Turn a registry "img" spec into a loaded nibabel image."""
    img = img_spec() if callable(img_spec) else img_spec
    if isinstance(img, (str, os.PathLike)):
        path = os.fspath(img)
        if not os.path.exists(path):
            raise DataUnavailableError(f"Atlas volume missing: {path}")
        return nib.load(path)
    return img


def _build_atlas(key: str) -> Atlas:
    if key in _CUSTOM:
        spec = _CUSTOM[key]
        if spec.get("img") is None and spec.get("loader") is not None:
            try:
                img_src, raw_labels = spec["loader"]()
            except DataUnavailableError:
                raise
            except FileNotFoundError as exc:
                raise DataUnavailableError(f"Atlas '{key}' data missing: {exc}") from exc
            img_obj, norm = _validated_atlas(img_src, raw_labels)
            spec.update(img=img_obj, labels=norm, loader=None)
        img = _resolve_img(spec["img"])
        labels = spec["labels"]
        ggseg = spec.get("ggseg", False)
        gmap = spec.get("ggseg_map", {})
        tier = spec.get("tier", "extra")
    elif key in _REGISTRY:
        spec = _REGISTRY[key]
        img = _resolve_img(spec["img"])
        if spec["ggseg"]:
            labels, gmap = spec["labels"]()
        else:
            labels = spec["labels"]()
            gmap = {}
        ggseg = spec["ggseg"]
        tier = "core"
    else:
        raise UnknownAtlasError(f"Unknown atlas '{key}'. Available: {list_atlases()}")
    return Atlas(key=key, name=spec["name"], img=img, labels=labels,
                 ggseg=ggseg, ggseg_map=gmap, tier=tier)


def load_atlas(key: str) -> Atlas:
    if not isinstance(key, str):
        raise UnknownAtlasError(f"Atlas key must be a string, got {type(key).__name__}")
    key = key.lower()
    atlas = _CACHE.get(key)
    if atlas is not None:
        return atlas
    if key not in _REGISTRY and key not in _CUSTOM:
        _discover_plugins()
        if key == "suit" and key not in _CUSTOM:
            raise DataUnavailableError(
                "The cerebellar lobule atlas (suit) isn't downloaded yet. Run `lqtpy fetch suit` "
                "(or lqtpy.datasets.fetch_suit()) once.")
    with _cache_lock:
        atlas = _CACHE.get(key)
        if atlas is not None:
            return atlas
        atlas = _build_atlas(key)
        _CACHE[key] = atlas
        return atlas


# --------------------------------------------------------------------------- #
# Host-application / plugin atlas registration
# --------------------------------------------------------------------------- #
def _coerce_region_meta(value) -> RegionMeta:
    """Accept a region name, a RegionMeta, a (RegionName, Group) pair, or a
    mapping/object with RegionName/Group (plus any optional schema-v2 field:
    Lobe, DisplayName, Hemisphere, Tissue, Network, Function, Territory) --
    and normalise to a RegionMeta."""
    if isinstance(value, RegionMeta):
        meta = value
    elif isinstance(value, str):
        meta = RegionMeta(RegionName=value, Group="")
    elif isinstance(value, dict):
        meta = RegionMeta(RegionName=value["RegionName"], Group=value.get("Group", ""),
                          **{f: value.get(f) for f in OPTIONAL_FIELDS})
    elif isinstance(value, (tuple, list)) and len(value) == 2:
        meta = RegionMeta(RegionName=value[0], Group=value[1])
    elif hasattr(value, "RegionName"):
        meta = RegionMeta(RegionName=value.RegionName, Group=getattr(value, "Group", ""),
                          **{f: getattr(value, f, None) for f in OPTIONAL_FIELDS})
    else:
        raise TypeError(
            f"Unsupported label value for register_atlas: {value!r}. Expected a "
            "region-name string, a RegionMeta, a (RegionName, Group) pair, or a "
            "mapping/object with RegionName/Group attributes.")
    if meta.Hemisphere is None:
        hemi = _hemisphere_from_name(meta.RegionName)
        if hemi is not None:
            meta = meta._replace(Hemisphere=hemi)
    return meta


def _validated_atlas(img, labels):
    """Validate a host/plugin label volume + table; returns (image, labels).

    Label ids must be non-negative integers. A float volume is accepted when
    every value is within 1e-3 of an integer (float storage/resampling
    noise) and is rounded -- not truncated, which would turn 1.9999 into 1.
    """
    img_obj = as_lesion_image(img)
    data = np.asanyarray(img_obj.dataobj)
    if data.dtype.kind == "f":
        if not np.all(np.isfinite(data)):
            raise InvalidAtlasError("Atlas label volume contains NaN/inf values.")
        rounded = np.rint(data)
        if data.size and np.max(np.abs(data - rounded)) > 1e-3:
            raise InvalidAtlasError(
                "Atlas label volume has non-integer values; label ids must be integers.")
        data = rounded.astype(np.int64)
        dtype = _smallest_signed_dtype(int(data.max()) if data.size else 0)
        img_obj = nib.Nifti1Image(data.astype(dtype), img_obj.affine)
    if data.size and data.min() < 0:
        raise InvalidAtlasError(
            f"Atlas label volume has negative label ids (min {data.min()}); "
            "label ids must be >= 0 (0 = background).")
    norm = {}
    for lid, meta in labels.items():
        ilid = int(lid)
        if ilid < 0:
            raise InvalidAtlasError(f"Label id {lid!r} is negative.")
        norm[ilid] = _coerce_region_meta(meta)
    return img_obj, norm


def register_atlas(key: str, name: str, img, labels, *, overwrite: bool = False,
                   tier: str = "extra", source: str = "host", license: str = "",
                   citation: str = "") -> None:
    """Register a host-supplied atlas.

    Once registered it works through :func:`load_atlas`, :func:`list_atlases`,
    and every function in :mod:`lqtpy.api`, exactly like a bundled atlas.

    ``img`` is anything :func:`lqtpy.io.as_lesion_image` accepts (a path, a
    nibabel image, or an ``(array, affine)`` tuple) -- an integer label
    volume, despite the name; the validation ``as_lesion_image`` does (3D,
    finite/non-singular affine, numeric dtype) applies equally to a label
    volume.

    ``labels`` maps an integer label id to a region name, or to a
    RegionMeta-compatible value (a :class:`RegionMeta`, a
    ``(RegionName, Group)`` pair, or a mapping/object with
    ``RegionName``/``Group`` and optionally any schema-v2 field).

    Registering an existing key without ``overwrite=True`` raises
    ``ValueError``. An unusable label volume (negative ids, non-integer
    values) raises :class:`~lqtpy.errors.InvalidAtlasError`.
    """
    key = key.lower()
    img_obj, norm_labels = _validated_atlas(img, labels)
    with _cache_lock:
        if (key in _REGISTRY or key in _CUSTOM) and not overwrite:
            raise ValueError(
                f"Atlas '{key}' is already registered. Pass overwrite=True to replace it.")
        _CUSTOM[key] = {"name": name, "img": img_obj, "labels": norm_labels, "ggseg": False,
                        "tier": tier, "source": source, "license": license,
                        "citation": citation}
        _CACHE.pop(key, None)


def register_atlas_spec(spec, *, source: str = "plugin") -> None:
    """Register a plugin :class:`~lqtpy.registry.AtlasSpec` lazily: its
    ``load()`` runs (and is validated) the first time the atlas is used.
    A plugin can't replace a bundled atlas or a host registration."""
    key = spec.key.lower()
    with _cache_lock:
        if key in _REGISTRY or key in _CUSTOM:
            return
        _CUSTOM[key] = {"name": spec.name, "img": None, "labels": None, "ggseg": False,
                        "loader": spec.load, "tier": spec.tier, "source": source,
                        "license": spec.license, "citation": spec.citation}


def atlas_from_arrays(key: str, name: str, data, affine, labels, *,
                      overwrite: bool = False, **kw) -> None:
    """Convenience wrapper: :func:`register_atlas` from a raw ``(data,
    affine)`` pair instead of an image-like object."""
    register_atlas(key, name, (data, affine), labels, overwrite=overwrite, **kw)


def preload(keys: "list[str] | None" = None) -> None:
    """Build (and cache) every atlas's label volume + per-label voxel counts
    up front.

    Call this once from FastAPI startup so the first request that needs an
    atlas doesn't pay the (one-time) load cost. Defaults to every *core*
    atlas plus host-registered ones (plugin atlases load on first use, since
    some fetch data). Also preloads the tract proxy index, the QC brain mask,
    the tissue distance maps, the border-zone map, the key-structure
    distance maps,
    and, if one has been built locally, the per-tract disconnection index --
    a no-op, not an error, when it hasn't.
    """
    if keys is None:
        keys = list(_REGISTRY) + [k for k, v in _CUSTOM.items() if v.get("img") is not None]
    for k in keys:
        atlas = load_atlas(k)
        atlas.label_array()
        atlas.region_voxels()
    from . import tracts as _tracts
    if _tracts.tracts_available():
        _tracts._load_tracts()
    if "dk" in keys and "aal" in keys:
        from . import qc as _qc
        _qc.brain_mask()  # QC's out-of-brain / MNI checks (~0.6 s once)
    if "tissue" in keys:
        from . import tissue as _tissue
        _tissue.distance_maps()  # location classes
        if "arterial" in keys:
            from . import watershed as _ws
            _ws.zone_map()       # border zones
        if _tracts.tracts_available() and "dk" in keys:
            from . import structures as _st
            for name in _st.KEY_STRUCTURES:
                _st._distance_map(name)  # key-structure distances
    # (Each of these is built once per installation and then read from the
    # disk cache, see lqtpy._diskcache.)
    from . import datasets as _datasets
    _datasets.preload_discon_index()


def cache_clear() -> None:
    """Drop every cached :class:`Atlas` (built-in and host-registered), and
    the cached disconnection index if one was loaded.

    Atlas *registrations* made via :func:`register_atlas` are unaffected --
    only the built objects (and their lazily-computed label arrays) are
    dropped, and rebuilt fresh, once, on next use.
    """
    with _cache_lock:
        _CACHE.clear()
    from . import datasets as _datasets
    _datasets.discon_index_cache_clear()
