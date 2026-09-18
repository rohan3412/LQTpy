"""Atlas registry.

Three bundled atlases, all resolved lazily to NIfTI + a label table. ``aal``
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

A host application can add its own atlases at runtime with
:func:`register_atlas`/:func:`atlas_from_arrays`; once registered they work
through :func:`load_atlas`, :func:`list_atlases`, and every function in
:mod:`lqtpy.api`.

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

from .errors import DataUnavailableError, UnknownAtlasError
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
    """Per-region metadata: mirrors the columns the code actually uses."""

    RegionName: str
    Group: str
    #: Anatomical lobe (Frontal/Parietal/Temporal/Occipital/Limbic/Insula/
    #: Subcortical/Cerebellum/...) -- independent of ``Group``, which is
    #: hemisphere for aal/dk and Yeo-7 network for schaefer. ``None`` when
    #: the atlas doesn't define a lobe for this region (currently schaefer).
    #: See ``data/aal_labels.csv``/``data/dkt31_cma_labels.csv`` for the
    #: mappings and their sourcing.
    Lobe: "str | None" = None


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
    #: dict mapping integer label id -> RegionMeta(RegionName, Group).
    labels: "dict[int, RegionMeta]"
    #: True when regions can be rendered as a ggseg Desikan-Killiany flat map.
    ggseg: bool = False
    #: Maps RegionName -> ggseg region key (only for ggseg atlases).
    ggseg_map: dict = field(default_factory=dict)
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
                    arr32 = np.asarray(self.img.get_fdata()).astype(np.int32)
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
    with opener(path, "rt", newline="") as fh:
        lines = (line for line in fh if not line.lstrip().startswith("#"))
        yield from csv.DictReader(lines)


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
        out[i] = RegionMeta(RegionName=name, Group=network(name))
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
        lobe = row.get("Lobe") or None
        out[lid] = RegionMeta(RegionName=name, Group=hemi(name), Lobe=lobe)
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
        lobe = row.get("Lobe") or None
        labels[lid] = RegionMeta(RegionName=name, Group=group, Lobe=lobe)
        if lid >= 1000:  # cortical DKT31 range -- see the docstring above.
            g = _fs_to_ggseg(name)
            if g is not None:
                gmap[name] = g
    return labels, gmap


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
_REGISTRY = {
    "aal": dict(
        name="AAL (116 anatomical regions)",
        img=lambda: _atlas_path("aal.nii.gz"),
        labels=_aal_labels,
        ggseg=False,
    ),
    "schaefer": dict(
        name="Schaefer-100 / 7 Networks",
        img=lambda: _atlas_path(
            "Schaefer2018_100Parcels_7Networks_order_FSLMNI152_1mm_resliced.nii.gz"),
        labels=_schaefer_labels,
        ggseg=False,
    ),
    "dk": dict(
        name="Desikan-Killiany-Tourville (DKT31 + CMA)",
        img=lambda: _atlas_path("OASIS-TRT-20_jointfusion_DKT31_CMA_labels_in_MNI152_v2.nii.gz"),
        labels=_dk_labels,
        ggseg=True,
    ),
}

#: aal is the default/first atlas.
DEFAULT_ATLASES = ["aal", "schaefer", "dk"]

#: Host-registered atlases (via register_atlas/atlas_from_arrays). Same
#: shape of entry as _REGISTRY, except "img" is already a validated nibabel
#: image (not a callable path-builder) and "labels" is already a plain
#: dict[int, RegionMeta] (not a callable loader).
_CUSTOM: "dict[str, dict]" = {}

#: Built Atlas objects, keyed by atlas key. Guarded by _cache_lock so
#: concurrent first callers build exactly once.
_CACHE: "dict[str, Atlas]" = {}
_cache_lock = threading.Lock()


def list_atlases() -> list[str]:
    return list(_REGISTRY) + list(_CUSTOM)


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
        img = _resolve_img(spec["img"])
        labels = spec["labels"]
        ggseg = spec.get("ggseg", False)
        gmap = spec.get("ggseg_map", {})
    elif key in _REGISTRY:
        spec = _REGISTRY[key]
        img = _resolve_img(spec["img"])
        if spec["ggseg"]:
            labels, gmap = spec["labels"]()
        else:
            labels = spec["labels"]()
            gmap = {}
        ggseg = spec["ggseg"]
    else:
        raise UnknownAtlasError(f"Unknown atlas '{key}'. Available: {list_atlases()}")
    return Atlas(key=key, name=spec["name"], img=img, labels=labels,
                ggseg=ggseg, ggseg_map=gmap)


def load_atlas(key: str) -> Atlas:
    key = key.lower()
    atlas = _CACHE.get(key)
    if atlas is not None:
        return atlas
    with _cache_lock:
        atlas = _CACHE.get(key)
        if atlas is not None:
            return atlas
        atlas = _build_atlas(key)
        _CACHE[key] = atlas
        return atlas


# --------------------------------------------------------------------------- #
# Host-application atlas registration
# --------------------------------------------------------------------------- #
def _coerce_region_meta(value) -> RegionMeta:
    """Accept a region name, a RegionMeta, a (RegionName, Group) pair, or a
    mapping/object with RegionName/Group -- and normalise to a RegionMeta."""
    if isinstance(value, RegionMeta):
        return value
    if isinstance(value, str):
        return RegionMeta(RegionName=value, Group="")
    if isinstance(value, dict):
        return RegionMeta(RegionName=value["RegionName"], Group=value.get("Group", ""))
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return RegionMeta(RegionName=value[0], Group=value[1])
    if hasattr(value, "RegionName"):
        return RegionMeta(RegionName=value.RegionName, Group=getattr(value, "Group", ""))
    raise TypeError(
        f"Unsupported label value for register_atlas: {value!r}. Expected a "
        "region-name string, a RegionMeta, a (RegionName, Group) pair, or a "
        "mapping/object with RegionName/Group attributes.")


def register_atlas(key: str, name: str, img, labels, *, overwrite: bool = False) -> None:
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
    ``RegionName``/``Group``).

    Registering an existing key without ``overwrite=True`` raises
    ``ValueError``.
    """
    key = key.lower()
    img_obj = as_lesion_image(img)
    norm_labels = {int(lid): _coerce_region_meta(meta) for lid, meta in labels.items()}
    with _cache_lock:
        if (key in _REGISTRY or key in _CUSTOM) and not overwrite:
            raise ValueError(
                f"Atlas '{key}' is already registered. Pass overwrite=True to replace it.")
        _CUSTOM[key] = {"name": name, "img": img_obj, "labels": norm_labels, "ggseg": False}
        _CACHE.pop(key, None)


def atlas_from_arrays(key: str, name: str, data, affine, labels, *,
                      overwrite: bool = False) -> None:
    """Convenience wrapper: :func:`register_atlas` from a raw ``(data,
    affine)`` pair instead of an image-like object."""
    register_atlas(key, name, (data, affine), labels, overwrite=overwrite)


def preload(keys: "list[str] | None" = None) -> None:
    """Build (and cache) every atlas's label volume + per-label voxel counts
    up front.

    Call this once from FastAPI startup so the first request that needs an
    atlas doesn't pay the (one-time) load cost. Defaults to every registered
    atlas (built-in + host-registered). Also preloads the per-tract
    disconnection index (:mod:`lqtpy.datasets`) if one has been built
    locally -- a no-op, not an error, when it hasn't (that dataset is
    optional, unlike the bundled atlases).
    """
    for k in (keys if keys is not None else list_atlases()):
        atlas = load_atlas(k)
        atlas.label_array()
        atlas.region_voxels()
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
