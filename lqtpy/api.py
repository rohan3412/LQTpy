"""Numeric API: the small, stable surface a web handler calls.

This module is what a FastAPI (or any other) web handler calls in-process,
on a threadpool, to get numbers back -- never plots, never an HTML report.
It **never imports** ``matplotlib``, ``nilearn``, ``pandas``, or lqtpy's own
``plotting``/``report``/``figures`` modules on any of its normal
(implemented) code paths, so calling it never pulls those into
``sys.modules``. Every function returns plain records (``list[dict]`` /
``dict``) -- see ``tests/test_api.py`` for a subprocess test that checks
this directly.

Every function accepts anything :func:`lqtpy.io.as_lesion_image` accepts for
its ``lesion`` argument: a path (``str``/``os.PathLike``), a nibabel image,
or an ``(array, affine)`` tuple -- or a :class:`~lqtpy.prepared.PreparedLesion`
from :func:`lqtpy.prepare`. Preparing once and passing the result to several
functions skips re-loading, re-thresholding and re-resampling the lesion;
a prepared lesion keeps the threshold it was prepared with, and the
``threshold`` argument is ignored for it.

Threshold convention
---------------------
Every function takes ``threshold`` (default ``0.5``). A lesion voxel counts
as lesioned when its value is **strictly greater than** ``threshold`` -- the
same binarization :func:`lqtpy.io.load_lesion` (and therefore
``analyze_single``/``analyze_group``) already use, so results here are
numerically identical to what those orchestrators produce for the same
lesion. The binarized 0/1 mask is then resampled onto the atlas grid with
nearest-neighbour interpolation.

This is a *different* comparison from the plain ``> 0`` used inside
:func:`lqtpy.io.resample_mask_to` to turn an *already-binary* resampled
mask's interpolated float data back into a boolean array -- that step never
sees a genuine probability value, so its threshold has nothing to do with
how a lesion gets binarized in the first place.

Exceptions raised here are the typed hierarchy in :mod:`lqtpy.errors`:
:class:`~lqtpy.errors.InvalidLesionError`/:class:`~lqtpy.errors.GeometryError`
for bad input (map to HTTP 4xx), :class:`~lqtpy.errors.UnknownAtlasError`
for a bad atlas key (HTTP 4xx), and
:class:`~lqtpy.errors.DataUnavailableError`/
:class:`~lqtpy.errors.MissingDependencyError` for environment problems
(HTTP 5xx).
"""
from __future__ import annotations

import numpy as np

from . import _discon_index as _di
from . import datasets as _datasets
from .atlases import load_atlas
from .morphometry import compute_morphometry
from .overlap import coverage as _coverage
from .overlap import lobe_rollup as _lobe_rollup
from .overlap import network_rollup as _network_rollup
from .overlap import region_overlap_from_mask as _region_overlap_from_mask
from .prepared import prepare
from .tracts import tract_overlap as _tract_overlap

__all__ = [
    "lesion_stats", "atlas_overlap", "atlas_coverage", "network_rollup_for",
    "lobe_overlap", "tract_proxy_overlap", "tract_disconnection",
]


def _overlap(lesion, atlas_key, threshold):
    """(prepared lesion, atlas, region records, lesion voxels on atlas grid)."""
    atlas = load_atlas(atlas_key)
    p = prepare(lesion, threshold)
    gm = p.on_grid(atlas.img)
    return p, atlas, _region_overlap_from_mask(gm, atlas), gm.n_voxels


def lesion_stats(lesion, *, threshold: float = 0.5) -> dict:
    """Morphometry for one lesion.

    Voxel count, volume in mm^3, and every other measure
    :func:`lqtpy.morphometry.compute_morphometry` computes (center of mass,
    bounding box, connected components, laterality...).
    """
    return compute_morphometry(prepare(lesion, threshold))


def atlas_overlap(lesion, atlas_key: str, *, threshold: float = 0.5) -> "list[dict]":
    """Per-region atlas overlap ("parcel damage") records for one lesion.

    ``atlas_key`` is anything :func:`lqtpy.atlases.list_atlases` lists
    (built-in or host-registered via
    :func:`lqtpy.atlases.register_atlas`/:func:`lqtpy.atlases.atlas_from_arrays`).
    Raises :class:`~lqtpy.errors.UnknownAtlasError` for an unregistered key.
    """
    return _overlap(lesion, atlas_key, threshold)[2]


def atlas_coverage(lesion, atlas_key: str, *, threshold: float = 0.5) -> dict:
    """How much of the lesion ``atlas_key`` can describe: labelled vs
    unlabelled lesion voxels on the atlas grid, and lesion voxels outside the
    atlas's field of view. See :func:`lqtpy.overlap.coverage`.

    Region rows' ``PercentOfLesion`` plus ``PercentUnlabelled`` add up to 100.
    """
    return _coverage(prepare(lesion, threshold), load_atlas(atlas_key))


def network_rollup_for(lesion, atlas_key: str, *, threshold: float = 0.5) -> "list[dict]":
    """Group-level rollup records for one lesion, one row per region group.

    "Network" means the atlas's region Group field, which differs per atlas:
    Yeo-7 functional networks for ``schaefer``, but hemispheres for ``aal``
    and ``dk``. Check the atlas before labelling this output as functional
    networks; :func:`lqtpy.report.fragments.network_rollup_fragment` titles
    itself from the data for the same reason.
    """
    _, atlas, overlap, total = _overlap(lesion, atlas_key, threshold)
    return _network_rollup(overlap, atlas, lesion_total=total)


def lobe_overlap(lesion, atlas_key: str, *, threshold: float = 0.5) -> "list[dict]":
    """Lobe-level rollup records for one lesion, one row per anatomical lobe.

    Unlike :func:`network_rollup_for` (which groups by the atlas's ``Group``
    field -- hemisphere for ``aal``/``dk``, Yeo-7 network for ``schaefer``),
    this groups by each region's dedicated ``Lobe`` field (frontal/
    parietal/temporal/occipital/limbic/insula/subcortical/cerebellum/...,
    see :class:`~lqtpy.atlases.RegionMeta`). ``aal`` and ``dk`` define a
    lobe for every region; ``schaefer`` currently does not.

    Raises :class:`~lqtpy.errors.NoLobeMappingError` if the atlas has no
    lobe mapping for any region.
    """
    atlas = load_atlas(atlas_key)
    from .overlap import atlas_has_lobes
    if not atlas_has_lobes(atlas):
        return _lobe_rollup([], atlas)  # raises NoLobeMappingError
    _, atlas, overlap, total = _overlap(lesion, atlas_key, threshold)
    return _lobe_rollup(overlap, atlas, lesion_total=total)


def tract_proxy_overlap(lesion, *, threshold: float = 0.5) -> "list[dict]":
    """Per-tract white-matter structural-overlap **proxy** -- NOT disconnection.

    This is the fraction of each white-matter tract's voxels the lesion
    touches. A small lesion can fully disconnect a tract while overlapping
    few of its voxels, so this is a coarse proxy for damage to a tract's
    *territory*, not streamline-level disconnection. For the real,
    streamline-level measure -- what fraction of a tract's actual
    reference streamlines the lesion severs -- see
    :func:`tract_disconnection`, which requires a locally-built index
    (:mod:`lqtpy.datasets`) that this function does not.
    """
    return _tract_overlap(prepare(lesion, threshold))


def tract_disconnection(lesion, *, threshold: float = 0.5,
                        index_path: "str | None" = None) -> "list[dict]":
    """Per-tract **real** streamline-level disconnection.

    For each of the ~70 named tracts in the HCP-842 population tractography
    atlas, a streamline counts as disconnected if **any** of its points
    falls in a lesion voxel (whole-streamline, binary -- no partial credit,
    no minimum-overlap-length; equivalent to
    ``dipy.tracking.utils.target(..., include=True)`` on the reference
    streamlines). Percent disconnection per tract is (disconnected
    streamlines in that tract) / (that tract's fixed reference streamline
    count, from the atlas, not recomputed per lesion) x 100. This
    reimplements the per-tract disconnection measure described in Griffis
    JC, Metcalf NV, Corbetta M, Shulman GL, "Lesion Quantification Toolkit:
    A MATLAB software tool for estimating grey matter damage and white
    matter disconnections in patients with focal brain lesions,"
    *NeuroImage: Clinical* 30:102639 (2021).

    This is a genuinely different measure from :func:`tract_proxy_overlap`:
    that function reports what fraction of a tract's *voxels* the lesion's
    footprint touches (a spatial proxy, computable with only the bundled
    atlas), while this function reports what fraction of a tract's actual
    reference *streamlines* the lesion severs (the real measure, but it
    requires the locally-built streamline index below -- there is no
    bundled fallback). A small, well-placed lesion can fully disconnect a
    tract while overlapping only a few of its voxels; ``tract_proxy_overlap``
    under-reports that case, which is exactly why this function exists.

    Requires a locally-built disconnection index (see
    ``tools/build_discon_index.py`` and :mod:`lqtpy.datasets`) --
    :class:`~lqtpy.errors.DataUnavailableError` is raised with build
    instructions if one hasn't been built (or ``index_path`` doesn't
    exist). ``threshold`` follows the same convention as every other
    function here (see the module docstring): the lesion is loaded and
    binarized via :func:`lqtpy.io.load_lesion` exactly like
    :func:`tract_proxy_overlap` does, then resampled (nearest-neighbour)
    onto the tract atlas's own grid.

    Parameters
    ----------
    lesion : anything :func:`lqtpy.io.as_lesion_image` accepts.
    threshold : float, default 0.5.
    index_path : str, optional
        Explicit path to a built index ``.npz``, bypassing the cached
        default index resolved from ``LQTPY_DATA_HOME``
        (:func:`lqtpy.datasets.get_discon_index_path`). Mainly for tests
        and for a host app managing multiple index builds; loaded fresh
        each call (not cached) when given explicitly.

    Returns
    -------
    list[dict]
        One record per tract, sorted by tract short code (deterministic
        order), with keys ``tract`` (short code, the stable identifier),
        ``name`` (corrected human-readable display name -- see
        :func:`lqtpy.datasets.get_tract_display_names`), ``n_streamlines``
        (the tract's fixed reference count), ``n_disconnected``, and
        ``percent_disconnected``. Tracts with 0% are included, not dropped.

    Raises
    ------
    lqtpy.errors.InvalidLesionError
        Empty lesion after thresholding (matches :func:`lesion_stats` /
        every other function here).
    lqtpy.errors.DataUnavailableError
        No disconnection index has been built locally.
    """
    # Validate the lesion itself first (InvalidLesionError for an empty
    # mask) before touching the index, so that failure mode is independent
    # of whether an index has been built -- matches lesion_stats/every
    # other function here, which never need index/atlas data to reject a
    # bad lesion.
    p = prepare(lesion, threshold)

    if index_path is not None:
        index = _di.load_index(index_path)
    else:
        index = _datasets._load_discon_index()

    lesion_voxel_keys = p.on_grid((tuple(index["shape"]), np.asarray(index["affine"]))).flat

    streamline_ids = _di.gather_streamline_ids(lesion_voxel_keys, index)
    tract_ids_hit = index["streamline_tract"][streamline_ids]
    n_tracts = len(index["tract_codes"])
    disc_counts = np.bincount(tract_ids_hit.astype(np.int64), minlength=n_tracts)

    names = _datasets.get_tract_display_names()
    records = []
    for tid in range(n_tracts):
        code = index["tract_codes"][tid]
        n_ref = int(index["tract_ref_counts"][tid])
        n_disc = int(disc_counts[tid])
        pct = round(100.0 * n_disc / n_ref, 2) if n_ref else 0.0
        records.append({
            "tract": code,
            "name": names.get(code, code),
            "n_streamlines": n_ref,
            "n_disconnected": n_disc,
            "percent_disconnected": pct,
        })
    records.sort(key=lambda r: r["tract"])
    return records
