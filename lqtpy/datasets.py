"""Data-home management for locally-built lqtpy datasets.

Right now this covers exactly one thing: the real per-tract streamline
disconnection index (see ``tools/build_discon_index.py`` and
``lqtpy.api.tract_disconnection``). It is built from the HCP-842
tractography atlas. At roughly 18 MB it is too large to bundle in the
wheel, so it is built locally and loaded from a per-user data home rather
than from package data. (Licensing does not prevent distributing one: the
source atlas is CC BY-SA 4.0, so a hosted index would simply have to carry
that licence and its attribution -- see ``lqtpy/data/atlases/SOURCE.md``.)

Thread safety mirrors ``lqtpy/atlases.py``: the index is loaded once,
lazily, behind a lock, cached as read-only arrays, and rebuildable via
:func:`discon_index_cache_clear`. :func:`preload_discon_index` is meant to
be called from ``lqtpy.preload()`` (see the hook in ``atlases.py``) so a
host app's startup can pay the load cost once, up front.
"""
from __future__ import annotations

import csv
import hashlib
import os
import tempfile
import threading

import numpy as np

from ._discon_index import FORMAT_VERSION, load_index
from .atlases import DATA_DIR
from .errors import DataUnavailableError

_INDEX_FILENAME = "discon_index_v1.npz"

_INDEX_CACHE = None
#: Guards the double-checked lazy load of _INDEX_CACHE, same pattern as
#: atlases.py's _cache_lock / Atlas._build_lock.
_index_lock = threading.Lock()

#: short_code -> corrected display name, package data (tiny, always
#: shipped -- unlike the built index above, this is committed to the repo
#: like aal_labels.csv). Lazily read once; 70 rows, not worth a lock.
_TRACT_NAMES_CSV = os.path.join(DATA_DIR, "tract_disconnection_names.csv")
_TRACT_NAMES_CACHE: "dict[str, str] | None" = None


def get_tract_display_names() -> "dict[str, str]":
    """short_code -> corrected human-readable display name.

    Source: ``lqtpy/data/tract_disconnection_names.csv``, built from the
    HCP-842 ``tract_info.csv`` but with four labelling errors found in that
    source file corrected (CS/CST both said "Corticospinal Tract"; PPT said
    "Pontine Tegmentum Tract" instead of Parietopontine; TPT said
    "Temporoparietal Tract" instead of Temporopontine; EMC said "External
    Medullary Lamina" instead of Extreme Capsule) -- see that CSV's own
    ``corrected``/``note`` columns for per-entry provenance. Tract short
    codes (``tract`` in :func:`lqtpy.api.tract_disconnection`'s output) are
    the stable identifier; these names are for humans only.
    """
    global _TRACT_NAMES_CACHE
    if _TRACT_NAMES_CACHE is None:
        names = {}
        with open(_TRACT_NAMES_CSV, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                names[row["short_code"]] = row["display_name"]
        _TRACT_NAMES_CACHE = names
    return _TRACT_NAMES_CACHE


def get_data_home() -> str:
    """Root directory for locally-built lqtpy data.

    Resolution order: the ``LQTPY_DATA_HOME`` environment variable if set,
    else a sensible per-user cache directory (``%LOCALAPPDATA%\\lqtpy`` on
    Windows, ``$XDG_CACHE_HOME/lqtpy`` or ``~/.cache/lqtpy`` elsewhere).

    Never creates the directory -- this is a path computation only, so
    importing/calling this at any time (including import time) has no
    filesystem side effect. Callers create it (or a parent of it) only when
    they're about to actually write something (see ``save_index`` usage in
    ``tools/build_discon_index.py``).
    """
    env = os.environ.get("LQTPY_DATA_HOME")
    if env:
        return env
    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~")
        return os.path.join(base, "lqtpy")
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "lqtpy")


def discon_index_target_path() -> str:
    """Where the disconnection index is looked for (and where
    ``tools/build_discon_index.py`` should write it), whether or not it
    exists yet."""
    return os.path.join(get_data_home(), _INDEX_FILENAME)


def get_discon_index_path() -> str:
    """Path to the built disconnection index, if it exists.

    Raises :class:`~lqtpy.errors.DataUnavailableError` with an actionable
    message (how to build the index, how to point ``LQTPY_DATA_HOME`` at an
    existing one) if it hasn't been built yet. No public download URL is
    published for this index, so this never attempts to fetch one.
    """
    data_home = get_data_home()
    path = os.path.join(data_home, _INDEX_FILENAME)
    if not os.path.exists(path):
        raise DataUnavailableError(
            "Real per-tract disconnection index not found "
            f"(looked for '{path}').\n\n"
            "Build it locally from the HCP-842 tractography atlas (70 "
            "*.trk.gz bundles + tract_info.csv):\n\n"
            f"    python tools/build_discon_index.py --source <All_Tracts dir> "
            f"--out {path}\n\n"
            "Or, if you already have a built index elsewhere, point "
            "LQTPY_DATA_HOME at its parent directory (it must contain a "
            f"file named '{_INDEX_FILENAME}'); this process currently "
            f"resolves LQTPY_DATA_HOME to '{data_home}'.\n\n"
            "No public download is published for this index. In the "
            "meantime, lqtpy.tract_proxy_overlap() remains available and "
            "needs no locally-built data."
        )
    return path


def discon_index_available() -> bool:
    """True if a disconnection index is present at the resolved path."""
    try:
        get_discon_index_path()
        return True
    except DataUnavailableError:
        return False


def _freeze(index: dict) -> dict:
    """Mark every ndarray value read-only, matching atlases.py's convention
    that a cached array can't be mutated by one caller and corrupt another's
    view of it."""
    for value in index.values():
        if isinstance(value, np.ndarray):
            value.flags.writeable = False
    return index


def _load_discon_index() -> dict:
    global _INDEX_CACHE
    if _INDEX_CACHE is not None:
        return _INDEX_CACHE
    with _index_lock:
        if _INDEX_CACHE is not None:
            return _INDEX_CACHE
        path = get_discon_index_path()
        _INDEX_CACHE = _freeze(load_index(path))
        return _INDEX_CACHE


def preload_discon_index() -> None:
    """Build (and cache) the disconnection index up front, if one exists.

    Call this from process startup (``lqtpy.preload()`` does, if the index
    is available) so the first disconnection query doesn't pay the load
    cost. A no-op (not an error) when no index has been built yet -- unlike
    the atlas caches, this dataset is genuinely optional.
    """
    if discon_index_available():
        _load_discon_index()


def discon_index_cache_clear() -> None:
    """Drop the cached index, forcing a fresh load (and re-check of
    ``get_discon_index_path()``) on next use. Called by
    ``lqtpy.cache_clear()``."""
    global _INDEX_CACHE
    with _index_lock:
        _INDEX_CACHE = None


# --------------------------------------------------------------------------- #
# Verified download helper (used by fetch_suit; the disconnection index
# has no public URL yet, see get_discon_index_path()'s docstring).
# --------------------------------------------------------------------------- #
def fetch(url: str, sha256: "str | None", dest: str) -> str:
    """Download ``url`` to ``dest``, verifying its sha256 digest, atomically.

    Streams to a temp file in ``dest``'s directory and ``os.replace``s it
    into place only after the digest matches, so a partial/corrupt download
    (or a mismatched digest) never lands at ``dest``. Raises ``ValueError``
    on a digest mismatch and cleans up the temp file either way.

    ``sha256=None`` skips the check (for sources that publish no digest;
    the digest of what was downloaded is recorded next to it instead).
    Used by :func:`fetch_suit`.
    """
    from urllib.request import urlopen

    dest_dir = os.path.dirname(os.path.abspath(dest)) or "."
    os.makedirs(dest_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=dest_dir, prefix=".lqtpy-fetch-")
    try:
        digest = hashlib.sha256()
        with os.fdopen(fd, "wb") as fh, urlopen(url) as resp:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                digest.update(chunk)
        got = digest.hexdigest()
        if sha256 and got.lower() != sha256.lower():
            raise ValueError(
                f"sha256 mismatch downloading {url}: expected {sha256}, got {got}")
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
    return dest


SUIT_BASE_URL = ("https://raw.githubusercontent.com/DiedrichsenLab/cerebellar_atlases/master/"
                 "Diedrichsen_2009/")
SUIT_FILES = ("atl-Anatom_space-MNI_dseg.nii", "atl-Anatom.tsv")


def suit_dir() -> str:
    return os.path.join(get_data_home(), "suit")


def suit_available() -> bool:
    return all(os.path.exists(os.path.join(suit_dir(), f)) for f in SUIT_FILES)


def fetch_suit(base_url: str = SUIT_BASE_URL, *, force: bool = False) -> str:
    """Download the cerebellar lobule atlas (Diedrichsen et al. 2009, MNI
    space, 28 lobules + 6 deep nuclei; from the DiedrichsenLab
    ``cerebellar_atlases`` repository) into ``<data home>/suit``. After
    this, ``atlases=[..., "suit"]`` works. Returns the folder.

    The atlas is academic data: cite Diedrichsen J, et al. NeuroImage
    46(1):39-46, 2009, and check the repository's terms. The files' SHA-256
    digests are written to ``SHA256SUMS`` beside them."""
    d = suit_dir()
    os.makedirs(d, exist_ok=True)
    sums = []
    for f in SUIT_FILES:
        dest = os.path.join(d, f)
        if force or not os.path.exists(dest):
            fetch(base_url.rstrip("/") + "/" + f, None, dest)
        h = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        sums.append(f"{h}  {f}")
    with open(os.path.join(d, "SHA256SUMS"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(sums) + "\n")
    from .atlases import _register_downloadable, cache_clear
    cache_clear()
    _register_downloadable()
    return d


def load_suit():
    """``(image, labels)`` for the fetched SUIT atlas (the atlas loader)."""
    import csv
    import nibabel as nib
    from .errors import DataUnavailableError
    if not suit_available():
        raise DataUnavailableError(
            "The cerebellar lobule atlas (suit) isn't downloaded yet. Run "
            "`lqtpy fetch suit` (or lqtpy.datasets.fetch_suit()) once.")
    d = suit_dir()
    labels = {}
    with open(os.path.join(d, "atl-Anatom.tsv"), encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            name = r["name"]
            side = "L" if name.startswith("Left_") else "R" if name.startswith("Right_") else "Midline"
            part = name.split("_", 1)[1] if "_" in name else name
            nucleus = part in ("Dentate", "Interposed", "Fastigial")
            disp = (f"{part.lower()} nucleus" if nucleus else
                    f"lobule {part}".replace("lobule Crus", "crus ") if not name.startswith("Vermis")
                    else f"vermis {part}".replace("vermis Crus", "vermis crus "))
            labels[int(r["index"])] = {
                "RegionName": name, "Group": "Cerebellum", "Lobe": "Cerebellum",
                "DisplayName": disp, "Hemisphere": side,
                "Tissue": "Deep grey matter" if nucleus else "Cerebellum"}
    return nib.load(os.path.join(d, "atl-Anatom_space-MNI_dseg.nii")), labels


__all__ = [
    "fetch_suit", "load_suit", "suit_available", "SUIT_BASE_URL", "discon_index_target_path",
    "get_data_home", "get_discon_index_path", "discon_index_available",
    "preload_discon_index", "discon_index_cache_clear", "fetch",
    "get_tract_display_names", "FORMAT_VERSION",
]
