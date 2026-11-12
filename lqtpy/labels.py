"""Multi-label masks: one result per label value (``--per-label``, opt-in).

By default a mask's non-zero labels are merged into one lesion, as before.
:func:`split_labels` gives one binary image per label value, and
:func:`analyze_labels` analyses each into ``<out>/<name>_label<value>/``
and writes ``<out>/<name>_labels.csv`` listing them.
"""
from __future__ import annotations

import csv
import os

import numpy as np

from .errors import InvalidLesionError

__all__ = ["label_values", "split_labels", "analyze_labels", "MAX_LABELS"]


def _data(img):
    d = np.asanyarray(img.dataobj)
    return d[..., 0] if d.ndim == 4 else d


def label_values(lesion) -> "list[int]":
    """The non-zero label values of an integer-valued mask, ascending."""
    from .io import as_lesion_image
    d = _data(as_lesion_image(lesion))
    if d.dtype.kind == "f":
        finite = d[np.isfinite(d)]
        if finite.size and not np.all(finite == np.round(finite)):
            raise InvalidLesionError("--per-label needs a mask of whole-number labels; "
                                     "this one has fractional values (a probability map?).")
    vals = np.unique(d[np.isfinite(d)] if d.dtype.kind == "f" else d)
    return [int(v) for v in vals if v != 0]


def split_labels(lesion) -> "dict[int, object]":
    """``{label value: binary NIfTI image}`` on the mask's own grid."""
    import nibabel as nib
    from .io import as_lesion_image
    img = as_lesion_image(lesion)
    d = _data(img)
    out = {}
    for v in label_values(img):
        m = nib.Nifti1Image((d == v).astype(np.uint8), img.affine)
        m.set_data_dtype(np.uint8)
        out[v] = m
    return out


_COLUMNS = ["Label", "Name", "Voxels", "VolumeMM3", "QC", "Report"]


#: More labels than this are refused unless ``max_labels`` says otherwise:
#: a parcellation or a probability map passed by mistake would otherwise
#: quietly make hundreds of reports.
MAX_LABELS = 20


def analyze_labels(lesion, out_dir=None, *, name: "str | None" = None,
                   max_labels: int = MAX_LABELS, **kw) -> dict:
    """:func:`lqtpy.analyze_single` for each label value of ``lesion``.
    ``kw`` goes to ``analyze_single`` (threshold is ignored: each label is
    already binary). A mask with more than ``max_labels`` label values is
    refused (raise it to allow more). Returns ``{"labels": {value: result},
    "index_path"}``."""
    from .analysis import analyze_single
    from .formats import strip_extension
    if name is None:
        name = strip_extension(os.path.basename(os.fspath(lesion))) \
            if isinstance(lesion, (str, os.PathLike)) else "lesion"
    n = len(label_values(lesion))
    if n > max_labels:
        raise InvalidLesionError(
            f"The mask has {n} label values, more than the limit of {max_labels} "
            f"(is it a parcellation?). Pass --max-labels {n} (max_labels={n}) to go ahead.")
    parts = split_labels(lesion)
    if not parts:
        raise InvalidLesionError("The mask has no non-zero labels.")
    out_dir = out_dir or os.path.join("lqtpy_output", name)
    kw.pop("threshold", None)
    results, rows = {}, []
    for v, img in parts.items():
        nm = f"{name}_label{v}"
        from .prepared import prepare
        p = prepare(img, 0.5, name=nm)
        res = analyze_single(p, out_dir=os.path.join(out_dir, nm), **kw)
        results[v] = res
        rows.append({"Label": v, "Name": nm, "Voxels": res["morphometry"]["n_voxels"],
                     "VolumeMM3": res["morphometry"]["volume_mm3"],
                     "QC": res.get("qc_verdict") or "",
                     "Report": os.path.relpath(res["report_path"], out_dir).replace(os.sep, "/")
                     if res.get("report_path") else ""})
    index = None
    if kw.get("write", True):
        os.makedirs(out_dir, exist_ok=True)
        index = os.path.join(out_dir, f"{name}_labels.csv")
        with open(index, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, _COLUMNS)
            w.writeheader()
            w.writerows(rows)
    return {"labels": results, "index_path": index}
