"""Lesion file formats beyond NIfTI.

=================  ==========================================================
``.nii``/``.nii.gz``  NIfTI (nibabel)
``.mgz``/``.mgh``     FreeSurfer MGH (nibabel)
``.nrrd``/``.nhdr``   NRRD, read by a small built-in reader (raw, gzip or
                      ASCII encoding; attached or detached data; LPS/RAS
                      spaces). No extra package needed.
``.dcm``              DICOM Segmentation (DICOM-SEG), via the optional
                      ``highdicom`` package (``pip install 'lqtpy[formats]'``).
                      All segments are combined into one lesion mask.
=================  ==========================================================

Every reader returns a ``nibabel.Nifti1Image`` in RAS world coordinates
(DICOM and most NRRD files use LPS: x and y are negated).
"""
from __future__ import annotations

import gzip
import os
import re

import numpy as np
import nibabel as nib

from .errors import InvalidLesionError, MissingDependencyError

__all__ = ["LESION_EXTENSIONS", "is_lesion_file", "strip_extension", "load_image",
           "read_nrrd", "write_nrrd", "read_dicom_seg"]

#: File name endings treated as lesion files (folder listings, file stems).
LESION_EXTENSIONS = (".nii.gz", ".nii", ".mgz", ".mgh", ".nrrd", ".nhdr", ".dcm")

GZIP_MAGIC = bytes([0x1F, 0x8B])
_LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])


def is_lesion_file(name: str) -> bool:
    return str(name).lower().endswith(LESION_EXTENSIONS)


def strip_extension(name: str) -> str:
    low = name.lower()
    for ext in LESION_EXTENSIONS:
        if low.endswith(ext):
            return name[: -len(ext)]
    return os.path.splitext(name)[0]


def load_image(path: str):
    """Load any supported lesion file as a nibabel image."""
    low = path.lower()
    if low.endswith((".nrrd", ".nhdr")):
        return read_nrrd(path)
    if low.endswith(".dcm"):
        return read_dicom_seg(path)
    try:
        return nib.load(path)
    except Exception:
        img = _load_by_content(path)
        if img is None:
            raise
        return img


def _load_by_content(path: str):
    """A NIfTI/MGH file whose name doesn't match its compression: gzip data
    named ``.nii`` (common after renaming), or plain data named ``.nii.gz``.
    Decided by the file's first bytes; ``None`` if that doesn't help."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if raw[:2] == GZIP_MAGIC:
        raw = gzip.decompress(raw)
    for cls in (nib.Nifti1Image, nib.Nifti2Image, nib.MGHImage):
        try:
            img = cls.from_bytes(raw)
            img.get_data_dtype()
            return img
        except Exception:
            continue
    return None


# --------------------------------------------------------------------------- #
# NRRD
# --------------------------------------------------------------------------- #
_NRRD_TYPES = {
    "int8": "i1", "signed char": "i1", "int8_t": "i1",
    "uint8": "u1", "uchar": "u1", "unsigned char": "u1", "uint8_t": "u1",
    "short": "i2", "short int": "i2", "signed short": "i2", "signed short int": "i2",
    "int16": "i2", "int16_t": "i2",
    "ushort": "u2", "unsigned short": "u2", "unsigned short int": "u2", "uint16": "u2",
    "uint16_t": "u2",
    "int": "i4", "signed int": "i4", "int32": "i4", "int32_t": "i4",
    "uint": "u4", "unsigned int": "u4", "uint32": "u4", "uint32_t": "u4",
    "longlong": "i8", "long long": "i8", "long long int": "i8", "signed long long": "i8",
    "signed long long int": "i8", "int64": "i8", "int64_t": "i8",
    "ulonglong": "u8", "unsigned long long": "u8", "unsigned long long int": "u8",
    "uint64": "u8", "uint64_t": "u8",
    "float": "f4", "double": "f8",
}


def _vec(text: str):
    return [float(v) for v in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", text)]


def read_nrrd(path: str) -> nib.Nifti1Image:
    """Read a 3-D NRRD file (``.nrrd`` with attached data, or ``.nhdr``
    with a detached data file) into a RAS nibabel image."""
    with open(path, "rb") as fh:
        raw = fh.read()
    if not raw.startswith(b"NRRD"):
        raise InvalidLesionError(f"'{path}' is not an NRRD file (no NRRD magic line).")
    end = raw.find(b"\n\n")
    if end < 0:
        end = raw.find(b"\r\n\r\n")
        sep = 4
    else:
        sep = 2
    header = raw[:end if end >= 0 else len(raw)].decode("latin-1")
    body = raw[end + sep:] if end >= 0 else b""
    fields = {}
    for line in header.splitlines()[1:]:
        if not line or line.startswith("#") or ":=" in line:
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            fields[k.strip().lower()] = v.strip()
    try:
        dtype = np.dtype(_NRRD_TYPES[fields["type"].lower()])
        sizes = [int(v) for v in fields["sizes"].split()]
    except KeyError as exc:
        raise InvalidLesionError(f"NRRD '{path}': missing or unknown field {exc}.") from exc
    if dtype.itemsize > 1:
        dtype = dtype.newbyteorder(">" if fields.get("endian", "little") == "big" else "<")
    datafile = fields.get("data file") or fields.get("datafile")
    if datafile:
        if datafile.split()[0].upper() == "LIST" or "%" in datafile:
            raise InvalidLesionError(f"NRRD '{path}': multi-file data is not supported.")
        dp = datafile if os.path.isabs(datafile) else os.path.join(os.path.dirname(path), datafile)
        with open(dp, "rb") as fh:
            body = fh.read()
    enc = fields.get("encoding", "raw").lower()
    n = int(np.prod(sizes))
    if enc in ("gzip", "gz"):
        body = gzip.decompress(body)
    elif enc == "bzip2" or enc == "bz2":
        import bz2
        body = bz2.decompress(body)
    if enc in ("ascii", "text", "txt"):
        data = np.array(body.decode("latin-1").split(), dtype=float).astype(dtype)[:n]
    elif enc in ("raw", "gzip", "gz", "bzip2", "bz2"):
        skip = int(fields.get("byte skip", 0) or 0)
        if skip == -1:
            body = body[len(body) - n * dtype.itemsize:]
        else:
            body = body[skip:]
        data = np.frombuffer(body, dtype=dtype, count=n)
    else:
        raise InvalidLesionError(f"NRRD '{path}': encoding '{enc}' is not supported.")
    data = data.reshape(sizes, order="F")

    # Geometry: spatial axes are those with a non-"none" space direction.
    dirs_txt = fields.get("space directions")
    space = fields.get("space", "").lower()
    if dirs_txt:
        tokens = re.findall(r"\([^)]*\)|none", dirs_txt)
        spatial = [i for i, t in enumerate(tokens) if t != "none"]
        vecs = [_vec(tokens[i]) for i in spatial]
        if len(spatial) != 3:
            raise InvalidLesionError(f"NRRD '{path}': expected 3 spatial axes, got {len(spatial)}.")
        other = [i for i in range(data.ndim) if i not in spatial]
        data = np.transpose(data, spatial + other)
        while data.ndim > 3:
            if data.shape[-1] != 1:
                raise InvalidLesionError(f"NRRD '{path}': more than one volume.")
            data = data[..., 0]
        aff = np.eye(4)
        aff[:3, :3] = np.array(vecs).T
        origin = _vec(fields.get("space origin", "(0,0,0)"))
        aff[:3, 3] = origin[:3]
        if space in ("left-posterior-superior", "lps", "scanner-xyz") or not space:
            # NRRD's default for medical data (Slicer, ITK) is LPS.
            aff = _LPS_TO_RAS @ aff if space != "scanner-xyz" else aff
        elif space in ("left-anterior-superior", "las"):
            aff = np.diag([-1.0, 1.0, 1.0, 1.0]) @ aff
    else:
        spacings = _vec(fields.get("spacings", "1 1 1")) or [1.0, 1.0, 1.0]
        aff = np.diag(list(spacings[:3]) + [1.0])
        data = data.reshape(data.shape[:3])
    if data.ndim != 3:
        raise InvalidLesionError(f"NRRD '{path}': expected a 3-D volume, got {data.ndim}-D.")
    return nib.Nifti1Image(np.ascontiguousarray(data), aff)


def write_nrrd(path: str, img_or_data, affine=None, *, encoding: str = "gzip") -> str:
    """Write a 3-D image as NRRD (LPS space, the common convention). Used by
    tests and for exporting masks to 3D Slicer."""
    if affine is None:
        data, affine = np.asarray(img_or_data.dataobj), img_or_data.affine
    else:
        data = np.asarray(img_or_data)
    aff = _LPS_TO_RAS @ np.asarray(affine, float)
    inv = {v: k for k, v in _NRRD_TYPES.items() if " " not in k and "_t" not in k}
    dt = data.dtype.newbyteorder("<") if data.dtype.itemsize > 1 else data.dtype
    tname = inv.get(dt.str.lstrip("<>|="), "float")
    if tname == "float" and dt.str[-2:] != "f4":
        data, tname = data.astype("<f4"), "float"
    dirs = " ".join("(" + ",".join(f"{v:.10g}" for v in aff[:3, i]) + ")" for i in range(3))
    head = ("NRRD0004\n"
            f"type: {tname}\ndimension: 3\nspace: left-posterior-superior\n"
            f"sizes: {' '.join(str(s) for s in data.shape)}\n"
            f"space directions: {dirs}\nkinds: domain domain domain\nendian: little\n"
            f"encoding: {encoding}\n"
            f"space origin: ({','.join(f'{v:.10g}' for v in aff[:3, 3])})\n\n")
    body = np.asarray(data, dtype=data.dtype.newbyteorder("<") if data.dtype.itemsize > 1
                      else data.dtype).tobytes(order="F")
    if encoding in ("gzip", "gz"):
        body = gzip.compress(body)
    with open(path, "wb") as fh:
        fh.write(head.encode("latin-1") + body)
    return path


# --------------------------------------------------------------------------- #
# DICOM-SEG
# --------------------------------------------------------------------------- #
def read_dicom_seg(path: str, segments=None) -> nib.Nifti1Image:
    """Read a DICOM Segmentation object; all segments (or the listed
    segment numbers) are combined into one binary mask in RAS space.

    Needs ``highdicom`` (``pip install 'lqtpy[formats]'``). The volume and
    its geometry come from ``highdicom``'s ``Segmentation.get_volume``; the
    frames must form a regular 3-D grid (true for segmentations made in
    3D Slicer, OHIF and most research tools)."""
    try:
        import highdicom as hd
    except ImportError as exc:
        raise MissingDependencyError(
            "formats", "Reading DICOM-SEG files needs the highdicom package: "
                       "pip install 'lqtpy[formats]'") from exc
    seg = hd.seg.segread(path)
    try:
        vol = seg.get_volume(segment_numbers=segments, combine_segments=True,
                             relabel=False, skip_overlap_checks=True)
    except TypeError:  # older/newer signatures
        vol = seg.get_volume(segment_numbers=segments, combine_segments=True)
    arr = np.asarray(vol.array)
    if arr.ndim == 4:
        arr = arr.any(axis=-1)
    aff = _LPS_TO_RAS @ np.asarray(vol.affine, float)
    return nib.Nifti1Image((arr > 0).astype(np.uint8), aff)
