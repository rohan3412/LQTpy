import nibabel as nib
import os
from pathlib import Path

def load_nifti(lesion):
    if isinstance(lesion, str):
        if not os.path.exists(lesion):
            raise FileNotFoundError(f"File not found: {lesion}")
        img = nib.load(lesion)
        return img

    elif isinstance(lesion, nib.Nifti1Image):
        return lesion

    else:
        raise TypeError("lesion must be a path to a NIfTI file (.nii or .nii.gz) or a nibabel Nifti1Image")


def load_path(output_path):
    if isinstance(output_path, Path):
        return output_path

    if isinstance(output_path, (str, os.PathLike)):
        return Path(output_path)

    raise TypeError(
        f"output_path must be a str, os.PathLike, or pathlib.Path, not {type(output_path)}"
    )
