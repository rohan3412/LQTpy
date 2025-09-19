import nibabel as nib
from pathlib import Path

def load_nifti(lesion):
    if isinstance(lesion, (str, Path)):
        lesion_path = Path(lesion)
        if not lesion_path.exists():
            raise FileNotFoundError(f"File not found: {lesion_path}")
        return nib.load(str(lesion_path))

    elif isinstance(lesion, nib.Nifti1Image):
        return lesion

    else:
        raise TypeError("lesion must be a path to a NIfTI file (.nii or .nii.gz) or a nibabel Nifti1Image")


def load_path(output_path):
    if isinstance(output_path, Path):
        return output_path

    if isinstance(output_path, (str, Path)):
        return Path(output_path)

    raise TypeError(
        f"output_path must be a str or pathlib.Path, not {type(output_path)}"
    )

def easy_time(seconds):
    seconds = int(seconds)
    hour, remainder = divmod(seconds, 3600)
    minute, seconds = divmod(remainder, 60)

    parts = []
    if hour:
        if hour==1:
            parts.append(f"{hour} HR")
        else:
            parts.append(f"{hour} HRS")
    if minute:
        parts.append(f"{minute} MIN")
    if seconds:
        parts.append(f"{seconds} SEC")

    return " ".join(parts)

