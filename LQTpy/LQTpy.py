import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from importlib import resources
from pathlib import Path
from .structural import compute_structural_analysis
from .structural_plot import create_structural_plots
from .disconnectome import compute_disconnectome

def LQTpy(lesion_file='lesion.nii', tck_file='global_tractography_sift.tck', reference_path=None, 
          modules={'structural': True, 'tract': True, 'disconnectome': True, 'network': True}, 
          atlas="Harvard_Oxford_test", output_path="output"):
    print(f"Running Lesion Analysis for {lesion_file} ...")
    if reference_path is None:
        reference_path = resources.files('LQTpy').joinpath('resources', 'mni152.nii.gz')
    mni_img = nib.load(reference_path)
    lesion_img = resample_to_img(nib.load(lesion_file), mni_img, interpolation='nearest', force_resample=True, copy_header=True)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if modules['structural']:
        structural_output = output_path / "structural"
        structural_output.mkdir(exist_ok=True)
        atlas_dir = resources.files('LQTpy').joinpath(Path('resources') / 'Atlas' / atlas)
        roi_paths = [path for path in atlas_dir.iterdir() if path.name.endswith(('.nii', '.nii.gz'))]
        df = compute_structural_analysis(lesion_img, mni_img, atlas_dir, roi_paths, structural_output, lesion_file)
        create_structural_plots(df, lesion_img, mni_img, roi_paths, structural_output)

    if modules['disconnectome']:
        disconnectome_output = output_path / "disconnectome"
        disconnectome_output.mkdir(exist_ok=True)
        compute_disconnectome(lesion_img, disconnectome_output, tck_file, mni_img)


