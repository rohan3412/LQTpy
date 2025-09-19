import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import resample_to_img
from pathlib import Path
from tqdm import tqdm
import time

def compute_overlap(lesion, roi):
    lesion_data = lesion.get_fdata() > 0
    roi_data = roi.get_fdata() > 0
    overlap = np.logical_and(lesion_data, roi_data)
    overlap_voxels = overlap.sum()
    roi_percent = overlap_voxels / roi_data.sum()
    lesion_percent = overlap_voxels / lesion_data.sum()
    return roi_percent, lesion_percent, overlap_voxels

def compute_structural_analysis(lesion_img, mni_img, atlas_dir, roi_paths, structural_output, lesion_file):
    structural_results = []
    print("Starting structural analysis")
    time.sleep(0.5)
    for roi_path in tqdm(roi_paths, desc="Processing ROIs"):
        roi_img = resample_to_img(nib.load(roi_path), mni_img, interpolation='nearest', force_resample=True,
                                  copy_header=True)
        roi_percent, lesion_percent, overlap_voxels = compute_overlap(lesion_img, roi_img)
        structural_results.append({
            "roi": Path(roi_path.stem).stem,
            "overlap_voxels": overlap_voxels,
            "percent_roi_overlap": roi_percent,
            "percent_lesion_overlap": lesion_percent
        })
    df = pd.DataFrame(structural_results, columns=["roi", "overlap_voxels", "percent_roi_overlap", "percent_lesion_overlap"])
    csv_path = structural_output / "structural_results.csv"
    df.to_csv(csv_path, index=False)
    return df
