import os
import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.image import resample_to_img
from nilearn.datasets import load_mni152_template
from nilearn import plotting

from importlib import resources
import tempfile

from LQTpy.util import load_nifti, load_path
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.cm as cm
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def compute_overlap(lesion, roi):
    lesion_data = lesion.get_fdata() > 0
    roi_data = roi.get_fdata() > 0

    overlap = np.logical_and(lesion_data, roi_data)
    overlap_voxels = overlap.sum()

    roi_percent = overlap_voxels / roi_data.sum()
    lesion_percent = overlap_voxels / lesion_data.sum()

    return roi_percent, lesion_percent, overlap_voxels

def LQTpy(lesion,modules={'structural':True,
                          'tract':True,
                          'disconnectome':True,
                          'network':True},atlas="Harvard_Oxford_test",output_path="output"):

    mni_img = load_mni152_template()

    lesion_img = resample_to_img(load_nifti(lesion),mni_img, interpolation='nearest', force_resample=True, copy_header=True)
    output_path = load_path(output_path)
    output_path.mkdir(parents=True,exist_ok = True)


    if modules['structural']:
        structural_output = output_path / "structural"
        structural_output.mkdir(exist_ok = True)
        structural_results = []

        atlas_dir = resources.files("LQTpy.resources.Atlas." + atlas)

        roi_paths = [path for path in atlas_dir.iterdir() if path.name.endswith(('.nii', '.nii.gz'))]

        for roi_path in roi_paths:
            print(Path(roi_path.stem).stem)
            roi_img = resample_to_img(nib.load(roi_path), mni_img, interpolation='nearest', force_resample=True,
                                      copy_header=True)

            roi_percent, lesion_percent, overlap_voxels = compute_overlap(lesion_img, roi_img)

            structural_results.append({
                "roi": Path(roi_path.stem).stem,
                "overlap_voxels": overlap_voxels,
                "percent_roi_overlap": roi_percent,
                "percent_lesion_overlap": lesion_percent
            })

        df = pd.DataFrame(structural_results, columns=["roi","overlap_voxels","percent_roi_overlap","percent_lesion_overlap"])

        csv_path = structural_output / "structural_results.csv"
        df.to_csv(csv_path, index = False)

        top_n = 10

        df_sorted = df.sort_values(by="overlap_voxels", ascending=False)

        df_sorted = df_sorted[df_sorted['overlap_voxels']>0]

        white_red_cmap = LinearSegmentedColormap.from_list("white_red", ["white", "red"])
        white_blue_cmap = LinearSegmentedColormap.from_list("white_blue", ["white", "blue"])

        # Bar Plot: Overlap Voxels
        plt.figure(figsize=(12, 6))
        plt.bar(df_sorted["roi"], df_sorted["overlap_voxels"])
        plt.xticks(rotation=90)
        plt.ylabel("Overlap Voxels")
        plt.title("Lesion-ROI Overlap (Voxel Count)")
        plt.tight_layout()
        plt.savefig(structural_output / "bar_overlap_voxels.png")
        plt.close()

        # Plot the lesion itself
        display = plotting.plot_glass_brain(
            lesion_img, display_mode='lyrz',
            colorbar=True, plot_abs=False, cmap=white_red_cmap,
            title="Lesion Glass Brain View"
        )
        display.savefig(structural_output / "lesion_glass_brain.png")
        display.close()

        roi_folder = structural_output / "roi_glass_brain"
        roi_folder.mkdir(exist_ok = True)

        # Plot top N overlapping ROIs
        top_rois = df_sorted[df_sorted["overlap_voxels"] > 0].head(top_n)
        for _, row in top_rois.iterrows():
            # Find ROI file by exact name match
            roi_path = next(f for f in roi_paths if Path(f.stem).stem == row['roi'])
            roi_img = resample_to_img(
                nib.load(roi_path), mni_img,
                interpolation='nearest', force_resample=True, copy_header=True
            )
            display = plotting.plot_glass_brain(
                roi_img, display_mode='lyrz',
                colorbar=True, plot_abs=False, cmap=white_blue_cmap,
                title=f"Top ROI: {row['roi']}"
            )
            display.savefig(roi_folder / f"{row['roi']}_glass_brain.png")
            display.close()

        df_sorted_roi = df.sort_values(by="percent_roi_overlap", ascending=False)
        df_sorted_roi = df_sorted_roi[df_sorted_roi['percent_roi_overlap'] > 0]

        top_rois = df_sorted_roi.head(min(top_n, len(df_sorted_roi)))

        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 2], hspace=0.3)

        ax_glass = fig.add_subplot(gs[0])

        combined_data = np.zeros_like(mni_img.get_fdata(), dtype=float)
        max_overlap = top_rois["percent_roi_overlap"].max()
        for idx, row in top_rois.iterrows():
            roi_path = next(f for f in roi_paths if Path(f.stem).stem == row['roi'])
            roi_img = resample_to_img(nib.load(roi_path), mni_img, interpolation='nearest', force_resample=True,
                                      copy_header=True)
            roi_data = roi_img.get_fdata()
            weight = row["percent_roi_overlap"]
            combined_data += np.where(roi_data > 0, weight, 0)

        reds = cm.Reds(np.linspace(0, 1, 256))
        white_to_red = np.vstack((np.array([1, 1, 1, 1])[None, :], reds))
        custom_cmap = LinearSegmentedColormap.from_list('white_to_red', white_to_red)

        combined_img = nib.Nifti1Image(combined_data, mni_img.affine, mni_img.header)
        display = plotting.plot_glass_brain(
            combined_img,
            display_mode='lyrz',
            colorbar=False,
            plot_abs=False,
            cmap=custom_cmap,
            vmin=0,
            vmax=max_overlap,
            axes=ax_glass
        )

        ax_bar = fig.add_subplot(gs[1])
        bars = ax_bar.barh(range(len(top_rois)), top_rois["percent_roi_overlap"].values)

        ax_bar.set_yticks(range(len(top_rois)))
        ax_bar.set_yticklabels(top_rois["roi"].values)
        ax_bar.set_xlabel("Percent of ROI Overlapped", fontsize=12)
        ax_bar.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax_bar.invert_yaxis()  # Largest at top
        ax_bar.set_title("% ROI Involved", fontsize=14, fontweight='bold')

        max_overlap = top_rois["percent_roi_overlap"].max()
        colors = plt.cm.Reds(top_rois["percent_roi_overlap"].values / max_overlap)
        for bar, color in zip(bars, colors):
            bar.set_color(color)

        for bar, value in zip(bars, top_rois["percent_roi_overlap"].values):
            width = bar.get_width()
            ax_bar.text(width + 0.01, bar.get_y() + bar.get_height() / 2,
                        f"{width:.1%}", va='center', fontsize=10, fontweight='bold')

        ax_bar.set_xlim(0, max_overlap)

        plt.tight_layout()
        plt.savefig(structural_output / "barplot_with_combined_glass_brain.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Top 5 ROI visualization with combined glass brain saved as top5_barplot_with_combined_glass_brain.png in {structural_output}")

