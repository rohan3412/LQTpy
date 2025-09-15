import nibabel as nib
import numpy as np
from dipy.io.streamline import load_tck
from dipy.tracking.utils import density_map
from dipy.tracking.streamline import Streamlines
from scipy.ndimage import label

print("imported")
tck_file = 'global_tractography_sift.tck'
print("loading")
tck = load_tck(tck_file, reference='mni152.nii.gz')
print("finished loading")
streamlines = tck.streamlines

lesion_nii = nib.load('lesion.nii')
lesion_data = lesion_nii.get_fdata().astype(bool)

affine = lesion_nii.affine

from tqdm import tqdm

def streamline_intersects_lesion(streamline, lesion_mask, affine):
    streamline_vox = np.round(nib.affines.apply_affine(np.linalg.inv(affine), streamline)).astype(int)
    for point in streamline_vox:
        x, y, z = point
        if (0 <= x < lesion_mask.shape[0] and
            0 <= y < lesion_mask.shape[1] and
            0 <= z < lesion_mask.shape[2]):
            if lesion_mask[x, y, z]:
                return True
    return False


disconnected_streamlines = Streamlines([
    sl for sl in tqdm(streamlines, desc="Checking streamlines")
    if streamline_intersects_lesion(sl, lesion_data, affine)
])

print(f"Total streamlines: {len(streamlines)}")
print(f"Streamlines passing through lesion: {len(disconnected_streamlines)}")

density_all = density_map(streamlines, affine, lesion_data.shape)
density_disconnected = density_map(disconnected_streamlines, affine, lesion_data.shape)



with np.errstate(divide='ignore', invalid='ignore'):
    disconnectome_map = np.true_divide(density_disconnected, density_all)
    disconnectome_map[np.isnan(disconnectome_map)] = 0  # Set NaNs to zero

disconnectome_nii = nib.Nifti1Image(disconnectome_map.astype(np.float32), affine)
nib.save(disconnectome_nii, 'disconnectome_map_sift.nii.gz')
