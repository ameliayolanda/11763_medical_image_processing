import os

import matplotlib
import pydicom
import numpy as np
import pandas as pd
import scipy
from matplotlib import pyplot as plt, animation

# from activity03
def median_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    #print('shape of sagittal place', img_dcm.shape[1])
    return img_dcm[:, :, img_dcm.shape[1]//2]  

def median_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the median sagittal plane of the CT image provided. """
    return img_dcm[:, img_dcm.shape[2]//2, :]


def MIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the sagittal orientation. """
    return np.max(img_dcm, axis=2)


def AIP_sagittal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the sagittal orientation. """
    return np.mean(img_dcm, axis=2)


def MIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the maximum intensity projection on the coronal orientation. """
    return np.max(img_dcm, axis=1)


def AIP_coronal_plane(img_dcm: np.ndarray) -> np.ndarray:
    """ Compute the average intensity projection on the coronal orientation. """
    return np.mean(img_dcm, axis=1)


def rotate_on_axial_plane(img_dcm: np.ndarray, angle_in_degrees: float) -> np.ndarray:
    """ Rotate the image on the axial plane. """
    return scipy.ndimage.rotate(img_dcm, angle_in_degrees, axes=(1,2), reshape=False)


# from activity02
# def apply_cmap(img: np.ndarray, cmap_name: str = 'bone') -> np.ndarray:
#     """ Apply a colormap to a 2D image. """
#     cmap_function = matplotlib.colormaps[cmap_name]
#     return cmap_function(img)

# def visualize_alpha_fusion(img: np.ndarray, mask: np.ndarray, alpha: float = 0.25):
#     """ Visualize both image and mask in the same plot. """
#     img_sagittal_cmapped = apply_cmap(img, cmap_name='bone')    # Why 'bone'?
#     mask_bone_cmapped = apply_cmap(mask, cmap_name='prism')     # Why 'prism'?
#     mask_bone_cmapped = mask_bone_cmapped * mask[..., np.newaxis].astype('bool')

#     alpha = 0.25
#     plt.imshow(img_sagittal_cmapped * (1-alpha) + mask_bone_cmapped * alpha, aspect=0.98/3.27)
#     plt.title(f'Segmentation with alpha {alpha}')
#     plt.show()


# from activity08
def find_centroid(mask: np.ndarray) -> np.ndarray:
    # Your code here:
    #   Consider using `np.where` to find the indices of the voxels in the mask
    #   ...
    idcs = np.where(mask == 1)
    centroid = np.stack([
        np.mean(idcs[0]),
        np.mean(idcs[1]),
        np.mean(idcs[2]),
    ])
    return centroid

def visualize_axial_slice(
        img: np.ndarray,
        mask: np.ndarray,
        mask_centroid: np.ndarray,
        ):
    """ Visualize the axial slice (first dim.) of a single region with alpha fusion. """

    img_slice = img[int(mask_centroid[0]), :, :]  # Extract slice based on mask centroid
    mask_slice = mask[int(mask_centroid[0]), :, :]

    cmap = matplotlib.cm.get_cmap('bone')  # Get the 'bone' colormap
    norm = matplotlib.colors.Normalize(vmin=np.amin(img_slice), vmax=np.amax(img_slice))  # Normalize image intensity
    img_colored = cmap(norm(img_slice))[:, :, :3]  # Apply colormap to image

    # Apply alpha fusion to visualize the mask on the image slice
    fused_slice = 0.5 * img_colored + 0.5 * np.stack([mask_slice, np.zeros_like(mask_slice), np.zeros_like(mask_slice)], axis=-1)

    plt.imshow(fused_slice)  # Display the fused slice
    plt.show()



def sequence_slices(segmentation):
    if 'SegmentSequence' in segmentation:
        segments_seq = segmentation.SegmentSequence
        sequence_slices = []
        
        for segment in segments_seq:
            segment_number = segment.SegmentNumber
            segment_label = segment.SegmentLabel

            # Check if Per-frame Functional Groups Sequence exists
            if 'PerFrameFunctionalGroupsSequence' in segment:
                per_frame_seq = segment.PerFrameFunctionalGroupsSequence
                # Access Image Position Patient from each frame if needed

            # Append segment information to the list
            sequence_slices.append({
                'SegmentNumber': segment_number,
                'SegmentLabel': segment_label,
            })
        return sequence_slices
    else:
        print("SegmentSequence not found in DICOM file.")
        return []


def extract_ct_data(ct_path_sort):
    ct_data = []
    ct_metadata = []

    for file in ct_path_sort:
        if file.endswith(".dcm"):
            path = os.path.join(ct_path, file)
            dataset = pydicom.dcmread(path)  # Load DICOM file
            ct_data.append(dataset.pixel_array)

            # Extract DICOM header information
            acquisition_number = dataset.get('AcquisitionNumber', 'N/A')
            slice_index = dataset.get('InstanceNumber', 'N/A')
            image_position = dataset.get('ImagePositionPatient', 'N/A')
            referenced_segment_number = dataset.get('ReferencedSegmentNumber', 'N/A')

            # Append DICOM metadata to list
            ct_metadata.append({
                'AcquisitionNumber': acquisition_number,
                'SliceIndex': slice_index,
                'ImagePositionPatient': image_position,
                'ReferencedSegmentNumber': referenced_segment_number
            })
    # Convert metadata list to DataFrame
    metadata_df = pd.DataFrame(ct_metadata)
    return ct_data, metadata_df


# def reslice_segmentation(segmentation_path, ct_metadata_df):
#     # Load segmentation DICOM file
#     segmentation = pydicom.dcmread(segmentation_path)
    
#     # Extract segmentation array
#     segmentation_array = segmentation.pixel_array
    
#     # Get DICOM header information from segmentation
#     acquisition_number = segmentation.get('AcquisitionNumber', 'N/A')
#     slice_index = segmentation.get('InstanceNumber', 'N/A')
    
#     # Match acquisition number and slice index to CT metadata
#     matching_row = ct_metadata_df[(ct_metadata_df['AcquisitionNumber'] == acquisition_number) & (ct_metadata_df['SliceIndex'] == slice_index)]
    
#     if not matching_row.empty:
#         # Perform reslicing based on matching CT metadata
#         resliced_segmentation_array = segmentation_array 
        
#         return resliced_segmentation_array
#     else:
#         print("No matching CT metadata found for segmentation.")
#         return None


if __name__ == '__main__':

    segmentation_path ="./manifest-1714835132158/HCC-TACE-Seg/HCC_015/11-01-1998-NA-CT ABDPEL WWO LIVER-64482/300.000000-Segmentation-54558/1-1.dcm"
    ct_path = "./manifest-1714835132158/HCC-TACE-Seg/HCC_015/11-01-1998-NA-CT ABDPEL WWO LIVER-64482/4.000000-Recon 2 LIVER 3 PHASE AP-06472/"

    ### Segmentation dataset        
    segmentation = pydicom.dcmread(segmentation_path)
    segmentation_array = segmentation.pixel_array 

    # Get centroid of the segmentation mask
    mask_centroid = find_centroid(segmentation_array)
    #print(mask_centroid) #46.42268408 198.31400495 187.65370395
    
    # labels = []
    # segments_seq = segmentation.SegmentSequence
    # for segment in segments_seq:
    #     segment_label = segment.SegmentLabel
    #     labels.append(segment_label)
    # print(labels)

    sequence_slice = sequence_slices(segmentation)
    if sequence_slice:
        print("Sequence slices found:")
        for slice_info in sequence_slice:
            print(slice_info)


    ### CT dataset
    ct_path_sort = sorted(os.listdir(ct_path))
    ct_data, metadata_df = extract_ct_data(ct_path_sort)

    print("CT Metadata DataFrame: ")
    print(metadata_df.head())
    
    # Confirm the slices of the CT image contain only a single acquisition.
    print("Unique value", metadata_df['AcquisitionNumber'].unique())
    


    pixel_len_mm = [3.27, 0.98, 0.98]   # Pixel length in mm [z, y, x]

    #img_dcm = np.flip(img_dcm, axis=0)      # Change orientation (better visualization)
    img_dcm = np.array(ct_data)
    

    # Visualize ROI in CT data using the mask centroid
    visualize_axial_slice(img_dcm, segmentation_array, mask_centroid)
    

    # Show median planes
    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(median_sagittal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[0].set_title('Sagittal')
    ax[1].imshow(median_coronal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[2])
    ax[1].set_title('Coronal')
    fig.suptitle('Median planes')
    plt.show()

    # Show MIP/AIP/Median Sagittal planes
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(median_sagittal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[0].set_title('Median')
    ax[1].imshow(MIP_sagittal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[1].set_title('MIP')
    ax[2].imshow(AIP_sagittal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[2].set_title('AIP')
    fig.suptitle('Sagittal planes')
    plt.show()
    
    # Show MIP/AIP/Median Coronal planes
    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(median_coronal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[0].set_title('Median')
    ax[1].imshow(MIP_coronal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[1].set_title('MIP')
    ax[2].imshow(AIP_coronal_plane(img_dcm), cmap=matplotlib.colormaps['bone'], aspect=pixel_len_mm[0]/pixel_len_mm[1])
    ax[2].set_title('AIP')
    fig.suptitle('Coronal planes')
    plt.show()

    # Create projections varying the angle of rotation
    # Configure visualization colormap
    img_min = np.amin(img_dcm)
    img_max = np.amax(img_dcm)
    cm = matplotlib.colormaps['bone']
    fig, ax = plt.subplots()
    
    #   Configure directory to save results
    os.makedirs('results/', exist_ok=True)
    
    #   Create projections
    n = 16
    projections = []
    for idx, alpha in enumerate(np.linspace(0, 360*(n-1)/n, num=n)):
        rotated_img = rotate_on_axial_plane(img_dcm, alpha)
        projection = MIP_sagittal_plane(rotated_img)
        plt.imshow(projection, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])
        plt.savefig(f'results/Projection_{idx}.png')      # Save animation
        projections.append(projection)  # Save for later animation
    # Save and visualize animation
    animation_data = [
        [plt.imshow(img, animated=True, cmap=cm, vmin=img_min, vmax=img_max, aspect=pixel_len_mm[0] / pixel_len_mm[1])]
        for img in projections
    ]
    anim = animation.ArtistAnimation(fig, animation_data,
                              interval=250, blit=True)
    anim.save('results/Animation.gif')  # Save animation
    plt.show()                              # Show animation



# reference for reading: https://www.hopkinsmedicine.org/health/treatment-tests-and-therapies/computed-tomography-ct-or-cat-scan-of-the-abdomen#:~:text=CT%20scans%20may%20be%20done,of%20time%20before%20the%20procedure.
