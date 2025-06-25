import os 
import random 
from tqdm import tqdm

import tifffile as tiff
from scipy.spatial import cKDTree

import pandas as pd
import numpy as np

from cellpose import models
import neuruloid_segmentation.utils as u

def segment_dapi_image(file_path, 
                       save_path, 
                       model_path, 
                       nuclei_diam=30):
    """
    Segment single dapi tiff file and return mask in .npy file
    Args:
        file_path: path to dapi image to segment
        save_path: file path to save mask in .npy
        model_path: path to trained model
        nuclei_diam: average diameter of nucleus
    Returns:
        mask
    """
    image = tiff.imread(file_path)
    model = models.CellposeModel(pretrained_model=model_path)

    masks, flows, _ = model.eval(image,
                                 diameter=nuclei_diam,
                                 channels=[0,0],
                                 do_3D=False)
    np.save(save_path, masks)
    return masks

def plot_mask(masks, ax, axis="off", origin="lower"):
    """
    Plot mask given axes by generating random colours
    """
    unique_masks = np.unique(masks)
    colored_mask = np.zeros((*masks.shape, 3), dtype=np.uint8)
    for mask_id in unique_masks:
        if mask_id == 0:
            continue  # Skip background
        color = np.random.randint(0, 255, size=(3,))
        colored_mask[masks == mask_id] = color

    ax.imshow(colored_mask, origin=origin)
    ax.axis(axis)

def plot_selected_masks(masks, mask_ids, ax):
    """
    Plot mask given subset of mask id and axes
    """
    selected_mask = np.zeros((*masks.shape, 3), dtype=np.uint8)
    np.random.seed(42)
    
    for mask_id in mask_ids:
        if mask_id == 0:
            continue  # Skip background
        color = np.random.randint(0, 255, size=(3,))
        selected_mask[masks == mask_id] = color

    ax.imshow(selected_mask)
    ax.axis("off")

def mask_to_dataframe(full_image_path, 
                      mask, 
                      pixel_size,
                      save_data_path):
    """
    Takes the tiff file of image with multiple channels (channel 1 is Dapi)
    Returns dataframe with columns:
        - cell_id
        - distance from edge
        - intensities for all channels 
    """
    # read image into matrix
    full_image = tiff.imread(full_image_path) # open tiff file
    n_channels = full_image.shape[0]

    dataframe = pd.DataFrame(columns=["cell_id", "distance"])
    dataframe["cell_id"] = np.sort(np.unique(mask))[1:] # save cell_ids in dataframe
    dataframe["distance"] = nearest_distance_from_edge(mask) * pixel_size # save nearest distance for each cell_id from edge

    for i in range(n_channels):
        # save intensity for each cell_id from all channels
        tf_image = full_image[i,:,:]
        dataframe[f"channel_{i+1}"] = extract_intensity_from_mask(mask, tf_image)

    dataframe.to_csv(save_data_path) # save dataframe 
    return dataframe

def extract_intensity_from_mask(mask, signal_image):
    """
    Given mask and image, extract average intensities for each roi of mask id
    """
    unique_masks = np.sort(np.unique(mask))[1:]
    avg_intensities = np.array([np.mean(signal_image[mask == mask_id]) for mask_id in unique_masks])
    return avg_intensities

def nearest_distance_from_edge(mask, return_full=False):
    """
    Computes the nearest distances of each unique mask id from the edge
    """
    # get contour of neuruloid
    bin_image = u.get_binary_image(mask, 
                                   intensity_threshold=0, 
                                   area_threshold=10**2,
                                   binary_dilation_iter=40,
                                   binary_closing_iter=2)
    contour, _  = u.get_contour_and_centroid(bin_image,
                                             smoothing_factor=5*10**4)
    # get centre coordinates of each cell_id
    nucleus_centres = get_mask_centers(mask)
    # get the nearest distance of each nucleus centre to contour
    distance_array = u.nearest_distance_from_contour(contour, nucleus_centres)
    if return_full==True:
        return distance_array, nucleus_centres, contour
    return distance_array

def get_mask_centers(masks):
    """
    Computes the center coordinates for each unique mask id.
    """
    unique_masks = np.sort(np.unique(masks))[1:]
    centers = []
    for mask_id in unique_masks:
        y, x = np.where(masks == mask_id)
        center_x = int(np.mean(x))
        center_y = int(np.mean(y))
        centers.append((center_y, center_x))
    return np.array(centers)