import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tiff

### USING LAPLACIAN TRANSFORM 
def save_one_best_slice_with_laplacian(file_path, 
                                       save_file_path, 
                                       z_axis=0, 
                                       channel_axis=1, 
                                       ksize=11):
    # open the full 3D image with multiple channels
    image_3D = tiff.imread(file_path)
    # extract 3D dapi image
    dapi_3D = np.take(image_3D, indices=0, axis=channel_axis) #image_3D[:,0,:,:]
    n_best_slice, _ = focus_detection_laplacian(dapi_3D, z_axis, ksize)
    # construct new tiff image for 2D image for all slices
    image_2D_channels = np.take(image_3D, indices=n_best_slice, axis=z_axis)
    tiff.imwrite(save_file_path, image_2D_channels)
    return image_2D_channels

def focus_detection_laplacian(image, z_axis=0, ksize=9):
    """
    Go through slices of a 3D image and find the best slice using variance of Laplacian transform
    Args:
        image (array): 3D image
        z_axis (int): axis of the z-coordinate
        ksize (int, odd): kernel size of laplacian transform
    """
    total_z_slices = image.shape[z_axis]
    lp_values = np.zeros(total_z_slices)
    for z in range(total_z_slices):
        _, lp_values[z] = obtain_laplacian_variance(np.take(image, indices=z, axis=z_axis), ksize)

    best_focused_index = lp_values.argmax() # to look for the index position with the highest sd value
    return best_focused_index, lp_values
    
def obtain_laplacian_variance(image_slice, ksize):
    laplacian = cv2.Laplacian(image_slice, cv2.CV_64F, ksize=ksize)
    variance = laplacian.var()
    return laplacian, variance

### USING SIMPLE VARIANCE OF SLICE
def save_one_best_slice_with_variance(file_path, 
                                      save_file_path, 
                                      z_axis=0, 
                                      channel_axis=1):
    # open the full 3D image with multiple channels
    image_3D = tiff.imread(file_path)
    # extract 3D dapi image
    dapi_3D = np.take(image_3D, indices=0, axis=channel_axis) #image_3D[:,0,:,:]
    n_best_slice, _ = focus_detection_variance(dapi_3D, z_axis)
    # construct new tiff image for 2D image for all slices
    image_2D_channels = np.take(image_3D, indices=n_best_slice, axis=z_axis)
    tiff.imwrite(save_file_path, image_2D_channels)
    return image_2D_channels

def focus_detection_variance(image, z_axis=0):
    """
    Go through slices of a 3D image and find the best slice using variance of Laplacian transform
    Args:
        image (array): 3D image
        z_axis (int): axis of the z-coordinate
    """
    total_z_slices = image.shape[z_axis]
    variance_values = np.zeros(total_z_slices)
    for z in range(total_z_slices):
       variance_values[z] = obtain_simple_variance(np.take(image, indices=z, axis=z_axis))

    best_focused_index = variance_values.argmax() # to look for the index position with the highest sd value
    return best_focused_index, variance_values

def obtain_simple_variance(image_slice):
    return image_slice.var()

def from_3D_to_2D_image(image_3D, best_focused_index, z_axis=0, n_slices=0, projection_type='none'):
    """
    Take 3D image and make projection to 2D
    """
    start_index = max(0, best_focused_index-n_slices)
    end_index = min(image_3D.shape[z_axis], best_focused_index+n_slices)

    # Use np.take to extract the slices along the specified axis
    indices = list(range(start_index, end_index))
    image_2project = np.take(image_3D, indices=indices, axis=z_axis)

    # Choose the projection type
    if projection_type == 'max':
        projection = np.max(image_2project, axis=z_axis)
    elif projection_type == 'min':
        projection = np.min(image_2project, axis=z_axis)
    elif projection_type == 'avg':
        projection = np.mean(image_2project, axis=z_axis)
    elif projection_type == 'none':
        projection = np.take(image_3D, indices=best_focused_index, axis=z_axis)
    else:
        raise ValueError("Invalid projection type. Choose from 'max', 'min', 'avg', or 'none'.")
    return projection