import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
import pandas as pd
plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "monospace"
from scipy import ndimage

from skimage import measure, morphology
from skimage.measure import find_contours
from skimage.morphology import disk
from scipy import ndimage, interpolate
from scipy.spatial import Delaunay
from scipy.signal import medfilt
from sklearn.cluster import KMeans

from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree

def get_binary_image(DAPI_image, 
                     intensity_threshold=0, 
                     area_threshold=10**2,
                     binary_dilation_iter=40,
                     binary_closing_iter=2):
    bin_image = DAPI_image > intensity_threshold
    for i in range(binary_dilation_iter):
        bin_image = ndimage.binary_dilation(bin_image)
    bin_image = ndimage.binary_erosion(bin_image, structure=disk(10), iterations=binary_closing_iter)
    bin_image = morphology.remove_small_objects(bin_image, min_size=area_threshold, connectivity=5)
    return bin_image

def contour_area(contour):
    x, y = contour[:, 1], contour[:, 0]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def smoothing_contour(contour, smoothing_factor=10**3):
    # Fit a spline to the contour points
    tck, u = interpolate.splprep([contour[:, 0], contour[:, 1]], s=smoothing_factor, per=True)
    u_new = np.linspace(u.min(), u.max(), len(contour))
    x_new, y_new = interpolate.splev(u_new, tck, der=0)
    smoothed_contour = np.stack([x_new, y_new], axis=1)
    return smoothed_contour

def calculate_area_centroid(points):
    """To calculate the area centroid of irregularly shaped polygon"""
    x0, y0 = points[:,0], points[:,1]
    x1, y1 = np.roll(points[:,0], -1), np.roll(points[:,1], -1)
    # calculate the cross product and areas
    cross_product = x0 * y1 - x1 * y0
    A = 0.5 * np.sum(cross_product)
    # compute centroid coordinates
    C_x = np.sum((x0+x1) * cross_product) / (6*A)
    C_y = np.sum((y0+y1) * cross_product) / (6*A)
    centre = np.array([C_x, C_y])
    return centre

def get_contour_and_centroid(binary_image, 
                             area_threshold=10**4, 
                             smoothing_factor=10**3, 
                             annulus=False):
    contours = measure.find_contours(binary_image, 0)
    areas = np.array([contour_area(contour) for contour in contours])

    if annulus:
        # for annulus, save two contours
        large_contour = contours[np.where(areas>area_threshold)[0][0]]
        small_contour = contours[np.where(areas>area_threshold)[0][1]]
        large_contour = smoothing_contour(large_contour, smoothing_factor)
        centroid = calculate_area_centroid(large_contour)
        return large_contour, small_contour, centroid

    else:
        contour = contours[np.where(areas>area_threshold)[0][0]]
        contour = smoothing_contour(contour, smoothing_factor)
        centroid = calculate_area_centroid(contour)
        return contour, centroid
    
def nearest_distance_from_contour(contour, coordinates):
    tree = cKDTree(contour)
    distance_array, _ = tree.query(coordinates)
    return distance_array 

def generate_colors(n):
    """Generate n colors where n<10 from tab10"""
    cmap = plt.get_cmap("tab10")  # Good for up to 10 colors
    return [cmap(i % 10) for i in range(n)]

def generate_random_colors(n):
    """"""
    return [(random.random(), random.random(), random.random()) for _ in range(n)]

def fit_linear_line_and_evaluate(x, y, prop=True):
    """
    Fits a linear line (y = mx + b) to the provided data and evaluates the fit.
    
    Parameters:
    - x: array-like, the independent variable (e.g., radii).
    - y: array-like, the dependent variable (e.g., mean or variance of gamma).
    - plot: bool, if True, plots the data and the fitted line.
    
    Returns:
    - slope: float, the slope of the fitted line.
    - intercept: float, the y-intercept of the fitted line.
    - r_squared: float, the coefficient of determination (R^2) of the fit.
    """
    # Fit a linear line y = mx + b
    slope, intercept = np.polyfit(x, y, 1)
    
    # Predicted values from the linear model
    y_pred = slope * x + intercept
    
    # Calculate R^2 (coefficient of determination)
    r_squared = calculate_r_squared(y, y_pred)

    return slope, intercept, r_squared

def fit_proportional_line_and_evaluate(x, y):
    """
    Fits a proportional linear line (y = mx) to the provided data and evaluates the fit.
    
    Parameters:
    - x: array-like, the independent variable (e.g., radii).
    - y: array-like, the dependent variable (e.g., mean or variance of gamma).
    - plot: bool, if True, plots the data and the fitted line.
    
    Returns:
    - slope: float, the slope of the fitted line.
    - r_squared: float, the coefficient of determination (R^2) of the fit.
    """
    # Reshape x to a 2D array as required by np.linalg.lstsq
    x_reshaped = x[:, np.newaxis]
    
    # Solve for the slope using np.linalg.lstsq, with intercept fixed to 0
    slope, _, _, _ = np.linalg.lstsq(x_reshaped, y, rcond=None)
    slope = slope[0]  # Extract the slope value from the result
    
    # Predicted values from the proportional linear model (y = mx)
    y_pred = slope * x
    
    # Calculate R^2 (coefficient of determination)
    r_squared = calculate_r_squared(y, y_pred)
    
    return slope, r_squared

def calculate_r_squared(y, y_pred):
    # Calculate R^2 (coefficient of determination)
    ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
    ss_residual = np.sum((y - y_pred)**2)  # Residual sum of squares
    r_squared = 1 - (ss_residual / ss_total)  # R^2 formula
    return r_squared