import numpy as np
from skimage.measure import label, regionprops
from scipy.optimize import linear_sum_assignment

def compute_roi_metrics(gt_mask, pred_mask, iou_threshold=0.5):
    # Label individual objects in the ground truth and predicted masks (in case not labelled)
    # gt_labelled = label(gt_mask)
    # pred_labelled = label(pred_mask)

    # Get properties of each region (bounding box, area, centroid, etc.)
    gt_regions, pred_regions = find_region_properties(gt_mask, pred_mask)
    
    # Create iou matrix for all regions
    iou_matrix = create_iou_matrix(gt_regions, pred_regions, gt_mask, pred_mask)

    # Count TP, FP, FN
    TP = calculate_true_positive(iou_matrix,
                                 iou_threshold)
    FP = len(pred_regions) - TP
    FN = len(gt_regions) - TP

    sensitivity, precision, f1 = calculate_metrics(TP, FP, FN)

    return {"True Positives": TP,
            "False Positives": FP,
            "False Negatives": FN,
            "Sensitivity (Recall)": sensitivity,
            "Precision": precision,
            "F1-score": f1}

def find_region_properties(gt_labelled, pred_labelled):
    return regionprops(gt_labelled), regionprops(pred_labelled)

def create_iou_matrix(gt_regions, 
                      pred_regions, 
                      gt_mask, 
                      pred_mask):
    
    
    # Create IoU matrix between GT objects and predicted objects
    iou_matrix = np.zeros((len(gt_regions), len(pred_regions)))

    for i, gt in enumerate(gt_regions):
        for j, pred in enumerate(pred_regions):
            # Compute Intersection over Union (IoU)
            intersection = np.logical_and(gt_mask == gt.label, pred_mask == pred.label).sum()
            union = np.logical_or(gt_mask == gt.label, pred_mask == pred.label).sum()
            iou_matrix[i, j] = intersection / union if union > 0 else 0

    return iou_matrix

def calculate_true_positive(iou_matrix, 
                            iou_threshold):
    """
    Calculate true positive value from iou matrix 
    """
    gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)  # Maximize IoU
    true_positive = sum(iou_matrix[gt, pred] > iou_threshold for gt, pred in zip(gt_indices, pred_indices))
    return true_positive

def calculate_metrics(TP, FP, FN):
    """
    Calculate Sensitivity, Precision and F1-score given TP, FP, FN
    """
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1 = (2 * precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
    return sensitivity, precision, f1