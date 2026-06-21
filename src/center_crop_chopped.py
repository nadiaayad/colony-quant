import cv2
import numpy as np
import math
from skimage.measure import label, regionprops

def generate_colony_masks_and_crops(img_raw_bf, filename_bf):
    """
    Finds the colony boundary, calculates geometry, and returns the crops and metrics.
    """
    img_h, img_w = img_raw_bf.shape[:2]
    
    # 1. Segment Smooth Colony Mask
    colony_mask = process_smooth_colony_outline(img_raw_bf)
    
    labels_colony = label(colony_mask)
    props = regionprops(labels_colony)
    
    # 2. Extract Geometric Metrics
    if len(props) > 0:
        colony_prop = max(props, key=lambda r: r.area)
        cy, cx = colony_prop.centroid
        colony_area = float(np.sum(colony_mask == 255))
        perimeter = colony_prop.perimeter
        solidity = colony_prop.solidity
        aspect_ratio = colony_prop.major_axis_length / colony_prop.minor_axis_length if colony_prop.minor_axis_length > 0 else 1.0
        roundness = (4.0 * colony_area) / (math.pi * (colony_prop.major_axis_length ** 2)) if colony_prop.major_axis_length > 0 else 1.0
    else:
        cx, cy, colony_area, perimeter, solidity, aspect_ratio, roundness = img_w/2, img_h/2, 0.0, 0.0, 1.0, 1.0, 1.0

    # 3. Calculate Crop Coordinates
    roi_x, roi_y, crop_w, crop_h = calculate_crop_coordinates(cx, cy, img_w, img_h, filename_bf)
    
    # 4. Execute Crops
    cropped_bf = img_raw_bf[int(roi_y):int(roi_y+crop_h), int(roi_x):int(roi_x+crop_w)]
    cropped_mask = colony_mask[int(roi_y):int(roi_y+crop_h), int(roi_x):int(roi_x+crop_w)]

    # 5. Pack Statistics
    colony_stats = {
        'cx': cx, 'cy': cy, 'roi_x': roi_x, 'roi_y': roi_y, 'crop_w': crop_w, 'crop_h': crop_h,
        'colony_area': colony_area, 'perimeter': perimeter, 'solidity': solidity, 
        'aspect_ratio': aspect_ratio, 'roundness': roundness
    }

    return cropped_bf, cropped_mask, colony_stats

def threshold_caspase(img_raw_crop, **kwargs):
    """
    Thresholds the Cleaved Caspase 3 channel.
    (Update logic below with your specific Caspase offset/thresholding parameters).
    """
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    ch_sub = cv2.morphologyEx(img_raw_crop, cv2.MORPH_TOPHAT, bg_kernel)
    
    ch_blur = cv2.GaussianBlur(ch_sub, (5, 5), 2)
    thresh_img = apply_threshold(ch_blur, method="triangle") 
    
    return thresh_img, None # Returning None for extra_data

def threshold_nuclei(img_raw_crop, **kwargs):
    """
    Thresholds the DAPI channel using Cellpose.
    """
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    ch_sub = cv2.morphologyEx(img_raw_crop, cv2.MORPH_TOPHAT, bg_kernel)
    
    # Segment via Cellpose
    total_nuclei, dapi_area_raw, cellpose_masks = segment_nuclei(ch_sub)
    
    # Convert instance map to binary mask
    dapi_binary_mask = (cellpose_masks > 0).astype(np.uint8) * 255
    
    # Passing out total_nuclei and the raw mask array as extra data
    extra_data = {'total_nuclei': total_nuclei, 'instance_masks': cellpose_masks}
    return dapi_binary_mask, extra_data