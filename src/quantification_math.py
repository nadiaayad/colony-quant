import numpy as np

def compute_radial_statistics(quant_mask, cx, cy, microns_per_pixel=0.645):
    """Calculates physical distances from the colony centroid."""
    y_indices, x_indices = np.where(quant_mask > 0)
    if len(x_indices) == 0:
        return 0.0, 0.0, 0.0, []
    distances_px = np.sqrt((x_indices - cx) ** 2 + (y_indices - cy) ** 2)
    distances_microns = distances_px * microns_per_pixel
    
    mean_rad = float(np.mean(distances_microns))
    median_rad = float(np.median(distances_microns))
    std_rad = float(np.std(distances_microns))
    
    return mean_rad, median_rad, std_rad, distances_microns.tolist()

def calculate_crop_coordinates(cx, cy, img_w, img_h, filename):
    """Establishes ROI boundaries around the detected colony."""
    if "_T" in filename or "-T" in filename:
        crop_w, crop_h = 1950, 1750 
        roi_x = int(cx - (crop_w / 2))
        roi_y = int((cy + 280) - (crop_h / 2))
    else:
        crop_w, crop_h = 1700, 1700
        roi_x = int(cx - (crop_w / 2))
        roi_y = int(cy - (crop_h / 2))
        
    if roi_x < 0: roi_x = 0
    if roi_y < 0: roi_y = 0
    if (roi_x + crop_w) > img_w: roi_x = img_w - crop_w
    if (roi_y + crop_h) > img_h: roi_y = img_h - crop_h
    return roi_x, roi_y, crop_w, crop_h