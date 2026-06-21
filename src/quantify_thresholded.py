def quantify_channel(img_raw_crop, colony_mask_crop, threshold_func, col_stats, **kwargs):
    """
    Universal quantification matrix. 
    Applies the specific thresholding function, masks out the background, and gathers metrics.
    """
    # 1. Generate Threshold
    thresh_img, extra_data = threshold_func(img_raw_crop, **kwargs)

    # 2. Mask Verification: Remove anything outside the colony mask
    masked_img = cv2.bitwise_and(thresh_img, colony_mask_crop)
    
    # 3. Core Metrics Extraction
    area_in_mask = float(np.sum(masked_img == 255))
    
    # Calculate localized center coordinates for the crop dimensions
    local_cx = col_stats['cx'] - col_stats['roi_x']
    local_cy = col_stats['cy'] - col_stats['roi_y']
    
    # 4. Generate Radial Distributions
    mean_rad, median_rad, stdev_rad, raw_radial_dist = compute_radial_statistics(masked_img, local_cx, local_cy)
    
    return {
        'masked_img': masked_img,
        'thresh_img': thresh_img,
        'area_in_mask': area_in_mask,
        'mean_rad': mean_rad,
        'median_rad': median_rad,
        'stdev_rad': stdev_rad,
        'raw_radial_dist': raw_radial_dist,
        'extra_data': extra_data
    }