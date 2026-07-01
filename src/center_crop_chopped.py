import cv2
import os
import json
import numpy as np
import math
import pandas as pd
import glob
from skimage.measure import label, regionprops
from src.quantification_math import compute_pixel_radial_distances, compute_object_radial_distances
from tqdm.auto import tqdm  # <--- Progress Bar Library

# Import all the isolated functions from our other modules
from src.metadata_utilities import *
from src.image_segmentation_ai import *
from src.quantification_math import *
from src.center_crop_quant import * # Assuming your modular functions are here
from src.visualization import plot_diagnostic_grid
# =========================================================
# MODULES
# =========================================================

def generate_colony_masks_and_crops(img_raw_bf, filename_bf, **kwargs):
    """Finds the colony boundary, calculates geometry, and returns the crops and metrics."""
    img_h, img_w = img_raw_bf.shape[:2]
    
    # Segment Smooth Colony Mask (Assume this is imported from image_segmentation_ai)
    colony_mask = process_smooth_colony_outline_clahe(img_raw_bf)
    
    labels_colony = label(colony_mask)
    props = regionprops(labels_colony)
    
    if len(props) > 0:
        colony_prop = max(props, key=lambda r: r.area)
        cy, cx = colony_prop.centroid
        colony_area = float(np.sum(colony_mask == 255))
    else:
        cx, cy, colony_area = img_w/2, img_h/2, 0.0

    # Calculate Crop Coordinates (Assume this is imported from metadata_utilities)
    roi_x, roi_y, crop_w, crop_h = calculate_crop_coordinates(cx, cy, img_w, img_h, filename_bf)
    
    cropped_bf = img_raw_bf[int(roi_y):int(roi_y+crop_h), int(roi_x):int(roi_x+crop_w)]
    cropped_mask = colony_mask[int(roi_y):int(roi_y+crop_h), int(roi_x):int(roi_x+crop_w)]

    colony_stats = {
        'cx': cx, 'cy': cy, 'roi_x': roi_x, 'roi_y': roi_y, 'crop_w': crop_w, 'crop_h': crop_h,
        'colony_area': colony_area
    }

    return cropped_bf, cropped_mask, colony_stats
def load_precomputed_mask(img_array, filename, **kwargs):
    """
    Loads a pre-cropped mask with robust auto-fixes for Fiji/ImageJ artifacts.
    """
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        
    # --- FIX 1: Fiji 1-value Bug ---
    # If the mask was saved with 1s instead of 255s, scale it up safely
    if np.max(img_array) == 1:
        img_array = (img_array * 255).astype(np.uint8)
    else:
        img_array = img_array.astype(np.uint8)

    # --- FIX 2: Inverted Mask Bug (Bottom-Left Corner Check) ---
    # img_h is the height, so img_h - 1 is the bottom row index.
    # We check if that corner is bright (white background).
    img_h, img_w = img_array.shape
    if img_array[img_h - 1, 0] > 127:
        img_array = cv2.bitwise_not(img_array)
        
    _, binary_mask = cv2.threshold(img_array, 127, 255, cv2.THRESH_BINARY)
    img_h, img_w = binary_mask.shape

    # Extract morphological features
    labels_colony = label(binary_mask)
    props = regionprops(labels_colony)

    if len(props) > 0:
        colony_prop = max(props, key=lambda r: r.area)
        cy, cx = colony_prop.centroid
        colony_area = float(np.sum(binary_mask == 255))
        perimeter = colony_prop.perimeter
        solidity = colony_prop.solidity
        aspect_ratio = colony_prop.major_axis_length / colony_prop.minor_axis_length if colony_prop.minor_axis_length > 0 else 1.0
        roundness = (4.0 * colony_area) / (math.pi * (colony_prop.major_axis_length ** 2)) if colony_prop.major_axis_length > 0 else 1.0
    else:
        cx, cy, colony_area, perimeter, solidity, aspect_ratio, roundness = img_w/2, img_h/2, 0.0, 0.0, 1.0, 1.0, 1.0

    # Pack the stats into the expected dictionary
    c_stats = {
        'cx': cx, 'cy': cy, 'colony_area': colony_area, 
        'perimeter': perimeter, 'solidity': solidity, 
        'aspect_ratio': aspect_ratio, 'roundness': roundness,
        'roi_x': 0, 'roi_y': 0, 'crop_w': img_w, 'crop_h': img_h
    }

    return binary_mask, binary_mask, c_stats
    
def threshold_caspase(img_raw_crop, **kwargs):
    """Thresholds the Cleaved Caspase 3 channel."""
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    ch_sub = cv2.morphologyEx(img_raw_crop, cv2.MORPH_TOPHAT, bg_kernel)
    
    ch_blur = cv2.GaussianBlur(ch_sub, (5, 5), 2)
    # Assume apply_threshold is your standard Triangle/Otsu function
    thresh_img = apply_threshold(ch_blur, method="triangle") 
    
    return thresh_img, None 

def threshold_phh3(img_raw_crop, **kwargs):
    """Thresholds the Cleaved Caspase 3 channel."""
    #bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    #ch_sub = cv2.morphologyEx(img_raw_crop, cv2.MORPH_TOPHAT, bg_kernel)
    
    ch_blur = cv2.GaussianBlur(img_raw_crop, (1, 1), 2)
    # Assume apply_threshold is your standard Triangle/Otsu function
    thresh_img = apply_threshold(ch_blur, method="triangle") 
    
    return thresh_img, None 

def threshold_nuclei(img_raw_crop, model_type='nuclei', diameter=None, **kwargs):
    """Thresholds the nuclear channel using Cellpose dynamically."""
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
    ch_sub = cv2.morphologyEx(img_raw_crop, cv2.MORPH_TOPHAT, bg_kernel)
    
    # Assume segment_nuclei calls Cellpose using the model_type and diameter
    total_nuclei, dapi_area_raw, cellpose_masks = segment_nuclei(ch_sub, model_type=model_type, diameter=diameter)
    
    dapi_binary_mask = (cellpose_masks > 0).astype(np.uint8) * 255
    extra_data = {'total_nuclei': total_nuclei, 'instance_masks': cellpose_masks}
    
    return dapi_binary_mask, extra_data

# =========================================================
# UNIVERSAL QUANTIFICATION ENGINE
# =========================================================

def quantify_channel(img_raw_crop, colony_mask_crop, threshold_func, col_stats, **kwargs):
    """Universal quantification matrix that extracts area and spatial distributions."""
    
    # 1. Generate Threshold using whichever function was passed
    thresh_img, extra_data = threshold_func(img_raw_crop, **kwargs)

    # 2. Enforce the Mask
    masked_img = cv2.bitwise_and(thresh_img, colony_mask_crop)
    
    # 3. Core Metrics
    area_in_mask = float(np.sum(masked_img > 0))
    local_cx = col_stats['cx'] - col_stats['roi_x']
    local_cy = col_stats['cy'] - col_stats['roi_y']
    
    # 4. Extract Spatial Distributions (Now receiving stats AND raw arrays!)
    mean_px, med_px, std_px, raw_px = compute_pixel_radial_distances(masked_img, local_cx, local_cy)
    mean_obj, med_obj, std_obj, raw_obj = compute_object_radial_distances(masked_img, local_cx, local_cy)
    
    return {
        'masked_img': masked_img,
        'thresh_img': thresh_img,
        'area_in_mask': area_in_mask,
        'mean_px': mean_px,
        'med_px': med_px,
        'std_px': std_px,
        'raw_px': raw_px,
        'mean_obj': mean_obj,
        'med_obj': med_obj,
        'std_obj': std_obj,
        'raw_obj': raw_obj,
        'extra_data': extra_data
    }

# =========================================================
# THE MASTER PIPELINE LOOP
# =========================================================

def run_unified_metadata_and_quantification(
    main_folder,           
    cellpose_model,
    outline_setup, 
    quant_channels_list, 
    nuc_setup
):
    """
    Executes the main segmentation and quantification pipeline using dynamic dictionaries.
    Handles progress tracking, comprehensive metadata extraction, and multi-channel metrics.
    Integrates direct-to-disk CSV streaming and runs headlessly for HPC compatibility.
    """
    
    # --- Headless Directory Validation ---
    if not main_folder or not os.path.exists(main_folder):
        print(f"Error: The target directory '{main_folder}' does not exist or is invalid. Pipeline terminated.")
        return
        
    print(f"Starting Unified Analysis Workflow Engine under: {main_folder}")
    all_records = []
    cached_folder_channels = {}

    valid_extensions = ('.nd2', '.lif', '.czi', '.tif', '.tiff', '.png', '.jpg', '.jpeg')

    # Discovery step for target directories containing baseline reference files (the outline channel)
    target_ref_folders = []
    outline_folder_name = outline_setup['folder']
    outline_token = outline_setup['token']
    
    for root, dirs, files in os.walk(main_folder):
        if os.path.basename(root) == outline_folder_name:
            target_ref_folders.append(root)
            
    if not target_ref_folders:
        print(f"Error: Could not discover any baseline directories named '{outline_folder_name}'.")
        return

    # --- SETUP INCREMENTAL CSV FOR RAW DISTANCES ---
    distribution_path = os.path.join(main_folder, "colony_radial_distances_raw_distribution.csv")
    pd.DataFrame(columns=[
        "Image_ID", "Treatment", "Shape", "Channel", "Type", "Distance_Microns"
    ]).to_csv(distribution_path, index=False)
    print(f"Initialized raw distribution stream at: {distribution_path}")

    # --- PROGRESS BAR SETUP ---
    
    target_token = outline_setup.get('token', '_Corr')
    
    total_files = sum([len([x for x in os.listdir(folder) if not x.startswith('.') and target_token in x]) for folder in target_ref_folders])
    print(f"Found {len(target_ref_folders)} datasets containing {total_files} total images to process.")
    
    pbar = tqdm(total=total_files, desc="Pipeline Progress", unit="img")
    # ---------------------------

    for ref_folder_path in target_ref_folders:
        parent_batch_dir = os.path.dirname(ref_folder_path)
        subfolder_name = os.path.basename(parent_batch_dir)              
        folder_name = os.path.basename(os.path.dirname(parent_batch_dir)) 
        
        folder_identifier = os.path.join(folder_name, subfolder_name)
        
        # Ensure Output Folders Exist
        fldr_stacks_bf = os.path.join(parent_batch_dir, "5_Stacks_BF-edited")
        fldr_masks = os.path.join(parent_batch_dir, "6_Masks")
        fldr_quant = os.path.join(parent_batch_dir, "7_Quantification_Output")
        for f in [fldr_stacks_bf, fldr_masks, fldr_quant]: mkdir_p(f)
            
        # Metadata Setup caching
        if folder_identifier not in cached_folder_channels:
            files_in_ref = os.listdir(ref_folder_path)
            valid_images = [f for f in files_in_ref if f.lower().endswith(valid_extensions) and not f.startswith('.')]
            if valid_images:
                sample_image_path = os.path.join(ref_folder_path, valid_images[0])
                cached_folder_channels[folder_identifier] = handle_ch_naming_ui(sample_image_path, subfolder_name)
            else:
                cached_folder_channels[folder_identifier] = {}

        channel_ui_info = cached_folder_channels[folder_identifier]

        path_bf_files = sorted([os.path.join(ref_folder_path, x) for x in os.listdir(ref_folder_path) 
                                if not x.startswith('.') and os.path.isfile(os.path.join(ref_folder_path, x)) and target_token in x])
        
        for file_path in path_bf_files:
            filename_bf = os.path.basename(file_path)
            filename_base = os.path.splitext(filename_bf)[0]
            
            # Extract Core ID by stripping the mask token to help Glob search safely
            core_id = filename_base.replace(outline_token, "").replace("_Ch0", "").replace("_BkgCorr", "").replace("_Ch1", "").replace("_Ch2", "").replace("_Ch3","")
            
            img_raw_bf = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_raw_bf is None: 
                pbar.update(1)
                continue
                
            # --- METADATA EXTRACTION ---
            shape = determine_shape(filename_bf)
            position = determine_position(filename_bf)
            treatment = determine_treatment(filename_bf)
            cell_line = determine_cell_line(folder_name, subfolder_name, filename_bf)
            live_time = determine_live_or_time(filename_bf, subfolder_name)
            
            # Initialize the Pandas Row
            record = {
                "Folder": folder_name, "Subfolder": subfolder_name, "Filename": filename_bf, "Shape": shape,
                "Position in 6wp": position, "Treatment": treatment, "Cell-Line": cell_line, "Live or Timepoint": live_time
            }
            record.update(channel_ui_info)
            
            # Temporary list to hold exploded rows just for THIS image
            image_exploded_rows = []

            # --- 1. OUTLINE GEOMETRY ---
            outline_func = outline_setup['method']
            outline_kwargs = outline_setup.get('kwargs', {})
            
            cropped_bf, cropped_mask, c_stats = outline_func(img_raw_bf, filename_bf, **outline_kwargs)
            
            is_precomputed = (outline_func.__name__ == 'load_precomputed_mask')

            # Only saving the images if not using pre-computed masks
            if not is_precomputed:
                cv2.imwrite(os.path.join(fldr_stacks_bf, filename_base + "-crop.tif"), cropped_bf)
                cv2.imwrite(os.path.join(fldr_masks, filename_base + "_mask.tif"), cropped_mask)
            else:
                print(f"   -> Skipping save: Using pre-computed mask for {filename_base}")
            
            # Save Base Metrics
            record['Colony_Area_px'] = c_stats['colony_area']
            record['Colony_Perimeter_px'] = c_stats.get('perimeter', 0.0)
            record['Colony_Aspect_Ratio'] = c_stats.get('aspect_ratio', 1.0)
            record['Colony_Roundness'] = c_stats.get('roundness', 1.0)
            record['Colony_Solidity'] = c_stats.get('solidity', 1.0)
            
            # Visualization Variables
            dashboard_ch2_raw, dashboard_ch2_mask = None, None
            dashboard_ch4_raw, dashboard_ch4_mask = None, None
            
            # --- 2. DYNAMIC QUANTIFYING CHANNELS ---
            for idx, q_chan in enumerate(quant_channels_list):
                label = q_chan['label']
                q_func = q_chan['method']
                q_kwargs = q_chan.get('kwargs', {})
                
                # --- NEW GLOB SEARCH LOGIC ---
                target_dir = os.path.join(parent_batch_dir, q_chan['folder'])
                search_pattern = os.path.join(target_dir, f"*{core_id}*{q_chan['token']}*")
                possible_files = glob.glob(search_pattern)
                
                print(f"\n[Path Debug] Linking {q_chan['label']}...")
                print(f"  -> Origin Mask: {file_path}") 
                print(f"  -> Built Target Pattern: {search_pattern}")
                print(f"  -> File Exists?: {len(possible_files) > 0}")
                print(f"  -> File is: {possible_files[0]}")

                if possible_files:
                    sibling_q_path = possible_files[0]
                    # Loaded with UNCHANGED to fix the 16-bit to black issue
                    img_q_raw = cv2.imread(sibling_q_path, cv2.IMREAD_UNCHANGED)
                    
                    cropped_q_raw = img_q_raw[
                        int(c_stats['roi_y']):int(c_stats['roi_y']+c_stats['crop_h']), 
                        int(c_stats['roi_x']):int(c_stats['roi_x']+c_stats['crop_w'])
                    ]
                    
                    filename_q_base = filename_base.replace(outline_token, q_chan['token'])
                    if not is_precomputed:
                        cv2.imwrite(os.path.join(fldr_stacks_bf, filename_q_base + "-crop.tif"), cropped_q_raw)
                    else:
                        print(f"   -> Skipping save: Using pre-computed mask for {filename_q_base}")
                    
                    
                    # Core Modular Quantification
                    q_results = quantify_channel(cropped_q_raw, cropped_mask, q_func, c_stats, **q_kwargs)
                    
                    cv2.imwrite(os.path.join(fldr_quant, f"{filename_q_base}_{label}_thr.tif"), q_results['thresh_img'])
                    cv2.imwrite(os.path.join(fldr_quant, f"{filename_q_base}_{label}_masked.tif"), q_results['masked_img'])
                    
                    # Update Spreadsheet
                    record[f'{label}_Area_in_Mask_px'] = q_results['area_in_mask']
                    record[f'Pct_{label}_per_Colony_Area'] = (q_results['area_in_mask'] / c_stats['colony_area'] * 100.0) if c_stats['colony_area'] > 0 else 0.0
                    
                    # Directly map the pre-calculated metrics
                    record[f'{label}_Mean_Px_Dist'] = q_results['mean_px']
                    record[f'{label}_Median_Px_Dist'] = q_results['med_px']
                    record[f'{label}_StDev_Px_Dist'] = q_results['std_px']
                    
                    record[f'{label}_Mean_Obj_Dist'] = q_results['mean_obj']
                    record[f'{label}_Median_Obj_Dist'] = q_results['med_obj']
                    record[f'{label}_StDev_Obj_Dist'] = q_results['std_obj']

                    # Save first quantifying channel to dashboard for legacy plotting
                    if idx == 0:
                        dashboard_ch2_raw = cropped_q_raw
                        dashboard_ch2_mask = q_results['masked_img']

                    # Incrementally collect raw distances for this channel
                    for dist in q_results.get('raw_px', []):
                        image_exploded_rows.append({"Image_ID": filename_base, "Treatment": treatment, "Shape": shape, "Channel": label, "Type": "Pixel", "Distance_Microns": dist})
                    for dist in q_results.get('raw_obj', []):
                        image_exploded_rows.append({"Image_ID": filename_base, "Treatment": treatment, "Shape": shape, "Channel": label, "Type": "Object", "Distance_Microns": dist})
            
            # --- 3. NUCLEAR CHANNEL ---
            nuc_label = nuc_setup['label']
            nuc_func = nuc_setup['method']
            nuc_kwargs = nuc_setup.get('kwargs', {})
            
            # --- NEW GLOB SEARCH LOGIC ---
            target_nuc_dir = os.path.join(parent_batch_dir, nuc_setup['folder'])
            nuc_pattern = os.path.join(target_nuc_dir, f"*{core_id}*{nuc_setup['token']}*")
            possible_nuc = glob.glob(nuc_pattern)
            
            if possible_nuc:
                sibling_nuc_path = possible_nuc[0]
                # Loaded with UNCHANGED to fix the 16-bit to black issue
                img_nuc_raw = cv2.imread(sibling_nuc_path, cv2.IMREAD_UNCHANGED)
                cropped_nuc_raw = img_nuc_raw[
                    int(c_stats['roi_y']):int(c_stats['roi_y']+c_stats['crop_h']), 
                    int(c_stats['roi_x']):int(c_stats['roi_x']+c_stats['crop_w'])
                ]
                
                filename_nuc_base = filename_base.replace(outline_token, nuc_setup['token'])
                if not is_precomputed:
                    cv2.imwrite(os.path.join(fldr_stacks_bf, filename_nuc_base + "-crop.tif"), cropped_nuc_raw)
                else:
                    print(f"   -> Skipping save: Using pre-computed mask for {filename_nuc_base}")
                                
                # Pass the loaded Cellpose model to the function dynamically
                if 'model_type' in nuc_kwargs and not isinstance(nuc_kwargs['model_type'], str):
                     nuc_kwargs['model_type'] = cellpose_model
                     
                nuc_results = quantify_channel(cropped_nuc_raw, cropped_mask, nuc_func, c_stats, **nuc_kwargs)
                
                cv2.imwrite(os.path.join(fldr_quant, f"{filename_nuc_base}_{nuc_label}_thr.tif"), nuc_results['thresh_img'])
                cv2.imwrite(os.path.join(fldr_quant, f"{filename_nuc_base}_{nuc_label}_masked.tif"), nuc_results['masked_img'])
                
                if nuc_results['extra_data'] and 'instance_masks' in nuc_results['extra_data']:
                    cv2.imwrite(os.path.join(fldr_quant, f"{filename_nuc_base}_{nuc_label}_individual_labels.tif"), nuc_results['extra_data']['instance_masks'].astype(np.uint16))
                
                # Nuc Metrics
                record[f'{nuc_label}_Area_in_Mask_px'] = nuc_results['area_in_mask']
                total_nuclei = nuc_results['extra_data'].get('total_nuclei', 0) if nuc_results['extra_data'] else 0
                record[f'{nuc_label}_Count'] = total_nuclei
                record['Overall_Density_nuclei_per_1000px2'] = (total_nuclei / c_stats['colony_area'] * 1000.0) if c_stats['colony_area'] > 0 else 0.0

                # Directly map pre-calculated metrics for nuclei
                record[f'{nuc_label}_Mean_Px_Dist'] = nuc_results['mean_px']
                record[f'{nuc_label}_Median_Px_Dist'] = nuc_results['med_px']
                record[f'{nuc_label}_StDev_Px_Dist'] = nuc_results['std_px']
                
                record[f'{nuc_label}_Mean_Obj_Dist'] = nuc_results['mean_obj']
                record[f'{nuc_label}_Median_Obj_Dist'] = nuc_results['med_obj']
                record[f'{nuc_label}_StDev_Obj_Dist'] = nuc_results['std_obj']

                dashboard_ch4_raw = cropped_nuc_raw
                dashboard_ch4_mask = nuc_results['masked_img']

                # Incrementally collect raw distances for Nuclear channel
                for dist in nuc_results.get('raw_px', []):
                    image_exploded_rows.append({"Image_ID": filename_base, "Treatment": treatment, "Shape": shape, "Channel": nuc_label, "Type": "Pixel", "Distance_Microns": dist})
                for dist in nuc_results.get('raw_obj', []):
                    image_exploded_rows.append({"Image_ID": filename_base, "Treatment": treatment, "Shape": shape, "Channel": nuc_label, "Type": "Object", "Distance_Microns": dist})

            all_records.append(record)
            
            # --- FLUSH IMAGE RAW DATA TO DISK IMMEDIATELY ---
            if image_exploded_rows:
                pd.DataFrame(image_exploded_rows).to_csv(distribution_path, mode='a', header=False, index=False)

            # --- VISUALIZATION DASHBOARD ---
            try:
                plot_diagnostic_grid(
                    filename_base, 
                    cropped_bf, dashboard_ch2_raw, dashboard_ch4_raw, 
                    cropped_mask, dashboard_ch2_mask, dashboard_ch4_mask
                )
            except Exception as e:
                pass # Silently skip plotting if missing channels

            # Tick progress
            pbar.update(1)

    pbar.close()

    if not all_records:
        print("Scanned directory block completed with no valid records logged.")
        return

    # Export Master DataFrame (The summarized excel sheet)
    master_summary_df = pd.DataFrame(all_records)
    summary_xlsx_path = os.path.join(main_folder, "combined_metadata_quantification_summary.xlsx")
    master_summary_df.to_excel(summary_xlsx_path, index=False)
    
    print(f"\n✓ Master Metadata & Quantification Summary Saved: {summary_xlsx_path}")
    print(f"✓ Raw Cellular Spatial Distribution CSV completely streamed and saved: {distribution_path}")
        
    print("\n" + "="*60 + "\nUNIFIED ANALYSIS MATRIX PIPELINE SECURED & COMPLETE\n" + "="*60)
    return master_summary_df