import os
import cv2
import math
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from tqdm.auto import tqdm  # <--- Progress Bar Library

# Import all the isolated functions from our other modules
from src.metadata_utilities import *
from src.image_segmentation_ai import *
from src.quantification_math import *
from src.visualization import plot_diagnostic_grid  # <--- New Dashboard

# ================================================================================
# PIPELINE EXECUTION MASTER ENGINE
# ================================================================================
def run_unified_metadata_and_quantification(
    cellpose_model,
    outline_folder="3_Channel_3",
    quant_folder="2_Channel_2",
    nuc_folder="4_Channel_4",
    outline_file_token="_Ch3",
    quant_file_token="_Ch2",
    nuc_file_token="_Ch4",
    quant_label="CleavCasp3",
    nuc_label="DAPI",
    mask_label="MYH10"
):
    """
    Executes the main segmentation and quantification pipeline.
    All channel strings and labels are configurable via parameters.
    """
    # -------------------------------------------------------------------------
    # SAFETY NET: Fixes the Python "Trailing Comma Tuple" Bug
    # If a user accidentally passes `folder="name",` instead of `folder="name"`
    # this forces it back into a standard string before the script runs.
    # -------------------------------------------------------------------------
    if isinstance(outline_folder, tuple): outline_folder = outline_folder[0]
    if isinstance(quant_folder, tuple): quant_folder = quant_folder[0]
    if isinstance(nuc_folder, tuple): nuc_folder = nuc_folder[0]
    if isinstance(outline_file_token, tuple): outline_file_token = outline_file_token[0]
    if isinstance(quant_file_token, tuple): quant_file_token = quant_file_token[0]
    if isinstance(nuc_file_token, tuple): nuc_file_token = nuc_file_token[0]

    main_folder = prompt_for_main_folder()
    if not main_folder:
        print("No folder selected. Pipeline terminated.")
        return
        
    print(f"Starting Unified Analysis Workflow Engine under: {main_folder}")
    all_records = []
    all_raw_distance_distributions = []
    cached_folder_channels = {}

    valid_extensions = ('.nd2', '.lif', '.czi', '.tif', '.tiff', '.png', '.jpg', '.jpeg')

    # Discovery step for target directories containing baseline reference files
    target_ref_folders = []
    for root, dirs, files in os.walk(main_folder):
        if os.path.basename(root) == outline_folder:
            target_ref_folders.append(root)
            
    if not target_ref_folders:
        print(f"Error: Could not discover any baseline directories named '{outline_folder}'.")
        return

    # --- PROGRESS BAR SETUP ---
    total_files = sum([len([x for x in os.listdir(folder) if not x.startswith('.') and '_Corr' in x]) for folder in target_ref_folders])
    print(f"Found {len(target_ref_folders)} datasets containing {total_files} total images to process.")
    
    pbar = tqdm(total=total_files, desc="Pipeline Progress", unit="img")
    # ---------------------------

    for ref_folder_path in target_ref_folders:
        parent_batch_dir = os.path.dirname(ref_folder_path)
        subfolder_name = os.path.basename(parent_batch_dir)               
        folder_name = os.path.basename(os.path.dirname(parent_batch_dir)) 
        
        folder_identifier = os.path.join(folder_name, subfolder_name)
        
        fldr_stacks_bf = os.path.join(parent_batch_dir, "5_Stacks_BF-edited_test")
        fldr_masks = os.path.join(parent_batch_dir, "6_Masks_test")
        fldr_quant = os.path.join(parent_batch_dir, "7_Quantification_Output_test")
        for f in [fldr_stacks_bf, fldr_masks, fldr_quant]: mkdir_p(f)
            
        if folder_identifier not in cached_folder_channels:
            files_in_ref = os.listdir(ref_folder_path)
            valid_images = [f for f in files_in_ref if f.lower().endswith(valid_extensions) and not f.startswith('.')]
            if valid_images:
                sample_image_path = os.path.join(ref_folder_path, valid_images[0])
                # Ensure this matches your metadata_utilities.py function name
                cached_folder_channels[folder_identifier] = handle_ch_naming_ui(sample_image_path, subfolder_name)
            else:
                cached_folder_channels[folder_identifier] = {}

        channel_ui_info = cached_folder_channels[folder_identifier]

        path_bf_files = sorted([os.path.join(ref_folder_path, x) for x in os.listdir(ref_folder_path) 
                                if not x.startswith('.') and os.path.isfile(os.path.join(ref_folder_path, x)) and '_Corr' in x])
        
        for file_path in path_bf_files:
            filename_bf = os.path.basename(file_path)
            filename_base = os.path.splitext(filename_bf)[0]
            
            img_raw_bf = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img_raw_bf is None: 
                pbar.update(1)
                continue
                
            img_h, img_w = img_raw_bf.shape[:2]
            
            shape = determine_shape(filename_bf)
            position = determine_position(filename_bf)
            treatment = determine_treatment(filename_bf)
            cell_line = determine_cell_line(folder_name, subfolder_name, filename_bf)
            live_time = determine_live_or_time(filename_bf, subfolder_name)
            
            colony_mask = process_smooth_colony_outline_clahe(img_raw_bf)
            
            labels_colony = label(colony_mask)
            props = regionprops(labels_colony)
            
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
            
            roi_x, roi_y, crop_w, crop_h = calculate_crop_coordinates(cx, cy, img_w, img_h, filename_bf)
            
            cropped_bf = img_raw_bf[roi_y:roi_y+crop_h, roi_x:roi_x+crop_w]
            cv2.imwrite(os.path.join(fldr_stacks_bf, filename_base + "-crop.tif"), cropped_bf)
            
            cropped_mask = colony_mask[roi_y:roi_y+crop_h, roi_x:roi_x+crop_w]
            cv2.imwrite(os.path.join(fldr_masks, filename_base + "_mask.tif"), cropped_mask)
            
            sibling_ch2_path = os.path.join(os.path.dirname(file_path.replace(outline_folder, quant_folder)), 
                                            os.path.basename(file_path.replace(outline_folder, quant_folder)).replace(outline_file_token, quant_file_token))
            
            sibling_ch4_path = os.path.join(os.path.dirname(file_path.replace(outline_folder, nuc_folder)), 
                                            os.path.basename(file_path.replace(outline_folder, nuc_folder)).replace(outline_file_token, nuc_file_token))
            
            ch2_area = 0.0
            total_nuclei = 0
            dapi_area = 0.0
            mean_rad, median_rad, stdev_rad = 0.0, 0.0, 0.0
            raw_radial_distribution = []
            
            # Visualizer Variables (Defaults to None in case a channel is missing)
            cropped_ch2_raw_blank = None
            ch2_masked = None
            cropped_ch4_raw_blank = None
            ch4_masked = None
            
            if os.path.exists(sibling_ch2_path):
                img_ch2_raw_full = cv2.imread(sibling_ch2_path, cv2.IMREAD_GRAYSCALE)
                cropped_ch2_raw_blank = img_ch2_raw_full[roi_y:roi_y+crop_h, roi_x:roi_x+crop_w]
                filename_ch2_base = filename_base.replace(outline_file_token, quant_file_token)
                
                cv2.imwrite(os.path.join(fldr_stacks_bf, filename_ch2_base + "-crop.tif"), cropped_ch2_raw_blank)
                
                _, offset_masks = process_caspase_focused_offsets(cropped_ch2_raw_blank, offset=0)
                
                # Safety formatting for mask to prevent cv2.imwrite from crashing
                if offset_masks.dtype == bool:
                    ch2_thresh = (offset_masks.astype(np.uint8) * 255)
                else:
                    ch2_thresh = offset_masks.astype(np.uint8)
                
                cv2.imwrite(os.path.join(fldr_quant, filename_ch2_base + "_" + quant_label + "_thr.tif"), ch2_thresh)
                ch2_masked = cv2.bitwise_and(ch2_thresh, cropped_mask)
                cv2.imwrite(os.path.join(fldr_quant, filename_ch2_base + "_" + quant_label + "_masked.tif"), ch2_masked)
                ch2_area = float(np.sum(ch2_masked == 255))
                
                local_cx = cx - roi_x
                local_cy = cy - roi_y
                mean_rad, median_rad, stdev_rad, raw_radial_distribution = compute_radial_statistics(ch2_masked, local_cx, local_cy)
                
                if len(raw_radial_distribution) > 0:
                    all_raw_distance_distributions.append({
                        "Image_ID": filename_base, 
                        "Condition_Group": treatment, 
                        "Raw_Micron_Distances": raw_radial_distribution
                    })

            if os.path.exists(sibling_ch4_path):
                img_ch4_raw_full = cv2.imread(sibling_ch4_path, cv2.IMREAD_GRAYSCALE)
                cropped_ch4_raw_blank = img_ch4_raw_full[roi_y:roi_y+crop_h, roi_x:roi_x+crop_w]
                filename_ch4_base = filename_base.replace(outline_file_token, nuc_file_token)
                
                cv2.imwrite(os.path.join(fldr_stacks_bf, filename_ch4_base + "-crop.tif"), cropped_ch4_raw_blank)
                
                bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))
                ch4_sub = cv2.morphologyEx(cropped_ch4_raw_blank, cv2.MORPH_TOPHAT, bg_kernel)
                
                total_nuclei, dapi_area_raw, cellpose_masks = segment_nuclei(ch4_sub, cellpose_model)
                dapi_binary_mask = (cellpose_masks > 0).astype(np.uint8) * 255
                cv2.imwrite(os.path.join(fldr_quant, filename_ch4_base + "_" + nuc_label + "_thr.tif"), dapi_binary_mask)
                
                ch4_masked = cv2.bitwise_and(dapi_binary_mask, cropped_mask)
                cv2.imwrite(os.path.join(fldr_quant, filename_ch4_base + "_" + nuc_label + "_masked.tif"), ch4_masked)
                
                dapi_area = float(np.sum(ch4_masked == 255))
                cv2.imwrite(os.path.join(fldr_quant, filename_ch4_base + "_" + nuc_label + "_individual_labels.tif"), cellpose_masks.astype(np.uint16))
            
            overall_density = (total_nuclei / colony_area * 1000.0) if colony_area > 0 else 0.0
            percent_ch2_in_colony = (ch2_area / colony_area * 100.0) if colony_area > 0 else 0.0
            percent_ch2_per_dapi  = (ch2_area / dapi_area * 100.0) if dapi_area > 0 else 0.0
            
            record = {
                "Folder": folder_name, "Subfolder": subfolder_name, "Filename": filename_bf, "Shape": shape,
                "Position in 6wp": position, "Treatment": treatment, "Cell-Line": cell_line, "Live or Timepoint": live_time,
                "Colony_Area_px": colony_area, "Colony_Perimeter_px": perimeter, "Colony_Aspect_Ratio": aspect_ratio,
                "Colony_Roundness": roundness, "Colony_Solidity": solidity,
                f"{quant_label}_Thresh_Area_In_Mask_px": ch2_area, f"{nuc_label}_Count": total_nuclei,
                f"Total_{nuc_label}_Area_In_Mask_px": dapi_area, "Overall_Density_nuclei_per_1000px2": overall_density,
                f"Pct_{quant_label}_per_{mask_label}_Area": percent_ch2_in_colony, f"Pct_{quant_label}_per_{nuc_label}_Area": percent_ch2_per_dapi,
                "Mean_Radial_Distance_um": mean_rad, "Median_Radial_Distance_um": median_rad, "Stdev_Radial_Distance_um": stdev_rad
            }
            record.update(channel_ui_info)
            all_records.append(record)

            # --- CALL THE VISUALIZATION DASHBOARD ---
            plot_diagnostic_grid(
                filename_base, 
                cropped_bf, cropped_ch2_raw_blank, cropped_ch4_raw_blank, 
                cropped_mask, ch2_masked, ch4_masked
            )
            
            # Tick the progress bar
            pbar.update(1)

    pbar.close()

    if not all_records:
        print("Scanned directory block completed with no valid records logged.")
        return

    master_summary_df = pd.DataFrame(all_records)
    summary_xlsx_path = os.path.join(main_folder, "combined_metadata_quantification_summary.xlsx")
    master_summary_df.to_excel(summary_xlsx_path, index=False)
    print(f"\n✓ Master Metadata & Quantification Summary Saved: {summary_xlsx_path}")
    
    exploded_rows = []
    for item in all_raw_distance_distributions:
        img_id = item["Image_ID"]
        cond = item["Condition_Group"]
        for dist in item["Raw_Micron_Distances"]:
            exploded_rows.append({"Image_ID": img_id, "Condition_Group": cond, "Distance_Microns": dist})
            
    if exploded_rows:
        raw_distribution_df = pd.DataFrame(exploded_rows)
        distribution_path = os.path.join(main_folder, "colony_radial_distances_raw_distribution.csv")
        raw_distribution_df.to_csv(distribution_path, index=False)
        print(f"✓ Raw Cellular Intensity Coordinate Distribution Exported: {distribution_path}")
        
    print("\n" + "="*60 + "\nUNIFIED ANALYSIS MATRIX PIPELINE SECURED & COMPLETE\n" + "="*60)
    return master_summary_df