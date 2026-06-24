import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
import scipy.ndimage as ndi
from ultralytics import FastSAM
from cellpose import models

def apply_threshold(img, method="otsu"):
    """Fallback standard threshold algorithm."""
    try:
        m = method.lower().strip()
        if m == "otsu": thresh = threshold_otsu(img)
        elif m == "li": thresh = threshold_li(img)
        elif m == "triangle": thresh = threshold_triangle(img)
        else: thresh = threshold_otsu(img)
        return (img > thresh).astype(np.uint8) * 255
    except Exception:
        return (img > np.mean(img)).astype(np.uint8) * 255

def apply_fiji_auto_contrast(img, saturate_percent=1.0):
    """Acts as our Outlier Filter to clip hot pixels/debris."""
    p_high = np.percentile(img, 100.0 - saturate_percent)
    p_low = np.percentile(img, 0.1)
    if p_high <= p_low: return img.astype(np.uint8)
    return ((np.clip(img, p_low, p_high) - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)

def segment_nuclei(img, model_type='nuclei', diameter=None):
    """
    Segments an image using Cellpose.
    Handles string names ('cpsam'), file paths, OR pre-loaded CellposeModel objects.
    """
    # 1. Check if model_type is already a pre-loaded Cellpose model object
    if hasattr(model_type, 'eval'):
        model = model_type
        
    # 2. Check if it is a built-in Cellpose model string
    elif isinstance(model_type, str) and model_type in ['nuclei', 'cyto', 'cyto2', 'cyto3', 'cpsam']:
        model = models.CellposeModel(gpu=True, model_type=model_type)
        
    # 3. Otherwise, treat it as a file path to a custom fine-tuned model
    else:
        model = models.CellposeModel(gpu=True, pretrained_model=model_type)

    # Execute the segmentation
    masks, flows, styles = model.eval(img, diameter=diameter, channels=[0, 0])

    # Extract base metrics
    total_nuclei = len(np.unique(masks)) - 1 if np.max(masks) > 0 else 0
    dapi_area_raw = float(np.sum(masks > 0))

    return total_nuclei, dapi_area_raw, masks
    
def process_caspase_focused_offsets(img, offset=10):
    """
    Robust Caspase Segmentation Engine.
    Resistant to hot-pixels and massive diffuse signal patches.
    """
    # 1. SAFER BACKGROUND SUBTRACTION
    # Using a 201x201 kernel prevents massive caspase patches from being deleted
    kernel_size = (101, 101)  
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    img_background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_subtracted = cv2.subtract(img, img_background)
    
    img_float = img_subtracted.astype(np.float32)
    
    # 2. THE HOT-PIXEL FIX
    # Ignore the top 0.1% of glowing debris. Normalize based on the 99.9th percentile.
    p_max = np.percentile(img_float, 99.95)
    if p_max <= 0: 
        p_max = 1.0 # Prevent divide-by-zero on completely blank images
        
    # Clip anything above p_max, then scale nicely to 0-255
    img_float_clipped = np.clip(img_float, 0, p_max)
    img_scaled = ((img_float_clipped / p_max) * 255).astype(np.uint8)
    
    # 3. THRESHOLDING
    img_blurred = cv2.GaussianBlur(img_scaled, (5, 5), sigmaX=2)
    t_base = threshold_otsu(img_blurred)
    
    # 4. DYNAMIC OFFSET (Optional but safer)
    # Instead of a hard +10, we make sure we don't offset past the actual signal
    t_offset = min(t_base + offset, 250) 
    
    # Generate the final mask safely
    masks = (img_blurred > t_offset).astype(np.uint8) * 255
    
    return masks, None
    
def process_smooth_colony_outline_clahe(img):
    """
    Advanced Edge Preserving Engine: Generates tight colony boundaries 
    matching original structures without mask bloat or ghost artifacts.
    """
    # 1. Locally equalize image contrast to reveal dim colony edges uniformly
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))
    enhanced_img = clahe.apply(img)
    
    # 2. Smooth internal textures while firmly locking down boundary edges
    bilateral = cv2.bilateralFilter(enhanced_img, d=9, sigmaColor=75, sigmaSpace=75)
    binary = apply_threshold(bilateral, method="triangle")
    
    # 3. Use an optimized structuring element to close minor gaps without expanding boundaries
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    processed_mask = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    
    # 4. Extract external contours to isolate the main colony structure and drop noise artifacts
    contours, _ = cv2.findContours(processed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_mask = np.zeros_like(processed_mask)
    
    if contours:
        # Retain exclusively the single largest continuous mass by area
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
    else:
        clean_mask = processed_mask

    # 5. Apply a soft smoothing pass to eliminate jagged, pixelated borders
    clean_mask = cv2.GaussianBlur(clean_mask, (7, 7), 0)
    _, clean_mask = cv2.threshold(clean_mask, 127, 255, cv2.THRESH_BINARY)
    
    return clean_mask
    
def process_smooth_colony_outline_stdev(img):
    """Equalized Texture + H-Maxima Pipeline."""
    # 1. OUTLIER REJECTION (Fiji Contrast)
    img_enhanced = apply_fiji_auto_contrast(img, saturate_percent=0.35)
    img_f = img_enhanced.astype(np.float32)
    
    # 2. LOCAL STANDARD DEVIATION (The Texture Map)
    k_size = 15
    mu = cv2.blur(img_f, (k_size, k_size))
    mu_sq = cv2.blur(img_f**2, (k_size, k_size))
    variance = mu_sq - (mu**2)
    variance[variance < 0] = 0
    texture_map = np.sqrt(variance)
    texture_norm = cv2.normalize(texture_map, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    # 3. BLUR & THRESHOLD
    texture_blurred = cv2.GaussianBlur(texture_norm, (15, 15), 0)
    t_val = threshold_triangle(texture_blurred)
    binary = (texture_blurred > t_val).astype(np.uint8) * 255
    
    if np.max(binary) == 0: 
        return np.zeros_like(img)
        
    # 4. CLOSE & FILL (Creates the base blob)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
    main_blob = ndi.binary_fill_holes(closed > 0).astype(np.uint8) * 255
    
    # 5. H-MAXIMA NECK-CUTTER (Watershedding Spurious Noise)
    dist_map = ndi.distance_transform_edt(main_blob)
    dist_map_smoothed = cv2.GaussianBlur(dist_map, (0, 0), sigmaX=5)
    
    h = 10 
    seed_h = dist_map_smoothed - h
    rec_h = reconstruction(seed_h, dist_map_smoothed, method='dilation')
    local_max_mask = (dist_map_smoothed - rec_h) > 0
    
    markers, _ = ndi.label(local_max_mask)
    labels = watershed(-dist_map, markers, mask=main_blob)
    
    # 6. KEEP ONLY THE LARGEST STRUCTURE
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) > 1:
        largest_label = unique[np.argmax(counts[1:]) + 1]
        main_blob_sliced = (labels == largest_label).astype(np.uint8) * 255
    else:
        main_blob_sliced = main_blob

    # 7. FINAL POLISH
    final_mask = np.zeros_like(img)
    contours, _ = cv2.findContours(main_blob_sliced, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(final_mask, [largest], -1, 255, thickness=cv2.FILLED)
            
    return final_mask

def process_smooth_colony_outline_fastsam_old(img_gray):
    DEVICE = 'cpu'  # CPU is blazing fast for single-point prompts and avoids M3 bugs

    print("Loading FastSAM Model...")
    model = FastSAM('FastSAM-s.pt') 
    
    if img_gray is None: return None, None, None
    h, w = img_gray.shape
    
    # 1. Normalize for the AI
    p_low, p_high = np.percentile(img_gray, (0, 99.5))
    #img_norm = np.clip(img_gray, p_low, p_high)
    #img_8u = ((img_norm - p_low) / (p_high - p_low) * 255).astype(np.uint8)

    if p_high <= p_low: img_8u = img_gray.astype(np.uint8)
    else: img_8u = ((np.clip(img_gray, p_low, p_high) - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)
    
    img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)

    # 2. LOCATE THE TARGET
    blurred = cv2.GaussianBlur(img_8u, (99, 99), 0)
    _, rough_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    M = cv2.moments(rough_bin)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = w // 2, h // 2  

    # 3. RUN FASTSAM WITH POINT PROMPT
    results = model(img_rgb, device=DEVICE, retina_masks=True, points=[[cX, cY]], labels=[1], verbose=False)
    
    final_mask = np.zeros_like(img_gray)
    
    # 4. POST-PROCESSING SCISSORS
    if results and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        ai_mask = (mask_resized * 255).astype(np.uint8)
        
        # --- FIX 1: DELETE FLOATING ISLANDS ---
        # Find all blobs and keep only the absolute largest one
        contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return img_rgb, final_mask, (cX, cY)
            
        largest_contour = max(contours, key=cv2.contourArea)
        core_mask = np.zeros_like(img_gray)
        cv2.drawContours(core_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

        # --- FIX 2: SEVER THIN APPENDAGES ---
        # A Morphological 'Opening' breaks thin connections (necks).
        # A kernel of (35, 35) means any connection thinner than 35 pixels will be snapped.
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
        opened_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_open)
        
        # --- FIX 3: THROW AWAY THE SEVERED APPENDAGE ---
        # Now that the appendage is disconnected, we find the largest contour one last time
        contours_final, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_final:
            largest_clean = max(contours_final, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_clean], -1, 255, thickness=cv2.FILLED)

    return final_mask

def process_smooth_colony_outline_fastsam(img_gray):
    DEVICE = 'cpu'
    print("Loading FastSAM Model...")
    model = FastSAM('FastSAM-s.pt') 
    if img_gray is None: return None, None, None
    h, w = img_gray.shape
    
    # 1. PRE-PROCESSING
    img_smooth = cv2.bilateralFilter(img_gray, d=5, sigmaColor=25, sigmaSpace=25)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16,16))
    img_enhanced = clahe.apply(img_smooth)
    img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_GRAY2RGB)

    # 2. LOCATE THE TARGET (Using original gray for stable Otsu)
    blurred = cv2.GaussianBlur(img_gray, (99, 99), 0)
    _, rough_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    M = cv2.moments(rough_bin)
    cX, cY = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])) if M["m00"] != 0 else (w // 2, h // 2)

    # 3. FASTSAM
    results = model(img_rgb, device=DEVICE, retina_masks=True, points=[[cX, cY]], labels=[1], verbose=False)
    
    final_mask = np.zeros_like(img_gray)
    
    # 4. UNIFIED POST-PROCESSING
    if results and results[0].masks is not None and len(results[0].masks.data) > 0:
        mask = results[0].masks.data[0].cpu().numpy()
        ai_mask = (cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
        
        # A) Isolate the largest connected component (Remove noise artifacts)
        contours, _ = cv2.findContours(ai_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            clean_mask = np.zeros_like(img_gray)
            cv2.drawContours(clean_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # B) Watershed to pinch off appendages
            dist_transform = cv2.distanceTransform(clean_mask, cv2.DIST_L2, 5)
            # Tuning Knob: Increase 0.15 to 0.25 if appendages persist
            _, sure_fg = cv2.threshold(dist_transform, 0.25 * dist_transform.max(), 255, 0)
            
            sure_bg = cv2.dilate(clean_mask, np.ones((3,3), np.uint8), iterations=3)
            unknown = cv2.subtract(sure_bg, np.uint8(sure_fg))
            
            _, markers = cv2.connectedComponents(np.uint8(sure_fg))
            markers += 1
            markers[unknown == 255] = 0
            
            mask_3c = cv2.cvtColor(clean_mask, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(mask_3c, markers)
            
            # C) Extract the winning colony (largest label > 1)
            unique_labels = np.unique(markers)
            max_area = 0
            best_label = -1
            for label_id in unique_labels:
                if label_id > 1:
                    area = np.sum(markers == label_id)
                    if area > max_area:
                        max_area, best_label = area, label_id
            
            if best_label != -1:
                final_mask[markers == best_label] = 255
            else:
                final_mask = clean_mask
                
    return final_mask