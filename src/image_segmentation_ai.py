import cv2
import numpy as np
from skimage.filters import threshold_otsu, threshold_triangle, threshold_li
from skimage.segmentation import watershed
from skimage.morphology import reconstruction
import scipy.ndimage as ndi
from ultralytics import FastSAM

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

def segment_nuclei(img_dapi, model):
    """DAPI Segmentation using Cellpose."""
    try:
        masks, flows, styles = model.eval(img_dapi, diameter=None, flow_threshold=0.4, cellprob_threshold=0.0)
        total_nuclei = int(np.max(masks))
        nuclear_pixels = float(np.sum(masks > 0))
        return total_nuclei, nuclear_pixels, masks
    except Exception as e:
        print(f"  [Cellpose Fallback] Processing via watershed morphology: {e}")
        thresh = apply_threshold(cv2.GaussianBlur(img_dapi, (5, 5), 2), method="otsu")
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)
        return int(num_labels - 1), float(np.sum(thresh > 0)), labels

def process_caspase_focused_offsets(img, offset = 15):
    """Focused Additive Offset Triangle Engine."""
    kernel_size = (101, 101)  
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
    img_background = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img_subtracted = cv2.subtract(img, img_background)
    
    img_float = img_subtracted.astype(np.float32)
    max_val = np.max(img_float) if np.max(img_float) > 0 else 1.0
    img_scaled = ((img_float / max_val) * 255).astype(np.uint8)
    
    img_blurred = cv2.GaussianBlur(img_scaled, (0, 0), sigmaX=2)
    t_base = threshold_triangle(img_blurred)

    t_offset = min(t_base + offset, 254)
    
    masks = (img_blurred > t_offset).astype(np.uint8) * 255
    
    return img_scaled, masks
    
def process_smooth_colony_outline_clahe(img):
    """
    Advanced Edge Preserving Engine: Generates tight colony boundaries 
    matching original structures without mask bloat or ghost artifacts.
    """
    # 1. Locally equalize image contrast to reveal dim colony edges uniformly
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
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

def process_smooth_colony_outline_fastsam(img_gray):
    DEVICE = 'cpu'  # CPU is blazing fast for single-point prompts and avoids M3 bugs

    print("Loading FastSAM Model...")
    model = FastSAM('FastSAM-s.pt') 
    
    if img_gray is None: return None, None, None
    h, w = img_gray.shape
    
    # 1. Normalize for the AI
    p_low, p_high = np.percentile(img_gray, (0, 99))
    img_norm = np.clip(img_gray, p_low, p_high)
    img_8u = ((img_norm - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    img_rgb = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2RGB)

    # 2. LOCATE THE TARGET
    blurred = cv2.GaussianBlur(img_8u, (99, 99), 0)
    _, rough_bin = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
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
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (95, 95))
        opened_mask = cv2.morphologyEx(core_mask, cv2.MORPH_OPEN, kernel_open)
        
        # --- FIX 3: THROW AWAY THE SEVERED APPENDAGE ---
        # Now that the appendage is disconnected, we find the largest contour one last time
        contours_final, _ = cv2.findContours(opened_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours_final:
            largest_clean = max(contours_final, key=cv2.contourArea)
            cv2.drawContours(final_mask, [largest_clean], -1, 255, thickness=cv2.FILLED)

    return final_mask