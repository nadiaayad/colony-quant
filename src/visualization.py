import cv2
import numpy as np
import matplotlib.pyplot as plt

def apply_display_contrast(img, saturate_percent=1.0):
    """Quick contrast enhancement strictly for visual rendering."""
    if img is None:
        return np.zeros((10, 10), dtype=np.uint8)
    p_high = np.percentile(img, 100.0 - saturate_percent)
    p_low = np.percentile(img, 0.1)
    if p_high <= p_low: return img.astype(np.uint8)
    return ((np.clip(img, p_low, p_high) - p_low) / (p_high - p_low) * 255.0).astype(np.uint8)

def plot_diagnostic_grid(img_name, raw_outline, raw_quant, raw_nuc, mask_outline, mask_quant, mask_nuc):
    """
    Plots a highly-compressed 2x3 grid of the raw and masked images.
    Compression prevents Jupyter Notebook from crashing when logging hundreds of files.
    """
    # Downsample to 20% of original size for lightning-fast rendering
    scale = 0.20 
    
    def fast_resize(img):
        if img is None: 
            return np.zeros((10, 10), dtype=np.uint8)
        h, w = img.shape[:2]
        return cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_NEAREST)

    # Set up the 2x3 grid
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    fig.suptitle(f"QC Dashboard: {img_name}", fontsize=6, fontweight='bold')
    
    # --- ROW 1: Raw Images with Auto-Contrast ---
    axes[0, 0].imshow(apply_display_contrast(fast_resize(raw_outline)), cmap='gray')
    axes[0, 0].set_title("Raw Outline (Enhanced)")
    
    axes[0, 1].imshow(apply_display_contrast(fast_resize(raw_quant)), cmap='gray')
    axes[0, 1].set_title("Raw Quant (Enhanced)")
    
    axes[0, 2].imshow(apply_display_contrast(fast_resize(raw_nuc)), cmap='gray')
    axes[0, 2].set_title("Raw Nuclear (Enhanced)")

    # --- ROW 2: Thresholded/Masked Results ---
    axes[1, 0].imshow(fast_resize(mask_outline), cmap='gray')
    axes[1, 0].set_title("Outline Mask")
    
    axes[1, 1].imshow(fast_resize(mask_quant), cmap='gray')
    axes[1, 1].set_title("Quant Masked")
    
    axes[1, 2].imshow(fast_resize(mask_nuc), cmap='gray')
    axes[1, 2].set_title("Nuclear Masked")

    # Clean up formatting
    for ax in axes.flatten():
        ax.axis('off')

    plt.tight_layout()
    plt.show()