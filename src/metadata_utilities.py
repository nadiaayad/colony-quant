import os
import matplotlib.pyplot as plt
from aicsimageio import AICSImage

def mkdir_p(path):
    """Safely handles directory creation."""
    os.makedirs(path, exist_ok=True)

def determine_shape(filename):
    filename_upper = filename.upper()
    c_keys = [f"C{i}" for i in range(1, 13)]
    s_keys = [f"S{i}" for i in range(1, 13)]
    t_keys = [f"T{i}" for i in range(1, 13)]
    pm_keys = [f"PM{i}" for i in range(1, 13)] + ["PM"]
    keys = c_keys + s_keys + t_keys + pm_keys
    
    mapping = {}
    for k in c_keys: mapping[k] = "Circle"
    for k in s_keys: mapping[k] = "Square"
    for k in t_keys: mapping[k] = "Triangle"
    for k in pm_keys: mapping[k] = "PacMan"

    for key in reversed(keys):
        if key in filename_upper:
            return mapping[key]
    return "Not Found"

def determine_position(filename):
    keys = ["A1", "A2", "A3", "B1", "B2", "B3"]
    filename_upper = filename.upper()
    for key in reversed(keys):
        if key in filename_upper:
            return key
    return "Not Found"

def determine_treatment(filename):
    keys = ["ctrl", "rocki", "smifh2", "blebb", "cytd", "jasplak", "zvad", "qvd",
            "unc", "maggel", "stretched", "pacman", "cd47-20ugml", "cd47_20ugml",
            "shluc", "shdock1","shmertk", "shcd24", "shcd47", "ps", "pc", "fs"]
    mapping = {
        "ctrl": "Ctrl", "rocki": "ROCKi", "smifh2": "SMIFH2", "blebb": "Blebb", 
        "cytd": "Cytd", "jasplak": "Jasplak", "zvad": "ZVAD", "qvd": "QVD", 
        "unc": "UNC2541", "maggel": "MagGel", "stretched": "Stretched", "pm":"PacMan",
        "cd47-20ugml":"CD47 20 ugml", "cd47_20ugml":"CD47 20 ugml",
        "shluc":"shLuc", "shdock1":"shDOCK1","shmertk":"shMERTK", "shcd24":"shCD24", "shcd47":"shCD47",
        "ps":"PSbeads", "pc":"PCbeads", "fs":"FSbeads"
    }
    filename_lower = filename.lower()
    if "maggel" in filename_lower:
        return mapping["maggel"]
    for key in keys:
        if key in filename_lower:
            return mapping[key]
    return "Not Found"

def determine_cell_line(folder, subfolder, filename):
    combined_context = (folder + "_" + subfolder + "_" + filename).lower()
    if "redt" in combined_context: return "red T-mNeongreen"
    elif "-t" in combined_context: return "T-mNeonGreen"
    elif "myh10" in combined_context: return "MYH10-eGFP"
    elif "shluc" in combined_context: return "shLuciferase"
    elif "shmertk" in combined_context: return "shMERTK"
    elif "shcd47" in combined_context: return "shCD47"
    elif "shcd24" in combined_context: return "shCD24"
    elif "shdock1" in combined_context: return "shDOCK1"
    return "Not Found"

def determine_live_or_time(filename, subfolder):
    combined_context = f"{subfolder}_{filename}".lower()
    if "time" in combined_context: return "Timelapse"
    elif "0h" in combined_context: return "Timepoint 0h"
    elif "48h" in combined_context: return "Timepoint 48h"
    elif "12h" in combined_context: return "Timepoint 12h - No Differentiation" 
    return "Not Found"

def handle_ch_naming_ui(image_path, subfolder_name):
    """Renders visual panel maps internally within cells for quick check parameters."""
    try:
        img_obj = AICSImage(image_path)
        num_channels = len(img_obj.channel_names) if img_obj.channel_names else (img_obj.dims.C if "C" in img_obj.dims.order else 1)
        if num_channels > 1:
            print(f"\n➔ Multi-Channel Data Detected for [{subfolder_name}] ({num_channels} channels found).")
            fig, axes = plt.subplots(1, num_channels, figsize=(4 * num_channels, 4), squeeze=False)
            for i in range(num_channels):
                plane_data = img_obj.get_image_data("YX", C=i, T=0, Z=0)
                ax = axes[0, i]
                ax.imshow(plane_data, cmap='gray')
                ax.set_title(f"Channel {i+1}", fontsize=12, fontweight='bold')
                ax.axis('off')
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)
            
            channel_mappings = {}
            for i in range(num_channels):
                name = input(f"   Enter label for Channel {i+1}: ").strip()
                channel_mappings[f"Channel {i+1} Name"] = name if name else f"Channel_{i+1}"
            plt.close(fig)
            return channel_mappings
    except Exception as e:
        print(f"  [Warning] Preview engine bypassed for {os.path.basename(image_path)}. Reason: {e}")
    return {}
