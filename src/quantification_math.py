import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ks_2samp, chisquare
from skimage.measure import label, regionprops

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
    
def get_object_distances(mask, colony_cx, colony_cy, microns_per_pixel=0.645):
    # Label distinct objects
    labeled_mask = label(mask > 0)
    props = regionprops(labeled_mask)
    
    # Calculate distance for each object centroid
    distances = []
    for prop in props:
        obj_y, obj_x = prop.centroid
        dist_px = np.sqrt((obj_x - colony_cx)**2 + (obj_y - colony_cy)**2)
        distances.append(dist_px * microns_per_pixel)
        
    return distances

def run_statistical_analysis(df, x_variable, y_variable, subset_categories=None, custom_palette=None, title=None, save_filename="plot_output"):
    """
    Performs robust statistics and generates publication-quality, Prism-style Violin plots.
    """
    # 1. Clean data (Keep only > 0 valid rows and drop NaNs)
    analysis_df = df[df[y_variable] > 0].copy()
    analysis_df = analysis_df.dropna(subset=[x_variable, y_variable])
    
    # 2. Subset the data if requested
    if subset_categories is not None:
        analysis_df = analysis_df[analysis_df[x_variable].isin(subset_categories)]
        
    if analysis_df.empty:
        print("Error: No data available after filtering. Check your variable names and subsets.")
        return
        
    # We must lock the order of the groups so x-coordinates match the stats brackets
    groups = list(analysis_df[x_variable].unique())
    group_data = [analysis_df[analysis_df[x_variable] == g][y_variable] for g in groups]
    
    # 3. Print Summary Stats
    print(f"--- SUMMARY STATISTICS: {y_variable} by {x_variable} ---")
    summary = analysis_df.groupby(x_variable)[y_variable].agg(['count', 'mean', 'std', 'median'])
    print(summary)
    print("\n" + "-"*60)
    
    # 4. Statistical Testing Engine
    significant_comparisons = [] # Will hold tuples of (group1, group2, p_value)
    
    if len(groups) == 2:
        t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
        print(f"Two-Sample Welch's t-test ({groups[0]} vs {groups[1]}): p-value = {p_val:.6e}")
        if p_val < 0.05:
            significant_comparisons.append((groups[0], groups[1], p_val))
            
    elif len(groups) > 2:
        f_stat, p_val = stats.f_oneway(*group_data)
        print(f"One-Way ANOVA: p-value = {p_val:.6e}")
        
        if p_val < 0.05:
            print("  Result: SIGNIFICANT variance. Running Tukey HSD...")
            tukey = pairwise_tukeyhsd(endog=analysis_df[y_variable], 
                                      groups=analysis_df[x_variable], 
                                      alpha=0.05)
            print(tukey.summary())
            
            # Extract significant pairs for plotting
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            sig_pairs = tukey_df[tukey_df['reject'] == True]
            
            for _, row in sig_pairs.iterrows():
                # Extract the adjusted p-value (statsmodels usually names it 'p-adj')
                significant_comparisons.append((row['group1'], row['group2'], row['p-adj']))

    # 5. Prettify and Plot (GraphPad Prism Style)
    sns.set_theme(style="ticks", context="talk")
    palette_to_use = custom_palette if custom_palette else "Pastel1"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Draw Violins & Swarms (enforce 'order' so we know exact X coordinates)
    # Added cut=0 to prevent the violins from extending past min/max data points
    sns.violinplot(data=analysis_df, x=x_variable, y=y_variable, order=groups,
                   palette=palette_to_use, inner=None, linewidth=1.5, alpha=0.7, ax=ax, cut=0)
    sns.swarmplot(data=analysis_df, x=x_variable, y=y_variable, order=groups,
                  color="black", alpha=0.6, size=5, ax=ax)

    # ==========================================
    # 6. DRAW SIGNIFICANCE BRACKETS
    # ==========================================
    y_max = analysis_df[y_variable].max()
    y_range = y_max - analysis_df[y_variable].min()
    
    # Increased height to clear the violins safely
    bracket_y_base = y_max + (y_range * 0.15) 
    bracket_step = y_range * 0.15  # Increased step size so multiple brackets don't overlap
    tick_len = y_range * 0.02      # The small downward legs of the bracket
    
    group_x_map = {g: i for i, g in enumerate(groups)}

    for (g1, g2, p) in significant_comparisons:
        x1, x2 = sorted([group_x_map[g1], group_x_map[g2]])
        
        # Convert P-value to 3 decimal places
        if p < 0.001: 
            sig_text = "p < 0.001"
        else: 
            sig_text = f"p = {p:.3f}"
        
        # Draw the line and ticks
        ax.plot([x1, x1, x2, x2], 
                [bracket_y_base, bracket_y_base + tick_len, bracket_y_base + tick_len, bracket_y_base], 
                lw=1.5, color='black')
        
        # Put the text directly on top of the bracket (moved slightly higher to clear the line)
        ax.text((x1 + x2) * 0.5, bracket_y_base + tick_len + (y_range * 0.02), sig_text, 
                ha='center', va='bottom', color='black', fontsize=12)
        
        # Increment height for the next bracket so they don't overlap
        bracket_y_base += bracket_step

    # ==========================================
    # 7. AXIS & SPINES (PRISM-STYLE ALIGNMENT)
    # ==========================================
    plot_title = title if title else f"Distribution of {y_variable} by {x_variable}"
    ax.set_title(plot_title, pad=20, fontweight='bold')
    ax.set_xlabel(x_variable.replace("_", " "), fontweight='bold')
    ax.set_ylabel(y_variable.replace("_", " "), fontweight='bold')
    
    # Increased final_y_top to ensure the highest p-value text is never cropped
    final_y_top = bracket_y_base + (y_range * 0.15) if significant_comparisons else y_max * 1.2
    ax.set_ylim(bottom=0, top=final_y_top)
    ax.set_xlim(left=-0.5, right=len(groups) - 0.5)
    
    # Remove top and right borders
    sns.despine(ax=ax, top=True, right=True)
    
    # Force the X and Y axes to physically connect at the (0,0) origin
    ax.spines['left'].set_bounds(0, final_y_top)
    ax.spines['bottom'].set_bounds(-0.5, len(groups) - 0.5)
    
    # Make axis lines thicker like Prism
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.tick_params(width=1.5)

    plt.tight_layout()
    
    # ==========================================
    # 8. EXPORT TO ILLUSTRATOR (Vector Graphic)
    # ==========================================
    #plt.savefig(f"{save_filename}.svg", format='svg', transparent=True, bbox_inches='tight')
    #plt.savefig(f"{save_filename}.png", format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    plt.show()

def test_radial_uniformity(data_distances, colony_radius=510):
    """
    Tests if distances are uniformly distributed within a circular colony.
    Returns: chi2_statistic, p_value
    """
    # Create radial bins
    bins = np.linspace(0, colony_radius, 10)
    observed_counts, _ = np.histogram(data_distances, bins=bins)
    
    # Expected: Density is proportional to r in a circular area
    bin_centers = (bins[:-1] + bins[1:]) / 2
    expected_counts = (bin_centers / np.sum(bin_centers)) * len(data_distances)
    
    chi2, p_val = chisquare(f_obs=observed_counts, f_exp=expected_counts)
    return chi2, p_val

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats

def plot_spatial_ridgeline(df, condition_col, distance_col, control_group, subset_categories=None, custom_palette=None, title=None, save_filename="ridgeline_output"):
    """
    Generates a publication-ready Ridgeline plot for spatial distributions, 
    annotated with Kolmogorov-Smirnov (K-S) test statistics.
    
    Parameters:
    - df: The master Pandas dataframe
    - condition_col: The categorical column to split groups by (e.g., 'Treatment')
    - distance_col: The continuous spatial distance metric (e.g., 'Normalized_Distance')
    - control_group: The exact string name of the control condition to compare others against.
    - subset_categories: (Optional) Ordered list of categories to include.
    - custom_palette: (Optional) Dictionary mapping categories to specific colors.
    - title: (Optional) Custom title for the plot.
    """
    # 1. Clean data
    analysis_df = df.dropna(subset=[condition_col, distance_col]).copy()
    
    # 2. Subset and Order Categories
    if subset_categories is not None:
        analysis_df = analysis_df[analysis_df[condition_col].isin(subset_categories)]
        categories = subset_categories
    else:
        # If no subset provided, put control group first, then the rest
        unique_cats = list(analysis_df[condition_col].unique())
        if control_group in unique_cats:
            unique_cats.remove(control_group)
            categories = [control_group] + unique_cats
        else:
            categories = unique_cats
            
    # Lock the categorical order for the plot
    analysis_df[condition_col] = pd.Categorical(analysis_df[condition_col], categories=categories, ordered=True)
    
    if analysis_df.empty:
        print("Error: No data available after filtering.")
        return

    # 3. Statistical Testing (Kolmogorov-Smirnov Test)
    print(f"--- K-S DISTRIBUTION STATS (Reference: {control_group}) ---")
    control_data = analysis_df[analysis_df[condition_col] == control_group][distance_col]
    
    stats_results = {}
    for cat in categories:
        if cat == control_group:
            stats_results[cat] = "Reference"
            continue
            
        test_data = analysis_df[analysis_df[condition_col] == cat][distance_col]
        stat, p_val = stats.ks_2samp(control_data, test_data)
        
        print(f"{cat} vs {control_group}: K-S stat = {stat:.4f} | p-value = {p_val:.6e}")
        
        if p_val < 0.001:
            stats_results[cat] = "p < 0.001"
        else:
            stats_results[cat] = f"p = {p_val:.3f}"

    # 4. Set up the Ridgeline Plot Geometry
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Initialize the FacetGrid
    pal = custom_palette if custom_palette else "husl"
    g = sns.FacetGrid(analysis_df, row=condition_col, hue=condition_col, 
                      aspect=6, height=1.2, palette=pal)

    # Draw the density plots
    # fill=True adds the solid color, color="w" adds a white outline to separate overlapping ridges
    g.map(sns.kdeplot, distance_col, bw_adjust=0.5, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)
    g.map(sns.kdeplot, distance_col, clip_on=False, color="w", lw=2, bw_adjust=0.5)
    
    # Draw a solid horizontal line at the base of each density curve
    g.map(plt.axhline, y=0, linewidth=1.5, linestyle="-", color="black", clip_on=False)

    # 5. Custom Labeling and Statistical Annotation
    def label_and_stats(x, color, label):
        ax = plt.gca()
        # Left side: Condition Name
        ax.text(0, 0.2, label, fontweight="bold", color=color, 
                ha="left", va="center", transform=ax.transAxes, fontsize=14)
        
        # Right side: K-S p-value
        p_text = stats_results.get(label, "")
        text_color = "black" if p_text == "Reference" else color
        ax.text(1.0, 0.2, p_text, fontweight="bold", color=text_color,
                ha="right", va="center", transform=ax.transAxes, fontsize=12)

    g.map(label_and_stats, distance_col)

    # ==========================================
    # 6. Formatting and Presentation
    # ==========================================
    # Overlap the plots completely to create the "Ridge" effect
    g.figure.subplots_adjust(hspace=-0.4)

    # Remove subplot titles and y-axis ticks/labels
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    
    # Remove ALL spines first so everything floats by default
    g.despine(bottom=True, left=True)
    
    # Format the overarching X-axis label
    g.set_xlabels(distance_col.replace("_", " "), fontweight="bold", fontsize=14)
    
    # Target ONLY the very last (bottom-most) plot
    bottom_ax = g.axes.flat[-1]
    
    # Reactivate the bottom spine for ONLY the bottom plot
    bottom_ax.spines['bottom'].set_visible(True)
    bottom_ax.spines['bottom'].set_linewidth(1.5)
    bottom_ax.spines['bottom'].set_color('black')
    
    # Force physical tick marks on ONLY the bottom plot
    bottom_ax.tick_params(axis='x', bottom=True, length=6, width=1.5, labelsize=12, color='black')
    
    # Ensure all upper plots are completely clean
    for ax in g.axes.flat[:-1]:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    
    # Add an overarching title
    plot_title = title if title else f"Spatial Distribution of {distance_col}"
    g.figure.suptitle(plot_title, fontweight='bold', fontsize=16, y=0.98)

    # 7. Vector Export for Illustrator
    #plt.savefig(f"{save_filename}.svg", format='svg', transparent=True, bbox_inches='tight')
    #plt.savefig(f"{save_filename}.png", format='png', dpi=300, transparent=True, bbox_inches='tight')
    
    plt.show()