import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ks_2samp, chisquare
from skimage.measure import label, regionprops
from scipy.spatial.distance import pdist


# =========================================================
# GEOMETRY & DISTANCE MATHEMATICS
# =========================================================

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
    
def compute_pixel_radial_distances(quant_mask, colony_cx, colony_cy, microns_per_pixel=0.645):
    """Calculates physical distances from the centroid and returns summary stats + raw list."""
    y_indices, x_indices = np.where(quant_mask > 0)
    
    if len(x_indices) == 0:
        return 0.0, 0.0, 0.0, []
        
    distances_px = np.sqrt((x_indices - colony_cx) ** 2 + (y_indices - colony_cy) ** 2)
    distances_microns = distances_px * microns_per_pixel
    
    mean_rad = float(np.mean(distances_microns))
    median_rad = float(np.median(distances_microns))
    std_rad = float(np.std(distances_microns))
    
    return mean_rad, median_rad, std_rad, distances_microns.tolist()
    
def compute_object_radial_distances(mask, colony_cx, colony_cy, microns_per_pixel=0.645):
    """Labels distinct objects and calculates distance stats + raw list."""
    labeled_mask = label(mask > 0)
    props = regionprops(labeled_mask)
    
    distances = []
    for prop in props:
        obj_y, obj_x = prop.centroid
        dist_px = np.sqrt((obj_x - colony_cx)**2 + (obj_y - colony_cy)**2)
        distances.append(dist_px * microns_per_pixel)
        
    if len(distances) == 0:
        return 0.0, 0.0, 0.0, []

    mean_rad = float(np.mean(distances))
    median_rad = float(np.median(distances))
    std_rad = float(np.std(distances))
 
    return mean_rad, median_rad, std_rad, distances

def test_radial_uniformity(data_distances, colony_radius=510):
    """
    Tests radial uniformity and calculates Cramer's V effect size.
    """
    # Create bins
    bins = np.linspace(0, colony_radius, 10)
    observed, _ = np.histogram(data_distances, bins=bins)
    
    # Calculate Expected
    bin_centers = (bins[:-1] + bins[1:]) / 2
    raw_expected = bin_centers
    expected = (raw_expected / np.sum(raw_expected)) * np.sum(observed)
    expected = np.maximum(expected, 1e-6)
    
    # 1. Chi-Square Test
    chi2, p_val = chisquare(f_obs=observed, f_exp=expected)
    
    # 2. Cramer's V Calculation
    # V = sqrt(chi2 / (N * (min(k, r) - 1)))
    # Since this is a 1-row table (1x10), we compare observed vs expected (2 rows)
    n = np.sum(observed)
    min_dim = min(2, 10) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim))
    
    return chi2, p_val, cramers_v
    
def plot_spatial_cdf(data_distances, colony_radius=510, title="Spatial Distribution CDF"):
    """
    Plots the empirical CDF of observed distances against the theoretical 
    CDF for a uniform distribution in a circle: F(r) = (r/R)^2.
    """
    # 1. Prepare Observed Data (Empirical CDF)
    sorted_data = np.sort(np.array(data_distances))
    # Filter data to be within radius for the plot
    sorted_data = sorted_data[sorted_data <= colony_radius]
    n_filtered = len(sorted_data)
    ecdf_y = np.arange(1, n_filtered + 1) / n_filtered
    
    # 2. Prepare Theoretical Uniform CDF
    r_theoretical = np.linspace(0, colony_radius, 100)
    cdf_theoretical = (r_theoretical / colony_radius) ** 2
    
    # 3. Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(r_theoretical, cdf_theoretical, 'k--', label='Theoretical Uniform', linewidth=2)
    plt.step(sorted_data, ecdf_y, where='post', label='Observed Data', linewidth=2)
    
    plt.xlabel('Distance from Centroid ($\mu m$)')
    plt.ylabel('Cumulative Fraction')
    plt.title(title)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Interpretation hints
    plt.text(0.05, 0.9, 'Curve above theory = Central Clustering', transform=plt.gca().transAxes)
    plt.text(0.05, 0.85, 'Curve below theory = Peripheral Clustering', transform=plt.gca().transAxes)
    
    plt.tight_layout()
    plt.show()
# =========================================================
# STATISTICAL VISUALIZATION ENGINES
# =========================================================

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
    #plt.savefig(f"{save_filename}.png", format='png', dpi=300

def bootstrap_ks_test_with_stats(dist_control, dist_treatment, n_samples=300, iterations=500):
    """
    Returns median p-value, median signed K-S statistic, 
    and median absolute K-S statistic.
    The K-S statistic (D) is reported as signed (D signed=Dabsxsign(Med treatment−Med control)). 
    Positive values indicate an outward spatial shift toward the colony periphery, 
    while negative values indicate an inward spatial shift toward the colony center
    """
    p_values = []
    ks_stats_signed = []
    ks_stats_abs = []
    
    n_c = min(n_samples, len(dist_control))
    n_t = min(n_samples, len(dist_treatment))
    
    for _ in range(iterations):
        c_sub = np.random.choice(dist_control, n_c, replace=False)
        t_sub = np.random.choice(dist_treatment, n_t, replace=False)
        
        # Calculate signed difference
        # We need to sort and interpolate to find the max difference
        # Scipy's ks_2samp does this internally
        stat, p = ks_2samp(c_sub, t_sub)
        
        # New Convention: (Treatment - Control)
        # If Treatment > Control, result is positive (shift to periphery)
        # If Treatment < Control, result is negative (shift to center)
        diff = np.median(t_sub) - np.median(c_sub)
        direction = 1 if diff > 0 else -1
        
        p_values.append(p)
        ks_stats_signed.append(stat * direction) # Directional
        ks_stats_abs.append(stat)                # Absolute magnitude
    
    return np.median(p_values), np.median(ks_stats_signed), np.median(ks_stats_abs)


def plot_spatial_ridgeline_bs(df, condition_col, distance_col, control_group, 
                           subset_categories=None, custom_palette=None, 
                           title=None, bootstrap=True, n_samples=300, iterations=500):
    """
    Generates a publication-ready Ridgeline plot for spatial distributions.
    Uses bootstrapping to ensure p-values are robust to large sample sizes.
    """
    # 1. Prepare Data
    analysis_df = df.dropna(subset=[condition_col, distance_col]).copy()
    
    # 2. Subset and Order
    if subset_categories is not None:
        analysis_df = analysis_df[analysis_df[condition_col].isin(subset_categories)]
        categories = subset_categories
    else:
        unique_cats = list(analysis_df[condition_col].unique())
        if control_group in unique_cats:
            unique_cats.remove(control_group)
            categories = [control_group] + unique_cats
        else:
            categories = unique_cats
    
    analysis_df[condition_col] = pd.Categorical(analysis_df[condition_col], categories=categories, ordered=True)
    
    # 3. Statistical Testing
    stats_results = {}
    control_data = analysis_df[analysis_df[condition_col] == control_group][distance_col]
    
    print(f"--- ROBUST BOOTSTRAP STATS (Reference: {control_group}) ---")
    for cat in categories:
        if cat == control_group:
            stats_results[cat] = "Reference"
        else:
            test_data = analysis_df[analysis_df[condition_col] == cat][distance_col]
            
            if bootstrap:
                # Helper function modified to return both p-value AND k-s stat
                p_val, ks_stat, ks_stat_abs = bootstrap_ks_test_with_stats(control_data, test_data, n_samples=n_samples, iterations=iterations)
                label_text = f"p{n_samples} = "
            else:
                ks_stat, p_val = ks_2samp(control_data, test_data)
                label_text = "p = "
                
            print(f"{cat} vs {control_group}: K-S stat = {ks_stat:.4f} | {label_text}{p_val:.6e}")
            
            # Formatted for the plot label
            p_str = f"{p_val:.3f}" if p_val >= 0.001 else "< 0.001"
            stats_results[cat] = f"K-S={ks_stat:.2f}\n{label_text}{p_str}"

    # 4. Ridgeline Plot
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    g = sns.FacetGrid(analysis_df, row=condition_col, hue=condition_col, 
                      aspect=6, height=1.2, palette=custom_palette or "husl")

    g.map(sns.kdeplot, distance_col, bw_adjust=0.5, clip_on=False, fill=True, alpha=0.8, linewidth=1.5)
    g.map(plt.axhline, y=0, linewidth=1.5, linestyle="-", color="black", clip_on=False)

    # 5. Labeling Function 
    def label_and_stats(x, color, label):
        ax = plt.gca()
        cat_name = str(label)
        
        # Position: Left-aligned, slightly below the label
        # Condition Name at y=0.35, Stats at y=0.15
        
        # Draw condition name
        ax.text(0.01, 0.35, cat_name, fontweight="bold", color=color, 
                ha="left", va="center", transform=ax.transAxes, fontsize=14)
        
        # Draw stats below the name, smaller font
        p_text = stats_results.get(cat_name, "")
        #text_color = "black" if p_text == "Reference" else color
        text_color = color
        ax.text(0.01, 0.15, p_text, fontweight="bold", color=text_color,
                ha="left", va="center", transform=ax.transAxes, fontsize=10)

    g.map(label_and_stats, distance_col)
    
    # 6. Formatting
    g.figure.subplots_adjust(hspace=-0.4)
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)
    g.set_xlabels(distance_col.replace("_", " "), fontweight="bold", fontsize=14)
    
    # Fix the bottom axis line
    bottom_ax = g.axes.flat[-1]
    bottom_ax.spines['bottom'].set_visible(True)
    bottom_ax.spines['bottom'].set_linewidth(1.5)
    bottom_ax.spines['bottom'].set_color('black')
    bottom_ax.tick_params(axis='x', bottom=True, length=6, width=1.5, labelsize=12, color='black')
    
    for ax in g.axes.flat[:-1]:
        ax.tick_params(axis='x', bottom=False, labelbottom=False)
    
    plt.show()

    # 7. Vector Export for Illustrator
    #plt.savefig(f"{save_filename}.svg", format='svg', transparent=True, bbox_inches='tight')
    #plt.savefig(f"{save_filename}.png", format='png', dpi=300

def compute_radial_distribution_function(coords, colony_area, max_distance=100, num_bins=50):
    """
    Computes the spatial Radial Distribution Function g(r) for a set of points.
    
    Parameters:
    - coords: List or array of (x, y) coordinates for each cell/object.
    - colony_area: The total area of the colony mask (in square units).
    - max_distance: The maximum object-to-object distance to analyze.
    - num_bins: Number of distance bins.
    
    Returns:
    - bin_centers: The distance values (r)
    - g_r: The radial distribution function values at each r.
    """
    coords = np.array(coords)
    N = len(coords)
    
    if N < 2:
        return np.array([]), np.array([])

    # 1. Calculate all pairwise distances between every cell
    pairwise_distances = pdist(coords)
    
    # 2. Create distance bins
    bins = np.linspace(0, max_distance, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    
    # 3. Count how many pairs fall into each distance bin
    counts, _ = np.histogram(pairwise_distances, bins=bins)
    
    # 4. Calculate expected pairs for a completely random (uniform) distribution
    # Area of the annulus (ring) for each bin: pi * (r_outer^2 - r_inner^2)
    annulus_areas = np.pi * (bins[1:]**2 - bins[:-1]**2)
    
    # Global density of cells in the colony
    global_density = N / colony_area
    
    # Expected number of pairs = (Total Pairs) * (Annulus Area / Total Area)
    # Total possible pairs is N*(N-1)/2
    total_pairs = (N * (N - 1)) / 2
    expected_pairs = total_pairs * (annulus_areas / colony_area)
    
    # 5. g(r) is the ratio of Observed vs Expected
    # Add a tiny number to avoid division by zero
    g_r = counts / (expected_pairs + 1e-10)
    
    return bin_centers, g_r

def plot_local_clustering_rdf(df, group_col, coords_col, area_col='Colony_Area_um2', 
                              max_dist=50, num_bins=30, title=None, custom_palette=None, save_filename=None):
    """
    Generates a publication-ready plot of the Radial Distribution Function g(r).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Filter out rows missing coordinate data
    analysis_df = df.dropna(subset=[coords_col, area_col]).copy()
    groups = analysis_df[group_col].unique()
    
    sns.set_theme(style="ticks", context="talk")
    palette_to_use = custom_palette if custom_palette else "Set1"
    
    plt.figure(figsize=(9, 6))
    
    # For mapping colors consistently
    colors = sns.color_palette(palette_to_use, n_colors=len(groups))
    
    for idx, group in enumerate(groups):
        group_data = analysis_df[analysis_df[group_col] == group]
        all_g_r = []
        
        for _, row in group_data.iterrows():
            coords = row[coords_col]
            area = row[area_col]
            
            # Only process images that actually have enough cells to measure interactions
            if isinstance(coords, list) and len(coords) >= 5:
                r, g_r = compute_radial_distribution_function(coords, area, max_distance=max_dist, num_bins=num_bins)
                if len(g_r) > 0:
                    all_g_r.append(g_r)
        
        if all_g_r:
            avg_g_r = np.mean(all_g_r, axis=0)
            # Optional: Calculate standard error for confidence intervals
            std_err = np.std(all_g_r, axis=0) / np.sqrt(len(all_g_r))
            
            plt.plot(r, avg_g_r, label=group, color=colors[idx], linewidth=2.5)
            plt.fill_between(r, avg_g_r - std_err, avg_g_r + std_err, color=colors[idx], alpha=0.2)

    # Add the Random Distribution Baseline
    plt.axhline(1.0, color='black', linestyle='--', linewidth=2, label='Random (Uniform)')
    
    plot_title = title if title else f"Local Spatial Clustering: {coords_col.split('_')[0]}"
    plt.title(plot_title, pad=20, fontweight='bold')
    plt.xlabel("Distance between objects ($\mu m$)", fontweight='bold')
    plt.ylabel("Radial Distribution Function $g(r)$", fontweight='bold')
    
    plt.legend(frameon=False)
    sns.despine(trim=True, offset=5)
    plt.tight_layout()
    
    if save_filename:
        plt.savefig(f"{save_filename}.svg", format='svg', dpi=300)
    
    plt.show()