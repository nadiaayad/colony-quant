import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd

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

def run_statistical_analysis(df, x_variable, y_variable, subset_categories=None, custom_palette=None, title=None):
    """
    Performs robust statistics and generates publication-quality Violin plots.
    
    Parameters:
    - df: The master Pandas dataframe
    - x_variable: The categorical column to split groups by (e.g., 'Treatment')
    - y_variable: The continuous data to measure (e.g., 'Pct_CleavCasp3_per_MYH10_Area')
    - subset_categories: (Optional) A list of specific categories to include.
    - custom_palette: (Optional) A dictionary mapping categories to specific colors.
    - title: (Optional) Custom title for the plot.
    """
    # 1. Clean data (Keep only > 0 valid rows and drop NaNs)
    analysis_df = df[df[y_variable] > 0].copy()
    analysis_df = analysis_df.dropna(subset=[x_variable, y_variable])
    
    # 2. Subset the data if requested
    if subset_categories is not None:
        analysis_df = analysis_df[analysis_df[x_variable].isin(subset_categories)]
        
    # Ensure we actually have data left
    if analysis_df.empty:
        print("Error: No data available after filtering. Check your variable names and subsets.")
        return
        
    groups = analysis_df[x_variable].unique()
    group_data = [analysis_df[analysis_df[x_variable] == g][y_variable] for g in groups]
    
    # 3. Print Summary Stats
    print(f"--- SUMMARY STATISTICS: {y_variable} by {x_variable} ---")
    summary = analysis_df.groupby(x_variable)[y_variable].agg(['count', 'mean', 'std', 'median'])
    print(summary)
    print("\n" + "-"*60)
    
    # 4. Statistical Testing Engine
    sig_text = ""
    
    if len(groups) == 2:
        # Two-Sample T-Test (Welch's)
        t_stat, p_val = stats.ttest_ind(group_data[0], group_data[1], equal_var=False)
        print(f"Two-Sample Welch's t-test ({groups[0]} vs {groups[1]}):")
        print(f"  t-statistic: {t_stat:.4f} | p-value: {p_val:.6e}")
        
        if p_val < 0.05:
            print("  Result: SIGNIFICANT difference between the two groups.")
            sig_text = f"Significant! (t-test p = {p_val:.4f})"
        else:
            print("  Result: No significant difference.")
            sig_text = f"Not Significant (t-test p = {p_val:.4f})"
            
    elif len(groups) > 2:
        # One-Way ANOVA
        f_stat, p_val = stats.f_oneway(*group_data)
        print(f"One-Way ANOVA (Across {len(groups)} conditions):")
        print(f"  F-statistic: {f_stat:.4f} | p-value: {p_val:.6e}")
        
        if p_val < 0.05:
            print("  Result: SIGNIFICANT variance across groups. Running Tukey HSD Post-Hoc...")
            sig_text = f"ANOVA Significant (p = {p_val:.4e})"
            
            # Run Tukey's HSD Post-Hoc Test
            tukey = pairwise_tukeyhsd(endog=analysis_df[y_variable], 
                                      groups=analysis_df[x_variable], 
                                      alpha=0.05)
            print("\n--- TUKEY HSD POST-HOC RESULTS ---")
            print(tukey.summary())
            
            # Extract significant pairs to put on the plot
            tukey_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            sig_pairs = tukey_df[tukey_df['reject'] == True]
            
            if not sig_pairs.empty:
                sig_text += "\nSig. Pairs (Tukey): " + ", ".join(
                    [f"{row['group1']} vs {row['group2']}" for _, row in sig_pairs.iterrows()]
                )
        else:
            print("  Result: No significant variance across groups.")
            sig_text = f"ANOVA Not Sig. (p = {p_val:.4f})"

    # 5. Prettify and Plot (Violin + Swarm)
    sns.set_theme(style="ticks", context="talk")
    
    # If no custom palette is provided, default to seaborn's Pastel1
    palette_to_use = custom_palette if custom_palette else "Pastel1"
    
    plt.figure(figsize=(10, 6))
    
    # Draw the Violin Plot
    # inner=None removes the miniature inner boxplots so it doesn't clash with the swarmplot
    ax = sns.violinplot(data=analysis_df, x=x_variable, y=y_variable, 
                        palette=palette_to_use, inner=None, linewidth=1.5, alpha=0.7)
    
    # Overlay the Swarmplot (Draws the actual individual data points over the violins)
    sns.swarmplot(data=analysis_df, x=x_variable, y=y_variable, 
                  color="black", alpha=0.6, size=5)

    # Styling and Annotations
    plot_title = title if title else f"Distribution of {y_variable} by {x_variable}"
    plt.title(plot_title, pad=20, fontweight='bold')
    plt.xlabel(x_variable.replace("_", " "), fontweight='bold')
    plt.ylabel(y_variable.replace("_", " "), fontweight='bold')
    
    sns.despine(trim=True, offset=5)
    
    # Add the Statistical Significance Text Box
    plt.text(0.95, 0.95, sig_text, transform=ax.transAxes, 
             fontsize=11, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='gray'))
    
    plt.tight_layout()
    plt.show()