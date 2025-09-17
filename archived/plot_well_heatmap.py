#!/usr/bin/env python3
"""
Plot a heatmap of Young's modulus values on a 96-well plate.
Input: summary CSV with columns 'Well', 'ElasticModulus', and optionally 'R2'.
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches # for circles
import matplotlib.colors as mcolors # for colorbar
import numpy as np
import string # for rows and columns
import sys # for command line arguments

plt.rcParams['font.family'] = 'Arial'  # Set global font to Arial

# Settings
ROWS = list(string.ascii_uppercase[:8])  # A-H
COLS = list(range(1, 13))                # 1-12

# Helper: map well name to (row, col) index
well_to_idx = {(f"{row}{col}"): (i, j) for i, row in enumerate(ROWS) for j, col in enumerate(COLS)}


def plot_well_heatmap(summary_csv, value_col='ElasticModulus', cmap='viridis', annotate=True, save_path=None, convert_to_mpa=True):
    # Read data
    df = pd.read_csv(summary_csv)
    has_r2 = 'R2' in df.columns
    has_std = 'Std' in df.columns
    # Build 2D array for heatmap
    heatmap = np.full((8, 12), np.nan)
    r2map = np.full((8, 12), np.nan) if has_r2 else None
    stdmap = np.full((8, 12), np.nan) if has_std else None
    for _, row in df.iterrows():
        well = str(row['Well'])
        value = row[value_col]
        if well in well_to_idx and not isinstance(value, (pd.Series, np.ndarray)) and pd.notnull(value):
            i, j = well_to_idx[well]
            # Convert Pa to MPa if requested
            if convert_to_mpa and value_col == 'ElasticModulus':
                heatmap[i, j] = value / 1e6  # Convert Pa to MPa
            else:
                heatmap[i, j] = value
            if has_r2 and r2map is not None:
                r2val = row['R2']
                if not isinstance(r2val, (pd.Series, np.ndarray)) and pd.notnull(r2val):
                    r2map[i, j] = r2val
            if has_std and stdmap is not None:
                stdval = row['Std']
                if not isinstance(stdval, (pd.Series, np.ndarray)) and pd.notnull(stdval):
                    # Convert uncertainty to MPa as well
                    if convert_to_mpa and value_col == 'ElasticModulus':
                        stdmap[i, j] = stdval / 1e6  # Convert Pa to MPa
                    else:
                        stdmap[i, j] = stdval
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 7))
    norm = mcolors.Normalize(vmin=np.nanmin(heatmap), vmax=np.nanmax(heatmap))
    cmap = plt.get_cmap(cmap)
    
    # Draw circles for each well
    for i, row_label in enumerate(ROWS):
        for j, col_label in enumerate(COLS):
            x, y = j, 7 - i  # y reversed for A at top
            value = heatmap[i, j]
            color = cmap(norm(value)) if not np.isnan(value) else (0.9, 0.9, 0.9, 1)
            circle = mpatches.Circle((x, y), 0.4, color=color, ec='black', lw=1.0)
            ax.add_patch(circle)
            # Annotate value ± std and R2
            if annotate and not np.isnan(value):
                # Determine text color based on background color brightness
                # Convert color to RGB and calculate brightness
                rgb_color = cmap(norm(value))[:3]  # Get RGB values
                brightness = (0.299 * rgb_color[0] + 0.587 * rgb_color[1] + 0.114 * rgb_color[2])
                text_color = 'black' if brightness > 0.5 else 'white'
                
                # Main value (Young's modulus)
                ax.text(x, y+0.1, f"{value:.2f}", ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')
                
                # Standard deviation on second row
                if has_std and stdmap is not None and not np.isnan(stdmap[i, j]):
                    ax.text(x, y-0.05, f"±{stdmap[i, j]:.2f}", ha='center', va='center', fontsize=8, color=text_color, fontweight='bold')
                
                # R² value on third row
                if has_r2 and r2map is not None and not np.isnan(r2map[i, j]):
                    ax.text(x, y-0.2, f"R²={r2map[i, j]:.2f}", ha='center', va='center', fontsize=8, color=text_color)
    
    # Set axis
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect('equal')
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(c) for c in COLS])
    ax.set_yticks(range(8))
    ax.set_yticklabels(ROWS[::-1])  # A at top
    ax.tick_params(axis='both', which='major', labelsize=24)
    #ax.set_xlabel('Column', fontsize=18)
    #ax.set_ylabel('Row', fontsize=18)
    # Set title based on units
    if convert_to_mpa and value_col == 'ElasticModulus':
        title = "96-Well Plate Young's Modulus Heatmap (MPa)"
    else:
        title = f"96-Well Plate {value_col} Heatmap"
    ax.set_title(title, fontsize=20)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, pad=0.02)
    # Set colorbar label based on units
    if convert_to_mpa and value_col == 'ElasticModulus':
        cbar.set_label(f"{value_col} (MPa)", fontsize=18)
    else:
        cbar.set_label(f"{value_col} (Pa)", fontsize=18)
    cbar.ax.tick_params(labelsize=16)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot 96-well plate heatmap from summary CSV.")
    parser.add_argument('csv', help='Summary CSV file (with Well and Elastic Modulus columns)')
    parser.add_argument('--value-col', default='ElasticModulus', help='Column to plot as heatmap (default: Elastic Modulus)')
    parser.add_argument('--cmap', default='viridis', help='Matplotlib colormap (default: viridis)')
    parser.add_argument('--no-annotate', action='store_true', help='Do not annotate values in wells')
    parser.add_argument('--no-mpa', action='store_true', help='Keep values in Pa instead of converting to MPa')
    parser.add_argument('--save', default=None, help='Path to save the plot (if not given, show interactively)')
    args = parser.parse_args()
    
    plot_well_heatmap(
        args.csv,
        value_col=args.value_col,
        cmap=args.cmap,
        annotate=not args.no_annotate,
        save_path=args.save,
        convert_to_mpa=not args.no_mpa
    ) 