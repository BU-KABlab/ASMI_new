"""
ASMI Plotting Functions - Visualization for Indentation Analysis
Contains all plotting functions for data visualization and analysis results

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from .version import __version__
from typing import List, Optional, Dict
from dataclasses import dataclass
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import string

@dataclass
class AnalysisResult:
    well: str
    elastic_modulus: float
    uncertainty: float
    poisson_ratio: float
    sample_height: float
    fit_quality: float
    depth_range: tuple
    fit_A: float
    fit_d0: float
    adjusted_forces: list
    depth_in_range: list
    material_type: str
    contact_z: float
    contact_force: float

class ASMIPlotter:
    """Handles all plotting functions for ASMI analysis"""
    
    def __init__(self):
        # self.FORCE_THRESHOLD = 2.0  # N - force threshold to detect contact
        pass
    
    def plot_raw_data_all_wells(self, run_folder: str, save_plot: bool = True):
        """Plot raw data (absolute values) for all wells in a single plot.

        Labels include direction suffixes (_down/_up) when present.
        """
        data_dir = "results/measurements"
        run_path = os.path.join(data_dir, run_folder)
        
        if not os.path.exists(run_path):
            print(f"❌ Run folder {run_path} not found")
            return
        
        # Find all well data files in this run (include split _down/_up files)
        well_files = [f for f in os.listdir(run_path) if f.startswith("well_") and f.endswith(".csv")]
        if not well_files:
            print(f"❌ No well data files found in {run_path}")
            return
        
        plt.figure(figsize=(12, 8))
        
        # Colors for different wells - 96-color palette for full well plates
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        
        # Generate 96 distinct colors using multiple color maps
        colors = []
        
        # Use tab20 colormap (20 colors)
        tab20 = cm.get_cmap('tab20')(np.linspace(0, 1, 20))
        colors.extend([mcolors.rgb2hex(color) for color in tab20])
        
        # Use Set3 colormap (12 colors)
        set3 = cm.get_cmap('Set3')(np.linspace(0, 1, 12))
        colors.extend([mcolors.rgb2hex(color) for color in set3])
        
        # Use Paired colormap (12 colors)
        paired = cm.get_cmap('Paired')(np.linspace(0, 1, 12))
        colors.extend([mcolors.rgb2hex(color) for color in paired])
        
        # Use Dark2 colormap (8 colors)
        dark2 = cm.get_cmap('Dark2')(np.linspace(0, 1, 8))
        colors.extend([mcolors.rgb2hex(color) for color in dark2])
        
        # Use hsv colormap for additional colors (44 colors)
        hsv_colors = cm.get_cmap('hsv')(np.linspace(0, 1, 44))
        colors.extend([mcolors.rgb2hex(color) for color in hsv_colors])
        
        color_idx = 0
        
        for well_file in sorted(well_files):
            # Extract well name from filename; keep direction suffix when present
            base = well_file
            try:
                parts = base.split('_')
                # well_<WELL>_<timestamp>[_down|_up].csv
                well_core = parts[1]
                direction_suffix = ''
                if base.endswith('_down.csv'):
                    direction_suffix = '_down'
                elif base.endswith('_up.csv'):
                    direction_suffix = '_up'
                well_name = f"{well_core}{direction_suffix}"
            except Exception:
                well_name = base
            
            filepath = os.path.join(run_path, well_file)
            
            try:
                # Load data
                import csv
                data_rows = []
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                            data_rows.append(row)
                
                if len(data_rows) < 2:
                    print(f"⚠️ Not enough data in {well_file}")
                    continue
                
                # Extract Z positions and corrected forces
                z_positions = [abs(float(row[1])) for row in data_rows]  # Z_Position(mm) - absolute value
                corrected_forces = [abs(float(row[3])) for row in data_rows]  # Corrected_Force(N) - absolute value
                
                # Plot raw data
                color = colors[color_idx % len(colors)]
                plt.plot(z_positions, corrected_forces, 'o-', color=color, alpha=0.7, 
                        label=f'Well {well_name}', markersize=3, linewidth=1)
                
                color_idx += 1
                
            except Exception as e:
                print(f"⚠️ Error processing {well_file}: {e}")
                continue
        
        plt.xlabel('Z Position (mm)')
        plt.ylabel('Force (N) - Absolute Value')
        plt.title(f'Raw Indentation Data - {run_folder} (labels include _down/_up)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_plot:
            # Create plots directory structure
            plots_dir = "results/plots"
            os.makedirs(plots_dir, exist_ok=True)
            # Use the same run folder name for plots
            run_folder_plots = os.path.join(plots_dir, run_folder)
            os.makedirs(run_folder_plots, exist_ok=True)
            plot_filename = os.path.join(run_folder_plots, f"raw_data_all_wells.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Raw data plot saved to: {plot_filename}")
        
        plt.close()

    def plot_raw_force_individual_wells(self, run_folder: str, save_plot: bool = True):
        """Generate individual raw force plots for each well in a run"""
        data_dir = "results/measurements"
        run_path = os.path.join(data_dir, run_folder)
        
        if not os.path.exists(run_path):
            print(f"❌ Run folder {run_path} not found")
            return
        
        # Find all well data files in this run
        well_files = [f for f in os.listdir(run_path) if f.startswith("well_") and f.endswith(".csv")]
        if not well_files:
            print(f"❌ No well data files found in {run_path}")
            return
        
        # Create plots directory structure
        plots_dir = "results/plots"
        os.makedirs(plots_dir, exist_ok=True)
        run_folder_plots = os.path.join(plots_dir, run_folder)
        os.makedirs(run_folder_plots, exist_ok=True)
        
        print(f"📊 Generating individual raw force plots for {len(well_files)} wells...")
        
        for well_file in sorted(well_files):
            # Extract well name from filename
            well_name = well_file.split('_')[1]  # well_A6_xxx.csv -> A6
            
            filepath = os.path.join(run_path, well_file)
            
            try:
                # Load data
                import csv
                data_rows = []
                with open(filepath, 'r') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row and len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                            data_rows.append(row)
                
                if len(data_rows) < 2:
                    print(f"⚠️ Not enough data in {well_file}")
                    continue
                
                # Extract Z positions and forces and optional Direction column
                has_direction = len(data_rows[0]) >= 5 and data_rows[0][4] in ("down", "up") or any((len(r) >= 5 and r[4] in ("down","up")) for r in data_rows)
                if has_direction:
                    z_down, z_up = [], []
                    f_down, f_up = [], []
                    for r in data_rows:
                        z = float(r[1])
                        f_corr = float(r[3])
                        if len(r) >= 5 and r[4] == 'up':
                            z_up.append(z); f_up.append(f_corr)
                        else:
                            z_down.append(z); f_down.append(f_corr)
                else:
                    z_positions = [float(row[1]) for row in data_rows]
                    corrected_forces = [float(row[3]) for row in data_rows]
                
                # Create individual plot for this well
                plt.figure(figsize=(10, 6))
                
                if has_direction:
                    # Plot corrected force vs Z position with direction separation
                    plt.plot(z_down, f_down, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Downward (corrected)')
                    if z_up:
                        plt.plot(z_up, f_up, 'orange', marker='o', alpha=0.7, markersize=3, linewidth=1, label='Return (corrected)')
                    plt.xlabel('Z Position (mm)')
                    plt.ylabel('Corrected Force (N)')
                    plt.title(f'Well {well_name} - Corrected Force (Down vs Return)')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                else:
                    # Fallback: plot raw and corrected in subplots (legacy)
                    plt.subplot(2, 1, 1)
                    raw_forces = [float(row[2]) for row in data_rows]
                    plt.plot(z_positions, raw_forces, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Raw Force')
                    plt.xlabel('Z Position (mm)')
                    plt.ylabel('Raw Force (N)')
                    plt.title(f'Well {well_name} - Raw Force Data')
                    plt.legend(); plt.grid(True, alpha=0.3)
                    plt.subplot(2, 1, 2)
                    plt.plot(z_positions, corrected_forces, 'r-o', alpha=0.7, markersize=3, linewidth=1, label='Corrected Force')
                    plt.xlabel('Z Position (mm)')
                    plt.ylabel('Corrected Force (N)')
                    plt.title(f'Well {well_name} - Corrected Force Data')
                    plt.legend(); plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plot:
                    suffix = "_down_up" if has_direction else ""
                    plot_filename = os.path.join(run_folder_plots, f"{well_name}_raw_force{suffix}.png")
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"💾 Raw force plot for {well_name} saved to: {plot_filename}")
                
                plt.close()  # Close the figure to free memory
                
            except Exception as e:
                print(f"⚠️ Error processing {well_file}: {e}")
                continue
        
        print(f"✅ Generated individual raw force plots for {len(well_files)} wells")

    def plot_contact_detection(self, z_positions: List[float], raw_forces: List[float], contact_idx: int, well_name: str = "Unknown", save_plot: bool = True, run_folder: Optional[str] = None, baseline: float = 0.0, baseline_std: float = 0.0, method: str = "unknown", directions: Optional[List[str]] = None, direction_label: Optional[str] = None):
        """Plot force data with contact point highlighted.

        If directions are provided, plot down vs return separately in one plot.
        If direction_label is provided, treat inputs as a single-direction subset and add to title/filename.
        """
        plt.figure(figsize=(12, 8))

        # Baseline-corrected forces
        corrected_forces = [f - baseline for f in raw_forces]

        # Split by direction only if actual 'down'/'up' labels are present
        has_direction = bool(directions) and len(directions) == len(z_positions) and any(
            (d in ("down", "up")) for d in directions
        )
        if has_direction:
            z_down, z_up = [], []
            f_down, f_up = [], []
            for i, (z, f) in enumerate(zip(z_positions, corrected_forces)):
                if directions[i] == 'up':
                    z_up.append(z); f_up.append(f)
                else:
                    z_down.append(z); f_down.append(f)
            if z_down:
                plt.plot(z_down, f_down, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Downward (corrected)')
            if z_up:
                # Ensure up is shown distinctly
                plt.plot(z_up, f_up, color='orange', marker='o', alpha=0.8, markersize=3, linewidth=1, label='Return (corrected)')
        else:
            # Fallback: single series
            plt.plot(z_positions, corrected_forces, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Corrected Force')

        # If extrapolation method used, draw the extrapolation threshold (default 1/3 × max |corrected force|)
        method_lower = (method or "").lower()
        is_extrapolation = any(key in method_lower for key in ["extrap"])  # extrapolation method only
        if corrected_forces and is_extrapolation:
            max_abs_force = max(abs(f) for f in corrected_forces)
            # Match analysis_2 default: ratio = 1/3
            ratio = 1.0 / 3.0
            threshold = ratio * max_abs_force
            plt.axhline(y=threshold, color='purple', linestyle='--', alpha=0.7, label=f'Extrapolation threshold (+{threshold:.3f} N = {ratio:.2f}×max)')
            plt.axhline(y=-threshold, color='purple', linestyle='--', alpha=0.3)
        
        # Highlight contact point (skip when overlaying both directions)
        show_contact = True
        if has_direction and not direction_label:
            show_contact = False
        if show_contact and 0 <= contact_idx < len(z_positions):
            contact_z = z_positions[contact_idx]
            contact_force = corrected_forces[contact_idx]
            contact_dir = None
            if has_direction:
                contact_dir = directions[contact_idx]
            label_dir = f", dir={contact_dir}" if contact_dir else ""
            plt.plot(contact_z, contact_force, 'ro', markersize=8, label=f'Contact Point (Z={contact_z:.3f}mm, F={contact_force:.3f}N{label_dir})')
            
            # Add vertical line at contact point
            plt.axvline(x=contact_z, color='red', linestyle='--', alpha=0.5)
        
        plt.xlabel('Z Position (mm)')
        plt.ylabel('Corrected Force (N)')
        title_method = method if method and method != "unknown" else "contact"
        # For legacy data (no direction info) do not append "down vs return"
        if has_direction:
            dir_title = f" ({direction_label})" if direction_label else " (down vs return)"
        else:
            dir_title = ""
        plt.title(f'Well {well_name} - Contact Point Detection{dir_title} (Method: {title_method})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plot:
            # Create plots directory structure
            plots_dir = "results/plots"
            os.makedirs(plots_dir, exist_ok=True)
            
            if run_folder is None:
                # Create new timestamp-based folder if no run folder provided
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder_plots = os.path.join(plots_dir, f"run_{timestamp}")
            else:
                # Use the provided run folder name
                run_folder_plots = os.path.join(plots_dir, run_folder)
            
            os.makedirs(run_folder_plots, exist_ok=True)
            method_lower = (method or "").lower().replace(" ", "_")
            method_suffix = f"_{method_lower}" if method_lower else ""
            explicit_dir_suffix = f"_{direction_label.lower()}" if (direction_label and has_direction) else ""
            # Only add _down_up for combined plots when directions exist
            dir_suffix = explicit_dir_suffix or ("_down_up" if (has_direction and not direction_label) else "")
            plot_filename = os.path.join(run_folder_plots, f"{well_name}_contact_detection_{method_lower}{dir_suffix}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Contact detection plot saved to: {plot_filename}")
        
        plt.close()

    def plot_results(self, result: AnalysisResult, save_plot: bool = True, run_folder: Optional[str] = None, method: Optional[str] = None, direction_label: Optional[str] = None):
        """Plot analysis results and save to results/plots/.

        If adjusted_forces are available, plot data points and fit.
        If not, plot fit-only using the available depth range.
        """
        depths_ok = bool(getattr(result, 'depth_in_range', None)) and len(result.depth_in_range) > 0
        forces_avail = hasattr(result, 'adjusted_forces') and bool(getattr(result, 'adjusted_forces', None))

        # Determine if this is a linear-fit result (spring constant) or Hertzian
        is_linear = bool(getattr(result, 'spring_constant', None)) and getattr(result, 'spring_constant') not in (None, 0)
        
        # Check if system compliance correction was used (for Hertzian fits only)
        use_system_correction = getattr(result, 'corrected_depths', None) is not None and not is_linear
        
        # Initialize variables to avoid UnboundLocalError
        shifted_depths_original = None
        shifted_depths_corrected = None

        if not depths_ok:
            # Summary plot only
            plt.figure(figsize=(10, 6))
            if is_linear:
                k_val = float(getattr(result, 'spring_constant', 0))
                b_val = float(getattr(result, 'linear_intercept', 0))
                r2_val = float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0)))
                summary_text = f'Well {result.well}\nF = {k_val:.3f}*d + {b_val:.3f} N\nR² = {r2_val:.3f}'
            else:
                summary_text = f'Well {result.well}\nE = {result.elastic_modulus} Pa\nA = {result.fit_A:.3f}\nd0 = {result.fit_d0:.3f} mm\nR² = {result.fit_quality}'
            plt.text(0.5, 0.5, summary_text,
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.axis('off')
            plt.title(f'Well {result.well}: Analysis Summary (No Detailed Data)')
        else:
            plt.figure(figsize=(10, 6))
            depths_array = np.array(result.depth_in_range)
            if is_linear:
                # Linear fit uses direct depths (already from contact), no shift needed
                shifted_depths_all = np.maximum(depths_array, 0)
                max_depth = float(np.max(shifted_depths_all)) if shifted_depths_all.size > 0 else 2.0
                fit_depths = np.linspace(0, max_depth, 100)
                k_val = float(getattr(result, 'spring_constant', 0))
                # Get intercept from fit parameters if available
                b_val = float(getattr(result, 'linear_intercept', 0))
                fit_forces = k_val * fit_depths + b_val
            else:
                # Hertzian: Handle system compliance correction
                corrected_depths = getattr(result, 'corrected_depths', None)
                
                # Always use original depths as base
                depths_array = np.array(result.depth_in_range)
                
                if use_system_correction:
                    print("📊 Using system compliance corrected depths for plotting")
                    # Use corrected depths for fitting
                    depths_array_corrected = np.array(corrected_depths)
                else:
                    # No correction, use original depths for both
                    depths_array_corrected = depths_array
                
                # Shift corrected depths by d0 for fitting
                shifted_depths_corrected = np.maximum(depths_array_corrected - result.fit_d0, 0)
                # Shift original depths by d0 for comparison
                shifted_depths_original = np.maximum(depths_array - result.fit_d0, 0)
                
                # Fit curve domain: from 0 to max shifted depth (use corrected for fitting)
                max_depth = float(np.max(shifted_depths_corrected)) if shifted_depths_corrected.size > 0 else 2.0
                fit_depths = np.linspace(0, max_depth, 100)
                fit_forces = result.fit_A * (fit_depths) ** 1.5

            if forces_avail:
                forces_array = np.array(result.adjusted_forces)
                if is_linear:
                    # For linear fits, plot the actual data points
                    if shifted_depths_all.size > 0 and forces_array.size == shifted_depths_all.size:
                        plt.scatter(shifted_depths_all, forces_array, alpha=0.6, s=30, 
                                  color='blue', label='Measured Data')
                elif use_system_correction and shifted_depths_original is not None and shifted_depths_corrected is not None:
                    # Plot both original and corrected data when system correction is used
                    if shifted_depths_original.size > 0 and forces_array.size == shifted_depths_original.size:
                        plt.scatter(shifted_depths_original, forces_array, alpha=0.6, s=30, 
                                  color='blue', label='Original Data (shifted)')
                    if shifted_depths_corrected.size > 0 and forces_array.size == shifted_depths_corrected.size:
                        plt.scatter(shifted_depths_corrected, forces_array, alpha=0.6, s=40, 
                                  color='purple', label='System Corrected Data (shifted)')
                else:
                    # Only plot corrected data when no system correction
                    if shifted_depths_corrected is not None and shifted_depths_corrected.size > 0 and forces_array.size == shifted_depths_corrected.size:
                        plt.scatter(shifted_depths_corrected, forces_array, alpha=0.6, label='Corrected Data (shifted)')
            else:
                # No forces; visualize depth support at y=0 only
                if is_linear:
                    # For linear fits, plot depth points at y=0
                    if shifted_depths_all.size > 0:
                        plt.scatter(shifted_depths_all, np.zeros_like(shifted_depths_all), s=10, alpha=0.6, 
                                  color='blue', label='Depth points')
                elif use_system_correction and shifted_depths_original is not None and shifted_depths_corrected is not None:
                    # Plot both original and corrected depth points
                    if shifted_depths_original.size > 0:
                        plt.scatter(shifted_depths_original, np.zeros_like(shifted_depths_original), s=8, alpha=0.6, 
                                  color='blue', label='Original Depth points (shifted)')
                    if shifted_depths_corrected.size > 0:
                        plt.scatter(shifted_depths_corrected, np.zeros_like(shifted_depths_corrected), s=12, alpha=0.6, 
                                  color='purple', label='System Corrected Depth points (shifted)')
                else:
                    if shifted_depths_corrected is not None and shifted_depths_corrected.size > 0:
                        plt.scatter(shifted_depths_corrected, np.zeros_like(shifted_depths_corrected), s=10, alpha=0.6, label='Depth points (shifted)')

            # Draw the fitted curve and labels
            if is_linear:
                plt.plot(fit_depths, fit_forces, 'r-', label=f'Linear Fit (k={k_val:.3f}, b={b_val:.3f})')
                plt.xlabel('Indentation Depth (mm)')
                plt.ylabel('Force (N)')
                dir_title = f" ({direction_label})" if direction_label else ""
                r2_val = float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0)))
                plt.title(f'Well {result.well}{dir_title}: F = {k_val:.3f}*d + {b_val:.3f}, R² = {r2_val:.3f}')
            else:
                plt.plot(fit_depths, fit_forces, 'r-', label=f'Hertzian Fit (A={result.fit_A:.3f}, d0={result.fit_d0:.3f} mm)')
                # Explicitly mark the model contact at (0,0)
                plt.scatter([0], [0], c='k', marker='x', s=40, label='Model contact (0,0)')
                plt.xlabel('Indentation Depth (mm)')
                plt.ylabel('Force (N)')
                dir_title = f" ({direction_label})" if direction_label else ""
                # Add system compliance correction note to title
                compliance_note = " (system corrected)" if getattr(result, 'corrected_depths', None) is not None else ""
                plt.title(f'Well {result.well}{dir_title}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}{compliance_note}')
            plt.legend()
            plt.grid(True, alpha=0.3)

        if save_plot:
            plots_dir = "results/plots"
            os.makedirs(plots_dir, exist_ok=True)
            if run_folder is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder_plots = os.path.join(plots_dir, f"run_{timestamp}")
            else:
                run_folder_plots = os.path.join(plots_dir, run_folder)
            os.makedirs(run_folder_plots, exist_ok=True)
            method_lower = (method or "").lower().replace(" ", "_")
            method_suffix = f"_{method_lower}" if method_lower else ""
            dir_suffix = f"_{direction_label.lower()}" if direction_label else ""
            plot_filename = os.path.join(run_folder_plots, f"{result.well}_analysis{method_suffix}{dir_suffix}.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {plot_filename}")

            summary_filename = os.path.join(run_folder_plots, f"{result.well}_summary.txt")
            with open(summary_filename, 'w') as f:
                f.write(f"ASMI Analysis Results for Well {result.well}\n")
                f.write("=" * 50 + "\n")
                if is_linear:
                    f.write(f"Spring Constant k: {getattr(result, 'spring_constant', 0):.3f} N/mm\n")
                    f.write(f"Linear Intercept b: {getattr(result, 'linear_intercept', 0):.3f} N\n")
                    f.write(f"Linear Fit R²: {float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0))):.3f}\n")
                else:
                    f.write(f"Elastic Modulus: {result.elastic_modulus} Pa\n")
                    f.write(f"Uncertainty: ±{result.uncertainty} Pa\n")
                f.write(f"Poisson's Ratio: {result.poisson_ratio}\n")
                f.write(f"Sample Height: {result.sample_height} mm\n")
                if is_linear:
                    f.write(f"Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm\n")
                else:
                    f.write(f"Fit Quality (R²): {result.fit_quality}\n")
                    f.write(f"Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm\n")
                    f.write(f"Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}\n")
                f.write(f"Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"💾 Summary saved to: {summary_filename}")

        plt.close()

    def plot_well_heatmap(self, summary_csv: str, value_col: str = 'ElasticModulus', cmap: str = 'viridis', annotate: bool = True, save_path: Optional[str] = None, convert_to_mpa: bool = True):
        """Plot a 96-well plate heatmap from a summary CSV.

        CSV columns expected: 'Well', value_col (default 'ElasticModulus'), optional 'R2', optional 'Std'.
        """
        ROWS = list(string.ascii_uppercase[:8])
        COLS = list(range(1, 13))
        well_to_idx = {(f"{row}{col}"): (i, j) for i, row in enumerate(ROWS) for j, col in enumerate(COLS)}

        df = pd.read_csv(summary_csv)
        has_r2 = 'R2' in df.columns
        has_std = 'Std' in df.columns

        import numpy as _np
        heatmap = _np.full((8, 12), _np.nan)
        r2map = _np.full((8, 12), _np.nan) if has_r2 else None
        stdmap = _np.full((8, 12), _np.nan) if has_std else None

        for _, row in df.iterrows():
            well = str(row['Well'])
            value = row[value_col]
            if well in well_to_idx and pd.notnull(value) and not isinstance(value, (pd.Series, _np.ndarray)):
                i, j = well_to_idx[well]
                if convert_to_mpa and value_col == 'ElasticModulus':
                    heatmap[i, j] = value / 1e6
                else:
                    heatmap[i, j] = value
                if has_r2 and r2map is not None:
                    r2val = row['R2']
                    if pd.notnull(r2val) and not isinstance(r2val, (pd.Series, _np.ndarray)):
                        r2map[i, j] = r2val
                if has_std and stdmap is not None:
                    stdval = row['Std']
                    if pd.notnull(stdval) and not isinstance(stdval, (pd.Series, _np.ndarray)):
                        stdmap[i, j] = (stdval / 1e6) if (convert_to_mpa and value_col == 'ElasticModulus') else stdval

        fig, ax = plt.subplots(figsize=(12, 7))
        norm = mcolors.Normalize(vmin=_np.nanmin(heatmap), vmax=_np.nanmax(heatmap))
        cmap_obj = plt.get_cmap(cmap)

        for i, row_label in enumerate(ROWS):
            for j, col_label in enumerate(COLS):
                x, y = j, 7 - i
                value = heatmap[i, j]
                color = cmap_obj(norm(value)) if not _np.isnan(value) else (0.9, 0.9, 0.9, 1)
                circle = mpatches.Circle((x, y), 0.4, color=color, ec='black', lw=1.0)
                ax.add_patch(circle)
                if annotate and not _np.isnan(value):
                    rgb = cmap_obj(norm(value))[:3]
                    brightness = (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2])
                    text_color = 'black' if brightness > 0.5 else 'white'
                    ax.text(x, y+0.1, f"{value:.2f}", ha='center', va='center', fontsize=12, color=text_color, fontweight='bold')
                    if has_std and stdmap is not None and not _np.isnan(stdmap[i, j]):
                        ax.text(x, y-0.05, f"±{stdmap[i, j]:.2f}", ha='center', va='center', fontsize=8, color=text_color, fontweight='bold')
                    if has_r2 and r2map is not None and not _np.isnan(r2map[i, j]):
                        ax.text(x, y-0.2, f"R²={r2map[i, j]:.2f}", ha='center', va='center', fontsize=8, color=text_color)

        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(c) for c in COLS])
        ax.set_yticks(range(8))
        ax.set_yticklabels(ROWS[::-1])
        ax.tick_params(axis='both', which='major', labelsize=24)

        # Determine appropriate title and units based on value column
        if value_col == 'ElasticModulus':
            if convert_to_mpa:
                title = "96-Well Plate Young's Modulus Heatmap (MPa)"
                unit_label = "MPa"
            else:
                title = "96-Well Plate Young's Modulus Heatmap (Pa)"
                unit_label = "Pa"
        elif value_col == 'SpringConstant_k':
            title = "96-Well Plate Spring Constant Heatmap (N/mm)"
            unit_label = "N/mm"
        elif value_col == 'Intercept_b':
            title = "96-Well Plate Intercept Heatmap (N)"
            unit_label = "N"
        else:
            title = f"96-Well Plate {value_col} Heatmap"
            unit_label = "units"
        
        ax.set_title(title, fontsize=20)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"{value_col} ({unit_label})", fontsize=18)
        cbar.ax.tick_params(labelsize=16)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"💾 Saved heatmap to {save_path}")
            plt.close()
        else:
            plt.show()

    def plot_iterative_refinement(self, results: dict, well_name: str = "Unknown", 
                                 save_plot: bool = True, run_folder: Optional[str] = None):
        """
        Plot the iterative refinement process and results.
        
        Args:
            results: Results from iterative_contact_refinement
            well_name: Name of the well for plot title
            save_plot: Whether to save the plot
            run_folder: Folder to save plots in
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Original vs Final data
        ax1.plot(results['original_depths'], results['original_forces'], 'b-', 
                label='Original Data', alpha=0.7)
        ax1.plot(results['final_depths'], results['final_forces'], 'r-', 
                label='Final Data (Aligned)', linewidth=2)
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title(f'Well {well_name} - Data Alignment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: d0 convergence
        iterations = range(1, len(results['d0_history']) + 1)
        ax2.plot(iterations, results['d0_history'], 'bo-', linewidth=2, markersize=6)
        ax2.axhline(y=0.01, color='r', linestyle='--', alpha=0.7, label='Tolerance')
        ax2.axhline(y=-0.01, color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('d0 (mm)')
        ax2.set_title('Contact Point Convergence')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: A parameter convergence
        ax3.plot(iterations, results['A_history'], 'go-', linewidth=2, markersize=6)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('A Parameter')
        ax3.set_title('A Parameter Convergence')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Final fit
        if results['final_depths']:
            depths_fit = np.linspace(0.24, 0.50, 100)
            forces_fit = results['final_A'] * np.power(depths_fit, 1.5)
            
            ax4.plot(results['final_depths'], results['final_forces'], 'ko', 
                    markersize=4, label='Data Points')
            ax4.plot(depths_fit, forces_fit, 'r-', linewidth=2, 
                    label=f'Fit: F = {results["final_A"]:.3f}·d^1.5')
            ax4.set_xlabel('Depth (mm)')
            ax4.set_ylabel('Force (N)')
            ax4.set_title(f'Final Fit (E = {results["final_E"]:.1f} Pa)')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            if run_folder is None:
                run_folder = "results/plots"
            os.makedirs(run_folder, exist_ok=True)
            plot_filename = os.path.join(run_folder, f"{well_name}_iterative_refinement.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Iterative refinement plot saved to: {plot_filename}")
        
        plt.close()

    def plot_original_analysis_test(self, depths: List[float], forces: List[float], results: dict, well_name: str = "Test_Well"):
        """Create a plot showing the original analysis test results"""
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Original data
        ax1.scatter(depths, forces, alpha=0.6, label='Raw Data')
        ax1.set_xlabel('Depth (mm)')
        ax1.set_ylabel('Force (N)')
        ax1.set_title('Original Data')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Analysis results
        if results and results.get('converged'):
            # Get data in analysis range
            depth_range = [d for d in depths if 0.24 <= d <= 0.5]
            force_range = [f for i, f in enumerate(forces) if 0.24 <= depths[i] <= 0.5]
            
            ax2.scatter(depth_range, force_range, alpha=0.6, label='Analysis Range Data')
            
            # Fit line
            fit_depths = np.linspace(min(depth_range), max(depth_range), 100)
            fit_forces = results['fit_A'] * (fit_depths - results['fit_d0']) ** 1.5
            ax2.plot(fit_depths, fit_forces, 'r-', linewidth=2, 
                    label=f'Fit: A={results["fit_A"]:.2f}, d0={results["fit_d0"]:.3f}')
        
        ax2.set_xlabel('Depth (mm)')
        ax2.set_ylabel('Force (N)')
        ax2.set_title(f'Analysis Results\nE = {results.get("elastic_modulus", 0):,.0f} Pa')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('original_analysis_test.png', dpi=300, bbox_inches='tight')
        print("📊 Plot saved as 'original_analysis_test.png'")
        plt.close()

# Create a global instance for easy access
plotter = ASMIPlotter()