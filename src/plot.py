"""
ASMI Plotting Functions - Visualization for Indentation Analysis
Contains all plotting functions for data visualization and analysis results

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
from datetime import datetime
from .version import __version__
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass
import pandas as pd
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import string
from .analysis import IndentationAnalyzer

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
    
    def __init__(self, font_size: float = 14.0):
        """Initialize plotter.
        
        Args:
            font_size: Base font size for plot labels. Use smaller values (e.g., 8) when
                plots will be resized smaller in PowerPoint for better proportions.
                Title = font_size+2, labels = font_size, ticks/small = font_size-2.
        """
        self._analyzer = IndentationAnalyzer()
        self.font_size = font_size
    
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
        fs = self.font_size
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
                    plt.xlabel('Z Position (mm)', fontsize=fs)
                    plt.ylabel('Corrected Force (N)', fontsize=fs)
                    plt.title(f'Well {well_name} - Corrected Force (Down vs Return)', fontsize=fs + 2)
                    plt.legend(fontsize=max(6, fs - 2))
                    plt.tick_params(axis='both', labelsize=max(6, fs - 2))
                    plt.grid(True, alpha=0.3)
                else:
                    # Fallback: plot raw and corrected in subplots (legacy)
                    plt.subplot(2, 1, 1)
                    raw_forces = [float(row[2]) for row in data_rows]
                    plt.plot(z_positions, raw_forces, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Raw Force')
                    plt.xlabel('Z Position (mm)', fontsize=fs)
                    plt.ylabel('Raw Force (N)', fontsize=fs)
                    plt.title(f'Well {well_name} - Raw Force Data', fontsize=fs + 2)
                    plt.legend(fontsize=max(6, fs - 2)); plt.grid(True, alpha=0.3)
                    plt.tick_params(axis='both', labelsize=max(6, fs - 2))
                    plt.subplot(2, 1, 2)
                    plt.plot(z_positions, corrected_forces, 'r-o', alpha=0.7, markersize=3, linewidth=1, label='Corrected Force')
                    plt.xlabel('Z Position (mm)', fontsize=fs)
                    plt.ylabel('Corrected Force (N)', fontsize=fs)
                    plt.title(f'Well {well_name} - Corrected Force Data', fontsize=fs + 2)
                    plt.legend(fontsize=max(6, fs - 2)); plt.grid(True, alpha=0.3)
                    plt.tick_params(axis='both', labelsize=max(6, fs - 2))
                
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
        fs = self.font_size
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
        
        plt.xlabel('Z Position (mm)', fontsize=fs)
        plt.ylabel('Corrected Force (N)', fontsize=fs)
        title_method = method if method and method != "unknown" else "contact"
        # For legacy data (no direction info) do not append "down vs return"
        if has_direction:
            dir_title = f" ({direction_label})" if direction_label else " (down vs return)"
        else:
            dir_title = ""
        plt.title(f'Well {well_name} - Contact Point Detection{dir_title} (Method: {title_method})', fontsize=fs + 2)
        plt.legend(fontsize=max(6, fs - 2))
        plt.tick_params(axis='both', labelsize=max(6, fs - 2))
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

        For Hertzian fits with system correction, three progressive plots are saved per well:
          _v1_nofc              : no force correction data + fit only
          _v2_nofc_orig         : no force correction + original (with force correction) fit
          _system_corrected     : no force correction + original + system corrected fit
        For linear fits or Hertzian without system correction, a single plot is saved.
        """
        plt.rcParams['font.family'] = 'Arial'
        fs = self.font_size
        depths_ok = bool(getattr(result, 'depth_in_range', None)) and len(result.depth_in_range) > 0
        forces_avail = hasattr(result, 'adjusted_forces') and bool(getattr(result, 'adjusted_forces', None))
        is_linear = bool(getattr(result, 'spring_constant', None)) and getattr(result, 'spring_constant') not in (None, 0)
        use_system_correction = getattr(result, 'corrected_depths', None) is not None and not is_linear

        # ── Resolve save folder once ─────────────────────────────────────────
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
        else:
            run_folder_plots = method_suffix = dir_suffix = ""

        def _savefig(fig, name_suffix):
            if not save_plot:
                return
            path = os.path.join(run_folder_plots, f"{result.well}_analysis{method_suffix}{dir_suffix}{name_suffix}.png")
            fig.savefig(path, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {path}")

        def _write_summary_txt():
            if not save_plot:
                return
            summary_filename = os.path.join(run_folder_plots, f"{result.well}_summary.txt")
            def _fmt(v, fmt_str=None):
                if v is None: return 'N/A'
                return f"{v:{fmt_str}}" if fmt_str else str(v)
            with open(summary_filename, 'w') as f:
                f.write(f"ASMI Analysis Results for Well {result.well}\n")
                f.write("=" * 50 + "\n")
                if is_linear:
                    f.write(f"Spring Constant k: {getattr(result, 'spring_constant', 0):.3f} N/mm\n")
                    f.write(f"Linear Intercept b: {getattr(result, 'linear_intercept', 0):.3f} N\n")
                    f.write(f"Linear Fit R²: {float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0))):.3f}\n")
                else:
                    f.write(f"Elastic Modulus (system corrected): {result.elastic_modulus} Pa\n")
                    f.write(f"  Uncertainty: ±{result.uncertainty} Pa, R²={result.fit_quality}, A={result.fit_A:.3f}, d0={result.fit_d0:.3f}\n")
                    E_0 = getattr(result, 'elastic_modulus_0_max', None)
                    u_0 = getattr(result, 'uncertainty_0_max', None)
                    r2_0 = getattr(result, 'fit_quality_0_max', None)
                    A_0 = getattr(result, 'fit_A_0_max', None)
                    d0_0 = getattr(result, 'fit_d0_0_max', None)
                    f.write(f"Elastic Modulus (0–max_depth, no force correction): {_fmt(E_0)}{' Pa' if E_0 is not None else ''}\n")
                    f.write(f"  Uncertainty: ±{_fmt(u_0)} Pa, R²={_fmt(r2_0, '.3f')}, A={_fmt(A_0, '.3f')}, d0={_fmt(d0_0, '.3f')}\n")
                    E_fc = getattr(result, 'elastic_modulus_min_max_fc', None)
                    u_fc = getattr(result, 'original_uncertainty', None)
                    r2_fc = getattr(result, 'original_fit_quality', None)
                    A_fc = getattr(result, 'original_fit_A', None)
                    d0_fc = getattr(result, 'original_fit_d0', None)
                    if E_fc is not None and u_fc is None and r2_fc is None and A_fc is None:
                        u_fc, r2_fc, A_fc, d0_fc = result.uncertainty, result.fit_quality, result.fit_A, result.fit_d0
                    f.write(f"Elastic Modulus (min–max_depth, with force correction): {_fmt(E_fc)}{' Pa' if E_fc is not None else ''}\n")
                    f.write(f"  Uncertainty: ±{_fmt(u_fc)} Pa, R²={_fmt(r2_fc, '.3f')}, A={_fmt(A_fc, '.3f')}, d0={_fmt(d0_fc, '.3f')}\n")
                f.write(f"Poisson's Ratio: {result.poisson_ratio}\n")
                f.write(f"Sample Height: {result.sample_height} mm\n")
                f.write(f"Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm\n")
                f.write(f"Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"💾 Summary saved to: {summary_filename}")

        # ── No-data fallback ─────────────────────────────────────────────────
        if not depths_ok:
            fig = plt.figure(figsize=(5, 4))
            if is_linear:
                k_val = float(getattr(result, 'spring_constant', 0))
                b_val = float(getattr(result, 'linear_intercept', 0))
                r2_val = float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0)))
                summary_text = f'Well {result.well}\nF = {k_val:.3f}*d + {b_val:.3f} N\nR² = {r2_val:.3f}'
            else:
                summary_text = f'Well {result.well}\nE = {result.elastic_modulus} Pa\nA = {result.fit_A:.3f}\nd0 = {result.fit_d0:.3f} mm\nR² = {result.fit_quality}'
            plt.text(0.5, 0.5, summary_text,
                    ha='center', va='center', transform=plt.gca().transAxes, fontsize=fs + 2,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
            plt.xlim(0, 1); plt.ylim(0, 1); plt.axis('off')
            plt.title(f'Well {result.well}: Analysis Summary (No Detailed Data)', fontsize=fs + 2)
            correction_suffix = "_system_corrected" if use_system_correction else ""
            _savefig(fig, correction_suffix)
            _write_summary_txt()
            plt.close(fig)
            return

        # ── Common data prep ─────────────────────────────────────────────────
        depths_array = np.array(result.depth_in_range)
        forces_array = np.array(result.adjusted_forces) if forces_avail else np.array([])
        fit_min = float(result.depth_range[0]) if result.depth_range else (min(result.depth_in_range) if result.depth_in_range else 0)
        fit_max = float(result.depth_range[1]) if result.depth_range else (max(result.depth_in_range) if result.depth_in_range else 2)

        # ── Linear fit: single plot ──────────────────────────────────────────
        if is_linear:
            shifted_depths_all = np.maximum(depths_array, 0)
            max_depth = float(np.max(shifted_depths_all)) if shifted_depths_all.size > 0 else 2.0
            fit_depths = np.linspace(0, max_depth, 100)
            k_val = float(getattr(result, 'spring_constant', 0))
            b_val = float(getattr(result, 'linear_intercept', 0))
            fit_forces = k_val * fit_depths + b_val
            fig, ax = plt.subplots(figsize=(5, 4))
            if forces_avail and shifted_depths_all.size > 0 and forces_array.size == shifted_depths_all.size:
                ax.scatter(shifted_depths_all, forces_array, alpha=0.6, s=30, color='blue', label='Measured Data')
            elif shifted_depths_all.size > 0:
                ax.scatter(shifted_depths_all, np.zeros_like(shifted_depths_all), s=10, alpha=0.6,
                           color='blue', label='Depth points')
            dir_title = f" ({direction_label})" if direction_label else ""
            r2_val = float(getattr(result, 'linear_fit_quality', getattr(result, 'fit_quality', 0)))
            ax.plot(fit_depths, fit_forces, 'r-', label=f'Linear Fit (k={k_val:.3f}, b={b_val:.3f})')
            ax.set_xlabel('Indentation Depth (mm)', fontsize=fs)
            ax.set_ylabel('Force (N)', fontsize=fs)
            ax.set_title(f'Well {result.well}{dir_title}: F = {k_val:.3f}*d + {b_val:.3f}, R² = {r2_val:.3f}', fontsize=fs + 2)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=max(6, fs - 2))
            ax.tick_params(axis='both', labelsize=max(6, fs - 2))
            ax.grid(True, alpha=0.3)
            _savefig(fig, "")
            _write_summary_txt()
            plt.close(fig)
            return

        # ── Hertzian: precompute no-fc (0–max_depth) series ─────────────────
        depth_full = getattr(result, 'depth_full', None)
        forces_full = getattr(result, 'forces_full', None)
        fit_A_0_max = getattr(result, 'fit_A_0_max', None)
        fit_d0_0_max = getattr(result, 'fit_d0_0_max', None)
        E_0_max = getattr(result, 'elastic_modulus_0_max', None)
        r2_0_max = getattr(result, 'fit_quality_0_max', None)
        has_nofc = bool(depth_full and forces_full and len(depth_full) > 0 and len(forces_full) == len(depth_full))

        if has_nofc:
            d_full_arr = np.array(depth_full)
            f_full_arr = np.array(forces_full)
            _d0_nofc = fit_d0_0_max if fit_d0_0_max is not None else result.fit_d0
            shifted_full = np.maximum(d_full_arr - _d0_nofc, 0)
            nofc_lbl_data = f'0–{fit_max:.2f} mm (no force correction)'
            if fit_A_0_max is not None and fit_d0_0_max is not None:
                x_max_nofc = max(0.0, float(np.max(d_full_arr)) - fit_d0_0_max) if d_full_arr.size > 0 else fit_max
                fit_depths_0_max = np.linspace(0, x_max_nofc, 100)
                fit_forces_0_max = fit_A_0_max * (fit_depths_0_max) ** 1.5
                nofc_lbl_fit = f'0–{fit_max:.2f} mm fit (no FC): E={E_0_max/1e6:.2f} MPa' if E_0_max else f'0–{fit_max:.2f} mm fit'
                if r2_0_max is not None:
                    nofc_lbl_fit += f', R²={r2_0_max:.3f}'
            else:
                fit_depths_0_max = None
                nofc_lbl_fit = f'0–{fit_max:.2f} mm fit'

        # ── Hertzian with system correction: 3 progressive plots ────────────
        if use_system_correction:
            print("📊 Plotting 3 progressive plots (no FC / + original / + system corrected)")
            corrected_depths = getattr(result, 'corrected_depths', None)
            depths_array_corrected = np.array(corrected_depths)

            # Fit original data (min–max_depth, with force correction, no system correction)
            if forces_avail and len(forces_array) > 0 and len(depths_array) == len(forces_array):
                lb = [0.0, -0.1]; ub = [np.inf, 2.0]
                fit_orig_res = self._analyzer.fit_hertz_model(depths_array, forces_array, bounds=(lb, ub))
                if fit_orig_res.params is not None:
                    A_original = float(fit_orig_res.params[0])
                    d0_original = float(fit_orig_res.params[1])
                    E_original = self._analyzer.adjust_E(self._analyzer.find_E(A_original, result.poisson_ratio))
                    mask = depths_array > d0_original
                    if np.sum(mask) > 5:
                        vd = depths_array[mask]; vf = forces_array[mask]
                        pred = A_original * (vd - d0_original) ** 1.5
                        r2_original = self._analyzer.calculate_r_squared(vf, pred)
                    else:
                        r2_original = 0.0
                else:
                    A_original = result.fit_A; d0_original = result.fit_d0
                    E_original = result.elastic_modulus; r2_original = result.fit_quality
            else:
                A_original = result.fit_A; d0_original = result.fit_d0
                E_original = result.elastic_modulus; r2_original = result.fit_quality

            A_corrected = result.fit_A; d0_corrected = result.fit_d0
            E_corrected = result.elastic_modulus; r2_corrected = result.fit_quality

            shifted_depths_original = np.maximum(depths_array - d0_original, 0)
            shifted_depths_corrected = np.maximum(depths_array_corrected - d0_corrected, 0)

            max_d_orig = float(np.max(shifted_depths_original)) if shifted_depths_original.size > 0 else 2.0
            max_d_corr = float(np.max(shifted_depths_corrected)) if shifted_depths_corrected.size > 0 else 2.0
            fit_depths_original = np.linspace(0, max_d_orig, 100)
            fit_depths_corrected = np.linspace(0, max_d_corr, 100)
            fit_forces_original = A_original * (fit_depths_original) ** 1.5
            fit_forces_corrected = A_corrected * (fit_depths_corrected) ** 1.5

            # ── Infer k_system and fit SC from no-FC data ────────────────────
            fit_nofc_sc_ok = False
            if has_nofc:
                diffs_ks = depths_array - depths_array_corrected
                mask_ks = np.abs(diffs_ks) > 1e-10
                if forces_avail and np.any(mask_ks):
                    k_system_inferred = float(np.median(forces_array[mask_ks] / diffs_ks[mask_ks]))
                else:
                    k_system_inferred = self._analyzer.K_SYSTEM
                corrected_d_nofc_arr = d_full_arr - f_full_arr / k_system_inferred
                lb_ns = [0.0, -0.1]; ub_ns = [np.inf, 2.0]
                fit_ns_res = self._analyzer.fit_hertz_model(corrected_d_nofc_arr, f_full_arr, bounds=(lb_ns, ub_ns))
                if fit_ns_res.params is not None:
                    A_nofc_sc = float(fit_ns_res.params[0])
                    d0_nofc_sc = float(fit_ns_res.params[1])
                    E_nofc_sc = self._analyzer.adjust_E(self._analyzer.find_E(A_nofc_sc, result.poisson_ratio))
                    mask_ns = corrected_d_nofc_arr > d0_nofc_sc
                    if np.sum(mask_ns) > 5:
                        vd_ns = corrected_d_nofc_arr[mask_ns]; vf_ns = f_full_arr[mask_ns]
                        pred_ns = A_nofc_sc * (vd_ns - d0_nofc_sc) ** 1.5
                        r2_nofc_sc = self._analyzer.calculate_r_squared(vf_ns, pred_ns)
                    else:
                        r2_nofc_sc = 0.0
                    shifted_d_nofc_sc = np.maximum(corrected_d_nofc_arr - d0_nofc_sc, 0)
                    max_d_ns = float(np.max(shifted_d_nofc_sc)) if shifted_d_nofc_sc.size > 0 else 2.0
                    fit_depths_nofc_sc = np.linspace(0, max_d_ns, 100)
                    fit_forces_nofc_sc = A_nofc_sc * (fit_depths_nofc_sc) ** 1.5
                    fit_nofc_sc_ok = True

            def _make_hertz_fig(include_nofc, include_orig, include_sc, include_nofc_sc=False, nofc_color='steelblue'):
                fig, ax = plt.subplots(figsize=(5, 4))
                if include_nofc and has_nofc:
                    ax.scatter(shifted_full, f_full_arr, alpha=0.6, s=50, color=nofc_color, label=nofc_lbl_data)
                    if fit_depths_0_max is not None:
                        ax.plot(fit_depths_0_max, fit_forces_0_max, '--', color=nofc_color, linewidth=2.5, label=nofc_lbl_fit)
                if include_orig:
                    if forces_avail and shifted_depths_original.size > 0 and forces_array.size == shifted_depths_original.size:
                        ax.scatter(shifted_depths_original, forces_array, alpha=0.7, s=50, color='blue',
                                   label=f'Original ({fit_min:.2f}–{fit_max:.2f} mm)')
                    ax.plot(fit_depths_original, fit_forces_original, 'b-', linewidth=2.5,
                            label=f'Original Fit: E = {E_original/1e6:.2f} MPa, A = {A_original:.3f}, d0 = {d0_original:.3f} mm, R² = {r2_original:.3f}')
                if include_sc:
                    if forces_avail and shifted_depths_corrected.size > 0 and forces_array.size == shifted_depths_corrected.size:
                        ax.scatter(shifted_depths_corrected, forces_array, alpha=0.7, s=60, color='purple',
                                   label=f'System Corrected ({fit_min:.2f}–{fit_max:.2f} mm)')
                    ax.plot(fit_depths_corrected, fit_forces_corrected, 'r-', linewidth=2.5,
                            label=f'System Corrected Fit: E = {E_corrected/1e6:.2f} MPa, A = {A_corrected:.3f}, d0 = {d0_corrected:.3f} mm, R² = {r2_corrected:.3f}')
                if include_nofc_sc and fit_nofc_sc_ok:
                    ax.scatter(shifted_d_nofc_sc, f_full_arr, alpha=0.6, s=50, color='green',
                               label=f'No FC (SC depths)')
                    ax.plot(fit_depths_nofc_sc, fit_forces_nofc_sc, 'g-', linewidth=2.5,
                            label=f'SC Fit (from no-FC): E = {E_nofc_sc/1e6:.2f} MPa, A = {A_nofc_sc:.3f}, d0 = {d0_nofc_sc:.3f} mm, R² = {r2_nofc_sc:.3f}')
                ax.set_xlabel('Indentation Depth (mm)', fontsize=fs)
                ax.set_ylabel('Force (N)', fontsize=fs)
                ax.set_title(f'Well {result.well.upper()} measurement', fontsize=fs + 2)
                ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=max(6, fs - 2))
                ax.tick_params(axis='both', labelsize=max(6, fs - 2))
                return fig

            fig1 = _make_hertz_fig(True, False, False)
            _savefig(fig1, '_v1_nofc')
            plt.close(fig1)

            fig2 = _make_hertz_fig(True, True, False)
            _savefig(fig2, '_v2_nofc_orig')
            plt.close(fig2)

            fig3 = _make_hertz_fig(True, True, True)
            _savefig(fig3, '_system_corrected')
            plt.close(fig3)

            fig4 = _make_hertz_fig(False, True, True)
            _savefig(fig4, '_v3_orig_sc')
            plt.close(fig4)

            fig5 = _make_hertz_fig(True, False, False, include_nofc_sc=True)
            _savefig(fig5, '_v4_nofc_sc_nofc')
            plt.close(fig5)

        # ── Hertzian without system correction: single plot ──────────────────
        else:
            shifted_depths_corrected = np.maximum(depths_array - result.fit_d0, 0)
            max_depth = float(np.max(shifted_depths_corrected)) if shifted_depths_corrected.size > 0 else 2.0
            fit_depths = np.linspace(0, max_depth, 100)
            fit_forces = result.fit_A * (fit_depths) ** 1.5

            fig, ax = plt.subplots(figsize=(5, 4))
            if has_nofc:
                ax.scatter(shifted_full, f_full_arr, alpha=0.6, s=50, color='steelblue', label=nofc_lbl_data)
                if fit_depths_0_max is not None:
                    ax.plot(fit_depths_0_max, fit_forces_0_max, '--', color='steelblue', linewidth=2.5, label=nofc_lbl_fit)
            if forces_avail and shifted_depths_corrected.size > 0 and forces_array.size == shifted_depths_corrected.size:
                ax.scatter(shifted_depths_corrected, forces_array, alpha=0.7, s=50, label='Corrected Data (shifted)')
            elif shifted_depths_corrected.size > 0:
                ax.scatter(shifted_depths_corrected, np.zeros_like(shifted_depths_corrected), s=40, alpha=0.6,
                           label='Depth points (shifted)')
            ax.plot(fit_depths, fit_forces, 'r-', linewidth=2.5,
                    label=f'Hertzian Fit: E = {result.elastic_modulus/1e6:.2f} MPa, A = {result.fit_A:.3f}, d0 = {result.fit_d0:.3f} mm, R² = {result.fit_quality:.3f}')
            ax.set_xlabel('Indentation Depth (mm)', fontsize=fs)
            ax.set_ylabel('Force (N)', fontsize=fs)
            ax.set_title(f'Well {result.well.upper()} measurement', fontsize=fs + 2)
            ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=max(6, fs - 2))
            ax.tick_params(axis='both', labelsize=max(6, fs - 2))
            _savefig(fig, "")
            plt.close(fig)

        _write_summary_txt()

    def plot_well_heatmap(self, summary_csv: str, value_col: str = 'ElasticModulus', cmap: str = 'viridis', annotate: bool = True, save_path: Optional[str] = None, convert_to_mpa: bool = True, title_suffix: Optional[str] = None):
        """Plot a 96-well plate heatmap from a summary CSV.

        CSV columns expected: 'Well', value_col (default 'ElasticModulus'), optional 'R2', optional 'Std'.
        Font sizes scale with plotter.font_size for consistent resizing in PowerPoint.
        
        Args:
            summary_csv: Path to CSV file with well data
            value_col: Column name to plot (default: 'ElasticModulus')
            cmap: Colormap name (default: 'viridis')
            annotate: Whether to annotate wells with values (default: True)
            save_path: Path to save the plot (if None, displays plot)
            convert_to_mpa: Convert Pa to MPa for ElasticModulus (default: True)
            title_suffix: Optional suffix to add to the title (e.g., " (System Corrected)")
        """
        fs = 8  # Fixed smaller font size for heatmap to fit 96-well annotations
        ROWS = list(string.ascii_uppercase[:8])
        COLS = list(range(1, 13))
        well_to_idx = {(f"{row}{col}"): (i, j) for i, row in enumerate(ROWS) for j, col in enumerate(COLS)}

        df = pd.read_csv(summary_csv)
        # Pick R2 and Std columns to match value_col
        r2_col = {'ElasticModulus_system_corrected': 'R2_system_corrected', 'ElasticModulus_min_max_fc': 'R2_min_max_fc', 'ElasticModulus_0_max': 'R2_0_max'}.get(value_col, 'R2')
        std_col = {'ElasticModulus_system_corrected': 'Std_system_corrected', 'ElasticModulus_min_max_fc': 'Std_min_max_fc', 'ElasticModulus_0_max': 'Std_0_max'}.get(value_col, 'Std')
        has_r2 = r2_col in df.columns
        has_std = std_col in df.columns

        import numpy as _np
        heatmap = _np.full((8, 12), _np.nan)
        r2map = _np.full((8, 12), _np.nan) if has_r2 else None
        stdmap = _np.full((8, 12), _np.nan) if has_std else None

        for _, row in df.iterrows():
            well = str(row['Well'])
            value = row[value_col]
            if well in well_to_idx and pd.notnull(value) and not isinstance(value, (pd.Series, _np.ndarray)):
                i, j = well_to_idx[well]
                # Convert Pa to MPa for ElasticModulus columns
                is_elastic_modulus = value_col in ('ElasticModulus', 'ElasticModulus_Original', 'ElasticModulus_system_corrected', 'ElasticModulus_min_max_fc', 'ElasticModulus_0_max')
                if convert_to_mpa and is_elastic_modulus:
                    heatmap[i, j] = value / 1e6
                else:
                    heatmap[i, j] = value
                if has_r2 and r2map is not None:
                    r2val = row[r2_col]
                    if pd.notnull(r2val) and not isinstance(r2val, (pd.Series, _np.ndarray)):
                        r2map[i, j] = r2val
                if has_std and stdmap is not None:
                    stdval = row[std_col]
                    if pd.notnull(stdval) and not isinstance(stdval, (pd.Series, _np.ndarray)):
                        stdmap[i, j] = (stdval / 1e6) if (convert_to_mpa and is_elastic_modulus) else stdval

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
                    ax.text(x, y+0.1, f"{value:.2f}", ha='center', va='center', fontsize=fs + 2, color=text_color, fontweight='bold')
                    if has_std and stdmap is not None and not _np.isnan(stdmap[i, j]):
                        ax.text(x, y-0.05, f"±{stdmap[i, j]:.2f}", ha='center', va='center', fontsize=fs, color=text_color, fontweight='bold')
                    if has_r2 and r2map is not None and not _np.isnan(r2map[i, j]):
                        ax.text(x, y-0.2, f"R²={r2map[i, j]:.2f}", ha='center', va='center', fontsize=fs, color=text_color)

        ax.set_xlim(-0.5, 11.5)
        ax.set_ylim(-0.5, 7.5)
        ax.set_aspect('equal')
        ax.set_xticks(range(12))
        ax.set_xticklabels([str(c) for c in COLS])
        ax.set_yticks(range(8))
        ax.set_yticklabels(ROWS[::-1])
        ax.tick_params(axis='both', which='major', labelsize=fs + 4)

        # Determine appropriate title and units based on value column
        if value_col in ('ElasticModulus', 'ElasticModulus_system_corrected'):
            if convert_to_mpa:
                title = "96-Well Plate Young's Modulus Heatmap (MPa)"
                unit_label = "MPa"
            else:
                title = "96-Well Plate Young's Modulus Heatmap (Pa)"
                unit_label = "Pa"
        elif value_col in ('ElasticModulus_Original', 'ElasticModulus_min_max_fc'):
            if convert_to_mpa:
                title = "96-Well Plate Young's Modulus Heatmap (MPa) - Original"
                unit_label = "MPa"
            else:
                title = "96-Well Plate Young's Modulus Heatmap (Pa) - Original"
                unit_label = "Pa"
        elif value_col == 'ElasticModulus_0_max':
            if convert_to_mpa:
                title = "96-Well Plate Young's Modulus Heatmap (MPa) - 0-max (no FC)"
                unit_label = "MPa"
            else:
                title = "96-Well Plate Young's Modulus Heatmap (Pa) - 0-max (no FC)"
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
        
        # Add title suffix if provided
        if title_suffix:
            title = title + title_suffix
        
        ax.set_title(title, fontsize=fs + 10)

        sm = plt.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label(f"{value_col} ({unit_label})", fontsize=fs + 8)
        cbar.ax.tick_params(labelsize=fs + 6)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300)
            print(f"💾 Saved heatmap to {save_path}")
            
            # Save heatmap data to CSV
            csv_path = save_path.replace('.png', '_data.csv')
            heatmap_data = []
            for i, row_label in enumerate(ROWS):
                for j, col_label in enumerate(COLS):
                    well = f"{row_label}{col_label}"
                    value = heatmap[i, j]
                    row_data = {
                        'Well': well,
                        'Row': row_label,
                        'Column': col_label,
                        value_col: value if not _np.isnan(value) else ''
                    }
                    if has_r2 and r2map is not None:
                        r2_val = r2map[i, j]
                        row_data['R2'] = r2_val if not _np.isnan(r2_val) else ''
                    if has_std and stdmap is not None:
                        std_val = stdmap[i, j]
                        row_data['Std'] = std_val if not _np.isnan(std_val) else ''
                    heatmap_data.append(row_data)
            
            # Write to CSV
            heatmap_df = pd.DataFrame(heatmap_data)
            heatmap_df.to_csv(csv_path, index=False)
            print(f"💾 Saved heatmap data to {csv_path}")
            
            plt.close()
        else:
            plt.show()

    def plot_correction_comparison(self, summary_csv: str, save_path: Optional[str] = None, convert_to_mpa: bool = True):
        """Plot comparison of elastic modulus before and after system correction.
        
        Shows scatter plot, distributions, and statistics to assess if correction
        improves consistency (reduces variance) of elastic modulus values.
        
        Args:
            summary_csv: Path to summary CSV with ElasticModulus_system_corrected and ElasticModulus_min_max_fc columns
            save_path: Path to save the plot (if None, displays plot)
            convert_to_mpa: Convert Pa to MPa for display (default: True)
        """
        import pandas as pd
        fs = self.font_size
        
        if not os.path.exists(summary_csv):
            print(f"❌ CSV file not found: {summary_csv}")
            return
        
        df = pd.read_csv(summary_csv)
        
        # Check if we have both original and corrected values
        orig_col = 'ElasticModulus_min_max_fc' if 'ElasticModulus_min_max_fc' in df.columns else 'ElasticModulus_Original'
        corr_col = 'ElasticModulus_system_corrected' if 'ElasticModulus_system_corrected' in df.columns else 'ElasticModulus'
        if corr_col not in df.columns or orig_col not in df.columns:
            print(f"❌ CSV must contain both '{corr_col}' and '{orig_col}' columns")
            return
        
        # Filter out empty/invalid values
        valid_data = df.copy()
        valid_data[corr_col] = pd.to_numeric(valid_data[corr_col], errors='coerce')
        valid_data[orig_col] = pd.to_numeric(valid_data[orig_col], errors='coerce')
        valid_data = valid_data.dropna(subset=[corr_col, orig_col])
        valid_data = valid_data[(valid_data[corr_col] > 0) & (valid_data[orig_col] > 0)]
        
        if len(valid_data) == 0:
            print(f"❌ No valid data found in CSV")
            return
        
        # Convert to MPa if requested
        if convert_to_mpa:
            original = valid_data[orig_col] / 1e6
            corrected = valid_data[corr_col] / 1e6
            unit = "MPa"
        else:
            original = valid_data[orig_col]
            corrected = valid_data[corr_col]
            unit = "Pa"
        
        # Calculate statistics
        orig_mean = np.mean(original)
        orig_std = np.std(original)
        orig_cv = (orig_std / orig_mean * 100) if orig_mean > 0 else 0
        
        corr_mean = np.mean(corrected)
        corr_std = np.std(corrected)
        corr_cv = (corr_std / corr_mean * 100) if corr_mean > 0 else 0
        
        # Create figure with subplots
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # 1. Scatter plot: Original vs Corrected
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(original, corrected, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        
        # Add y=x line
        min_val = min(original.min(), corrected.min())
        max_val = max(original.max(), corrected.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='y=x (no change)')
        
        ax1.set_xlabel(f'Original Elastic Modulus ({unit})', fontsize=fs)
        ax1.set_ylabel(f'Corrected Elastic Modulus ({unit})', fontsize=fs)
        ax1.set_title('Original vs Corrected Elastic Modulus', fontsize=fs + 2, fontweight='bold')
        ax1.legend(fontsize=max(6, fs - 2))
        ax1.tick_params(axis='both', labelsize=max(6, fs - 2))
        ax1.grid(True, alpha=0.3)
        ax1.set_aspect('equal', adjustable='box')
        
        # 2. Distribution comparison
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(original, bins=20, alpha=0.6, label=f'Original (μ={orig_mean:.2f}, σ={orig_std:.2f})', color='blue', edgecolor='black')
        ax2.hist(corrected, bins=20, alpha=0.6, label=f'Corrected (μ={corr_mean:.2f}, σ={corr_std:.2f})', color='red', edgecolor='black')
        ax2.set_xlabel(f'Elastic Modulus ({unit})', fontsize=fs)
        ax2.set_ylabel('Frequency', fontsize=fs)
        ax2.set_title('Distribution Comparison', fontsize=fs + 2, fontweight='bold')
        ax2.legend(fontsize=max(6, fs - 2))
        ax2.tick_params(axis='both', labelsize=max(6, fs - 2))
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Box plot comparison
        ax3 = fig.add_subplot(gs[0, 2])
        box_data = [original, corrected]
        bp = ax3.boxplot(box_data, labels=['Original', 'Corrected'], patch_artist=True)
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')
        ax3.set_ylabel(f'Elastic Modulus ({unit})', fontsize=fs)
        ax3.set_title('Box Plot Comparison', fontsize=fs + 2, fontweight='bold')
        ax3.tick_params(axis='both', labelsize=max(6, fs - 2))
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. Difference plot (Corrected - Original)
        ax4 = fig.add_subplot(gs[1, 0])
        difference = corrected - original
        percent_change = (difference / original * 100) if original.mean() > 0 else difference
        ax4.scatter(original, difference, alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='r', linestyle='--', linewidth=2)
        ax4.set_xlabel(f'Original Elastic Modulus ({unit})', fontsize=fs)
        ax4.set_ylabel(f'Difference: Corrected - Original ({unit})', fontsize=fs)
        ax4.set_title('Correction Effect', fontsize=fs + 2, fontweight='bold')
        ax4.tick_params(axis='both', labelsize=max(6, fs - 2))
        ax4.grid(True, alpha=0.3)
        
        # 5. Percent change distribution
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.hist(percent_change, bins=20, alpha=0.7, color='green', edgecolor='black')
        ax5.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax5.set_xlabel('Percent Change (%)', fontsize=fs)
        ax5.set_ylabel('Frequency', fontsize=fs)
        ax5.set_title('Percent Change Distribution', fontsize=fs + 2, fontweight='bold')
        mean_pct = np.mean(percent_change)
        ax5.axvline(x=mean_pct, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_pct:.1f}%')
        ax5.legend(fontsize=max(6, fs - 2))
        ax5.tick_params(axis='both', labelsize=max(6, fs - 2))
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. Statistics summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        # Calculate additional statistics
        orig_median = np.median(original)
        corr_median = np.median(corrected)
        orig_range = original.max() - original.min()
        corr_range = corrected.max() - corrected.min()
        
        # Calculate improvement metrics
        std_reduction = ((orig_std - corr_std) / orig_std * 100) if orig_std > 0 else 0
        cv_reduction = ((orig_cv - corr_cv) / orig_cv * 100) if orig_cv > 0 else 0
        
        stats_text = f"""
        STATISTICS SUMMARY
        {'='*50}
        
        ORIGINAL (Before Correction):
        • Mean: {orig_mean:.2f} {unit}
        • Median: {orig_median:.2f} {unit}
        • Std Dev: {orig_std:.2f} {unit}
        • CV: {orig_cv:.2f}%
        • Range: {orig_range:.2f} {unit}
        • Count: {len(original)}
        
        CORRECTED (After Correction):
        • Mean: {corr_mean:.2f} {unit}
        • Median: {corr_median:.2f} {unit}
        • Std Dev: {corr_std:.2f} {unit}
        • CV: {corr_cv:.2f}%
        • Range: {corr_range:.2f} {unit}
        • Count: {len(corrected)}
        
        IMPROVEMENT:
        • Std Reduction: {std_reduction:.1f}%
        • CV Reduction: {cv_reduction:.1f}%
        • Mean Change: {corr_mean - orig_mean:.2f} {unit} ({mean_pct:.1f}%)
        """
        
        ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes, fontsize=fs,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Overall title
        fig.suptitle('System Correction Analysis: Elastic Modulus Comparison', 
                    fontsize=fs + 6, fontweight='bold', y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"💾 Correction comparison plot saved to: {save_path}")
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