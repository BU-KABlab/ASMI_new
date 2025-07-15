import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pandas as pd
from datetime import datetime

# Configuration
RUN_FOLDER = 'results/well_measurements/run_007_20250711_053933'
PLOTS_FOLDER = 'results/plots/run_007_analysis'
POISSON_RATIO = 0.33

# Well information mapping
WELL_INFO = {
    'A5': ('5.0M', '1%', 'blue'),
    'B5': ('5.0M', '3%', 'green'),
    'C5': ('5.0M', '5%', 'red'),
    'A6': ('6.0M', '5%', 'purple'),
    'B6': ('6.0M', '3%', 'orange'),
    'C6': ('6.0M', '1%', 'brown'),
}

def hertz_func(depth, A, d0):
    """Hertz model: Force = A * (depth - d0)^1.5"""
    return A * (depth - d0) ** 1.5

def create_plots_folder():
    """Create the plots folder if it doesn't exist"""
    os.makedirs(PLOTS_FOLDER, exist_ok=True)
    print(f"📁 Created plots folder: {PLOTS_FOLDER}")

def analyze_well_data(filename, well):
    """Analyze a single well's data"""
    print(f"🔍 Analyzing {well} from {filename}")
    
    # Load data
    data = []
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) >= 4 and row[0] != 'Timestamp(s)' and row[0] != 'Test_Time':
                try:
                    timestamp = float(row[0])
                    z_pos = float(row[1])
                    force = float(row[2])
                    movement = row[3] if len(row) > 3 else 'Down'
                    data.append([timestamp, z_pos, force, movement])
                except ValueError:
                    continue
    
    if not data:
        print(f"❌ No valid data found for {well}")
        return None
    
    # Convert to numpy arrays
    data = np.array(data)
    timestamps = data[:, 0].astype(float)
    z_positions = data[:, 1].astype(float)
    forces = data[:, 2].astype(float)  # Ensure forces are float
    movements = data[:, 3]
    
    # Select only Down movement
    down_mask = movements == 'Down'
    if np.sum(down_mask) < 10:
        print(f"❌ Not enough down movement data for {well}")
        return None
    
    z_down = z_positions[down_mask]
    f_down = forces[down_mask]
    
    # Only use data up to z = -15 (target z)
    target_mask = z_down >= -15
    if np.sum(target_mask) < 5:
        print(f"❌ Not enough data up to target Z for {well}")
        return None
    
    z_target = z_down[target_mask]
    f_target = f_down[target_mask]
    
    # Find the point where force becomes negative (start of indentation)
    negative_mask = f_target < 0
    if np.sum(negative_mask) < 5:
        print(f"❌ Not enough negative force data for {well}")
        return None
    
    # Get the first negative force point
    first_negative_idx = np.where(negative_mask)[0][0]
    
    # Use data from first negative force onwards
    z_indent = z_target[first_negative_idx:]
    f_indent = f_target[first_negative_idx:]
    
    if len(z_indent) < 5:
        print(f"❌ Not enough indentation data for {well}")
        return None
    
    # Find monotonically increasing |force| segment
    abs_forces = np.abs(f_indent)
    max_idx = np.argmax(abs_forces)
    
    # Find the start of monotonically increasing segment
    start_idx = 0
    for i in range(1, max_idx + 1):
        if abs_forces[i] < abs_forces[i-1]:
            start_idx = i
    
    # Extract the monotonic segment
    z_monotonic = z_indent[start_idx:max_idx+1]
    f_monotonic = f_indent[start_idx:max_idx+1]
    
    if len(z_monotonic) < 5:
        print(f"❌ Not enough monotonic data for {well}")
        return None
    
    # Shift first force to zero and set first z as zero
    f_shifted = f_monotonic - f_monotonic[0]
    indentation = np.abs(z_monotonic - z_monotonic[0])
    
    # Fit Hertz model
    try:
        popt, pcov = curve_fit(hertz_func, indentation, np.abs(f_shifted), 
                              p0=[2, 0.03], bounds=([0, 0], [100, 1]))
        A_fit, d0_fit = popt
        
        # Calculate R-squared
        f_pred = hertz_func(indentation, A_fit, d0_fit)
        ss_res = np.sum((np.abs(f_shifted) - f_pred) ** 2)
        ss_tot = np.sum((np.abs(f_shifted) - np.mean(np.abs(f_shifted))) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate Young's modulus (E = A * 4/(3*R) * (1-nu^2))
        # Assuming spherical indenter with radius R = 1.5 mm
        R = 1.5  # mm
        E = A_fit * 4 / (3 * R) * (1 - POISSON_RATIO**2)  # MPa
        
        return {
            'well': well,
            'z_data': z_monotonic,
            'f_data': f_shifted,
            'indentation': indentation,
            'A_fit': A_fit,
            'd0_fit': d0_fit,
            'r_squared': r_squared,
            'E': E,
            'max_force': np.max(np.abs(f_shifted)),
            'max_indentation': np.max(indentation)
        }
        
    except Exception as e:
        print(f"❌ Fitting failed for {well}: {e}")
        return None

def plot_individual_well(result, plots_folder):
    """Create individual plot for a well"""
    well = result['well']
    cm, intensity, color = WELL_INFO[well]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Force vs Z position
    ax1.plot(result['z_data'], result['f_data'], 'o-', color=color, 
             label=f'{well} ({cm}, {intensity})', markersize=4)
    ax1.set_xlabel('Z Position (mm)')
    ax1.set_ylabel('Force (N)')
    ax1.set_title(f'Force vs Z Position - {well}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Force vs Indentation with fit
    indentation_fine = np.linspace(0, result['max_indentation'], 100)
    force_fit = hertz_func(indentation_fine, result['A_fit'], result['d0_fit'])
    
    ax2.plot(result['indentation'], np.abs(result['f_data']), 'o', color=color, 
             label='Data', markersize=4)
    ax2.plot(indentation_fine, force_fit, '-', color='red', linewidth=2, 
             label=f'Fit (R²={result["r_squared"]:.3f})')
    ax2.set_xlabel('Indentation (mm)')
    ax2.set_ylabel('|Force| (N)')
    ax2.set_title(f'Hertz Fit - {well}\nE = {result["E"]:.1f} MPa')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, f'{well}_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved individual plot for {well}")

def create_summary_plot(all_results, plots_folder):
    """Create summary plot comparing all wells"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Group by concentration
    cm5_results = [r for r in all_results if WELL_INFO[r['well']][0] == '5.0M']
    cm6_results = [r for r in all_results if WELL_INFO[r['well']][0] == '6.0M']
    
    # Plot 1: Young's modulus comparison
    wells_5m = [r['well'] for r in cm5_results]
    E_5m = [r['E'] for r in cm5_results]
    intensities_5m = [WELL_INFO[r['well']][1] for r in cm5_results]
    
    wells_6m = [r['well'] for r in cm6_results]
    E_6m = [r['E'] for r in cm6_results]
    intensities_6m = [WELL_INFO[r['well']][1] for r in cm6_results]
    
    # Plot bars
    x_pos_5m = np.arange(len(wells_5m))
    x_pos_6m = np.arange(len(wells_6m)) + len(wells_5m) + 1
    
    bars1 = ax1.bar(x_pos_5m, E_5m, color='skyblue', alpha=0.7, label='Cm=5.0M')
    bars2 = ax1.bar(x_pos_6m, E_6m, color='lightcoral', alpha=0.7, label='Cm=6.0M')
    
    # Add value labels on bars
    for bar, E in zip(bars1, E_5m):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{E:.1f}', ha='center', va='bottom', fontsize=9)
    for bar, E in zip(bars2, E_6m):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{E:.1f}', ha='center', va='bottom', fontsize=9)
    
    ax1.set_xlabel('Well')
    ax1.set_ylabel('Young\'s Modulus (MPa)')
    ax1.set_title('Young\'s Modulus Comparison')
    ax1.set_xticks(np.concatenate([x_pos_5m, x_pos_6m]))
    ax1.set_xticklabels(wells_5m + wells_6m)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Force vs Indentation overlay
    colors_5m = [WELL_INFO[r['well']][2] for r in cm5_results]
    colors_6m = [WELL_INFO[r['well']][2] for r in cm6_results]
    
    for i, (result, color) in enumerate(zip(cm5_results, colors_5m)):
        intensity = WELL_INFO[result['well']][1]
        ax2.plot(result['indentation'], np.abs(result['f_data']), 'o-', 
                color=color, markersize=3, alpha=0.7, 
                label=f'{result["well"]} ({intensity})')
    
    for i, (result, color) in enumerate(zip(cm6_results, colors_6m)):
        intensity = WELL_INFO[result['well']][1]
        ax2.plot(result['indentation'], np.abs(result['f_data']), 's-', 
                color=color, markersize=3, alpha=0.7, 
                label=f'{result["well"]} ({intensity})')
    
    ax2.set_xlabel('Indentation (mm)')
    ax2.set_ylabel('|Force| (N)')
    ax2.set_title('Force vs Indentation - All Wells')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_folder, 'summary_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Saved summary comparison plot")

def save_results_table(all_results, plots_folder):
    """Save results to a CSV table"""
    results_data = []
    for result in all_results:
        well = result['well']
        cm, intensity, _ = WELL_INFO[well]
        results_data.append({
            'Well': well,
            'Concentration': cm,
            'Intensity': intensity,
            'Young_Modulus_MPa': result['E'],
            'R_squared': result['r_squared'],
            'Max_Force_N': result['max_force'],
            'Max_Indentation_mm': result['max_indentation'],
            'A_Fit': result['A_fit'],
            'd0_Fit': result['d0_fit']
        })
    
    df = pd.DataFrame(results_data)
    df = df.sort_values(['Concentration', 'Well'])
    
    csv_path = os.path.join(plots_folder, 'analysis_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"📊 Saved results table to {csv_path}")
    
    # Also print summary
    print("\n" + "="*60)
    print("ANALYSIS RESULTS SUMMARY")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)

def main():
    """Main analysis function"""
    print("🔬 Starting analysis of run_007 data...")
    print(f"📁 Data folder: {RUN_FOLDER}")
    
    # Create plots folder
    create_plots_folder()
    
    # Get all well files
    well_files = [f for f in os.listdir(RUN_FOLDER) 
                  if f.startswith('well_') and f.endswith('.csv') and 'summary' not in f]
    
    print(f"📋 Found {len(well_files)} well files to analyze")
    
    all_results = []
    
    # Analyze each well
    for filename in well_files:
        well = filename.split('_')[1]  # Extract well name from filename
        filepath = os.path.join(RUN_FOLDER, filename)
        
        result = analyze_well_data(filepath, well)
        if result:
            all_results.append(result)
            plot_individual_well(result, PLOTS_FOLDER)
    
    if all_results:
        # Create summary plot
        create_summary_plot(all_results, PLOTS_FOLDER)
        
        # Save results table
        save_results_table(all_results, PLOTS_FOLDER)
        
        print(f"\n✅ Analysis complete! {len(all_results)} wells analyzed successfully.")
        print(f"📁 All plots saved to: {PLOTS_FOLDER}")
    else:
        print("❌ No wells could be analyzed successfully.")

if __name__ == "__main__":
    main() 