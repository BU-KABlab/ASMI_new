import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

RUN_FOLDER = 'results/well_measurements/run_007_20250711_053933'
PLOTS_FOLDER = os.path.join('results/plots', os.path.basename(RUN_FOLDER))
os.makedirs(PLOTS_FOLDER, exist_ok=True)
POISSON_RATIO = 0.33

def hertz_func(depth, A, d0):
    """Hertz model: Force = A * (depth - d0)^1.5"""
    return A * (depth - d0) ** 1.5

def calculate_E(A, poisson_ratio=0.33):
    """Calculate Young's modulus from Hertz fit parameter A"""
    # Assuming spherical indenter with radius R = 1.5 mm
    R = 1.5  # mm
    E = A * 4 / (3 * R) * (1 - poisson_ratio**2)  # MPa
    return E

for fname in os.listdir(RUN_FOLDER):
    if fname.startswith('well_') and fname.endswith('.csv') and 'summary' not in fname:
        well = fname.split('_')[1]
        print(f"🔍 Analyzing {well}...")
        
        # Load and process data
        with open(os.path.join(RUN_FOLDER, fname), 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Find header
        for i, row in enumerate(rows):
            if row and row[0] == 'Timestamp(s)':
                header_idx = i
                break
        else:
            print(f"❌ No header found for {well}")
            continue
        
        header = rows[header_idx]
        z_idx = header.index('Z_Position(mm)')
        f_idx = header.index('Force(N)')
        
        # Extract data
        data_rows = []
        for row in rows[header_idx+1:]:
            try:
                z = float(row[z_idx])
                force = float(row[f_idx])
                data_rows.append([z, force])
            except Exception:
                continue
        
        # Only negative force (indentation)
        neg_data = [row for row in data_rows if row[1] < 0]
        if not neg_data:
            print(f"❌ No negative force data for {well}")
            continue
        
        # Find monotonically increasing |z| segment (deeper indentation)
        z_values = [row[0] for row in neg_data]
        abs_z_values = [abs(z) for z in z_values]
        max_z_idx = np.argmax(abs_z_values)
        
        # Find start of monotonic |z| segment
        start_idx = 0
        for i in range(1, max_z_idx + 1):
            if abs_z_values[i] < abs_z_values[i-1]:
                start_idx = i
        
        # Extract monotonic |z| segment
        z_monotonic = neg_data[start_idx:max_z_idx+1]
        
        # Within this segment, find first point where |force| decreases
        abs_forces = [abs(row[1]) for row in z_monotonic]
        force_start_idx = 0
        
        for i in range(1, len(abs_forces)):
            if abs_forces[i] < abs_forces[i-1]:
                force_start_idx = i
                break
        
        # Extract final segment from the force starting point onwards
        segment = z_monotonic[force_start_idx:]
        
        # Filter to only include points where both |force| and |z| are increasing
        filtered_segment = []
        last_force = abs(segment[0][1]) if segment else 0
        last_z = abs(segment[0][0]) if segment else 0
        
        for row in segment:
            current_force = abs(row[1])
            current_z = abs(row[0])
            
            if current_force >= last_force and current_z >= last_z:
                filtered_segment.append(row)
                last_force = current_force
                last_z = current_z
            else:
                break
        
        segment = filtered_segment
        
        # Only use data up to z = -15 (target z)
        segment = [row for row in segment if row[0] >= -15]
        
        if len(segment) < 5:
            print(f"❌ Not enough monotonic data for {well}")
            continue
        
        # Use raw absolute Z positions as indentation (no shifting to zero)
        indentation = np.array([abs(row[0]) for row in segment])
        forces = np.array([abs(row[1]) for row in segment])  # Use absolute force
        
        # Debug output
        print(f"📊 {well}: {len(indentation)} points, indentation range: {indentation[0]:.3f}-{indentation[-1]:.3f} mm")
        print(f"📊 {well}: force range: {forces[0]:.3f}-{forces[-1]:.3f} N")
        
        # Fit Hertz model
        try:
            # Use better initial guesses
            p0 = [forces[-1] / (indentation[-1] ** 1.5), 0.0]  # Estimate A from last point, d0=0
            popt, pcov = curve_fit(hertz_func, indentation, forces, p0=p0, bounds=([0, -0.1], [100, 0.1]))
            A_fit, d0_fit = popt
            
            # Calculate R-squared
            f_pred = hertz_func(indentation, A_fit, d0_fit)
            ss_res = np.sum((forces - f_pred) ** 2)
            ss_tot = np.sum((forces - np.mean(forces)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            # Calculate Young's modulus
            E = calculate_E(A_fit, POISSON_RATIO)
            
            print(f"✅ {well}: E = {E:.1f} MPa, A = {A_fit:.3f}, d0 = {d0_fit:.4f}, R² = {r_squared:.3f}")
            
            # Plot
            plt.figure(figsize=(8, 5))
            plt.plot(indentation, forces, 'o', label='Data', markersize=4)
            
            # Plot fit
            indentation_fine = np.linspace(0, max(indentation), 100)
            fit_curve = hertz_func(indentation_fine, A_fit, d0_fit)
            plt.plot(indentation_fine, fit_curve, '-', label=f'Hertz Fit (A={A_fit:.3f}, d0={d0_fit:.4f})', linewidth=2)
            
            # Add coordinate annotations
            n_points = len(indentation)
            if n_points > 0:
                # First point
                plt.annotate(f'({indentation[0]:.3f}, {forces[0]:.3f})', 
                           xy=(indentation[0], forces[0]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                # Last point
                plt.annotate(f'({indentation[-1]:.3f}, {forces[-1]:.3f})', 
                           xy=(indentation[-1], forces[-1]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
                
                # Middle point
                if n_points > 2:
                    mid_idx = n_points // 2
                    plt.annotate(f'({indentation[mid_idx]:.3f}, {forces[mid_idx]:.3f})', 
                               xy=(indentation[mid_idx], forces[mid_idx]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            plt.xlabel('Indentation (mm)')
            plt.ylabel('Force (N)')
            plt.title(f'{well} Hertz Analysis\nE = {E:.1f} MPa, R² = {r_squared:.3f}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(PLOTS_FOLDER, f'{well}_analysis.png'), dpi=300)
            plt.close()
            
            print(f"📊 Saved plot for {well}")
            
        except Exception as e:
            print(f"❌ Fitting failed for {well}: {e}") 