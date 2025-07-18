"""
ASMI Data Analysis - Elastic Modulus Calculation from Indentation Data
Uses Hertzian contact mechanics to analyze force-indentation measurements
Incorporates comprehensive correction factors and iterative fitting procedures
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
import math
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import sys
from collections import namedtuple
FitResult = namedtuple("FitResult", ["params", "covariance", "model_func"])

@dataclass
class AnalysisResult:
    well: str
    elastic_modulus: float
    uncertainty: float
    poisson_ratio: float
    sample_height: float
    fit_quality: float
    depth_range: tuple  # or Tuple[float, float] if you import Tuple
    fit_A: float
    fit_d0: float
    adjusted_forces: list  # or List[float] if you import List
    depth_in_range: list   # or List[float]
    material_type: str
    contact_z: float  # Z position at first contact
    contact_force: float  # Force at first contact

class IndentationAnalyzer:
    """Analyzes indentation data to calculate elastic modulus with full correction factors"""
    
    # Physical constants
    SPHERE_RADIUS = 0.0025  # m
    SPHERE_E = 1.8e11       # Pa
    SPHERE_NU = 0.28
    
    # Analysis parameters
    INDENTATION_DEPTH_THRESHOLD = 2.5  # mm - indentation depth threshold
    FORCE_THRESHOLD = 2.0      # N - force threshold to detect contact
    FORCE_LIMIT = 45.0         # N - force limit for filtering data
    
     # Well geometry constants
    WELL_DEPTH = 10.9 # mm - total depth of the well
    WELL_TOP_Z = -8.5  # mm - Z position when indenter is at well top
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.data = None
    
    def load_data(self, filename: str) -> bool:
        """Load CSV data file"""
        try:
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, 'r') as file:
                reader = csv.reader(file)
                self.data = [row for row in reader if row]
            print(f"✅ Loaded {len(self.data)} data points from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return False
    
    def find_contact_point(self, corrected_forces: List[float], force_threshold: float = FORCE_THRESHOLD) -> int:
        """Find the first contact point where force starts to change significantly.
        Uses multiple strategies to robustly detect contact even with noisy data.
        
        Args:
            corrected_forces: List of baseline-corrected force values
            force_threshold: Force threshold to detect contact (default: 2.0)
        Returns:
            Index of the first contact point
        """
        if len(corrected_forces) < 10:
            # Fallback to simple threshold for very short datasets
            for i, force in enumerate(corrected_forces):
                if abs(force) > force_threshold:
                    return i
            return 0
        
        # Strategy 1: Look for sustained positive force trend
        # Find the first point where force becomes positive and stays positive for several points
        min_sustained_points = 3
        for i in range(len(corrected_forces) - min_sustained_points):
            if (corrected_forces[i] > force_threshold and 
                all(corrected_forces[j] > force_threshold for j in range(i+1, i+min_sustained_points+1))):
                return i
        
        # Strategy 2: Look for significant force increase from baseline
        # Calculate rolling average to smooth noise
        window_size = min(5, len(corrected_forces) // 4)
        smoothed_forces = []
        for i in range(len(corrected_forces)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(corrected_forces), i + window_size // 2 + 1)
            smoothed_forces.append(np.mean(corrected_forces[start_idx:end_idx]))
        
        # Find first point where smoothed force exceeds threshold
        for i, force in enumerate(smoothed_forces):
            if abs(force) > force_threshold:
                return i
        
        # Strategy 3: Look for change in force trend
        # Calculate force derivatives to find where trend changes
        if len(corrected_forces) > 5:
            derivatives = []
            for i in range(1, len(corrected_forces)):
                derivatives.append(corrected_forces[i] - corrected_forces[i-1])
            
            # Find first point where derivative becomes consistently positive
            for i in range(len(derivatives) - 2):
                if (derivatives[i] > 0 and derivatives[i+1] > 0 and 
                    abs(corrected_forces[i+1]) > force_threshold):
                    return i + 1
        
        # Fallback: simple threshold
        for i, force in enumerate(corrected_forces):
            if abs(force) > force_threshold:
                return i
        
        return 0  # Default to first point if no contact detected

    def find_true_contact_point(self, z_positions: List[float], corrected_forces: List[float]) -> int:
        """Find the true contact point by extrapolating back to zero force.
        This compensates for the 2N threshold delay by finding where force would be zero.
        
        Args:
            z_positions: List of Z positions
            corrected_forces: List of baseline-corrected force values
        Returns:
            Index of the true contact point (where force would be zero)
        """
        # First find the 2N threshold contact point
        threshold_contact_idx = self.find_contact_point(corrected_forces)
        
        if threshold_contact_idx == 0 or threshold_contact_idx >= len(corrected_forces) - 1:
            return threshold_contact_idx
        
        # Get the force and position at threshold contact
        threshold_force = corrected_forces[threshold_contact_idx]
        threshold_z = z_positions[threshold_contact_idx]
        
        # If threshold force is very close to 2N, we can extrapolate back
        if abs(threshold_force) > 1.5:  # If we're at least 1.5N, we can extrapolate
            # Look at the trend before contact to estimate true contact point
            # Use a few points before the threshold contact to establish trend
            lookback_points = min(5, threshold_contact_idx)
            
            if lookback_points >= 2:
                # Get the trend in the region before threshold contact
                pre_contact_forces = corrected_forces[threshold_contact_idx - lookback_points:threshold_contact_idx]
                pre_contact_z = z_positions[threshold_contact_idx - lookback_points:threshold_contact_idx]
                
                # Calculate average force gradient (N/mm)
                if len(pre_contact_forces) >= 2:
                    force_gradient = (pre_contact_forces[-1] - pre_contact_forces[0]) / (pre_contact_z[-1] - pre_contact_z[0])
                    
                    if abs(force_gradient) > 0.1:  # If there's a meaningful gradient
                        # Extrapolate back to zero force
                        z_offset_to_zero = threshold_force / force_gradient
                        true_contact_z = threshold_z - z_offset_to_zero
                        
                        # Find the closest Z position to this extrapolated point
                        closest_idx = threshold_contact_idx
                        min_distance = abs(z_positions[threshold_contact_idx] - true_contact_z)
                        
                        for i in range(max(0, threshold_contact_idx - 10), threshold_contact_idx):
                            distance = abs(z_positions[i] - true_contact_z)
                            if distance < min_distance:
                                min_distance = distance
                                closest_idx = i
                        
                        print(f"🔍 True contact extrapolated to Z={true_contact_z:.3f}mm (index {closest_idx})")
                        return closest_idx
        
        # Fallback to threshold contact point
        return threshold_contact_idx
    
    def calculate_indentation_depth(self, z_positions: List[float], first_contact_idx: int, corrected_forces: Optional[List[float]] = None) -> Tuple[List[float], float, List[float]]:
        """Calculate indentation depths relative to first contact point and shift forces to zero at contact.
        Args:
            z_positions: List of Z positions
            first_contact_idx: Index of first contact point
            corrected_forces: List of corrected forces (optional, for force shifting)
        Returns:
            Tuple of (depths, z_contact, shifted_forces) starting from first contact
        """
        z_contact = z_positions[first_contact_idx]
        depths = [abs(z - z_contact) for z in z_positions[first_contact_idx:]]
        
        # Shift forces to zero at contact point
        shifted_forces = []
        if corrected_forces is not None:
            contact_force = corrected_forces[first_contact_idx]
            shifted_forces = [f - contact_force for f in corrected_forces[first_contact_idx:]]
        else:
            shifted_forces = [0.0] * len(depths)  # Placeholder
            
        return depths, z_contact, shifted_forces
    
    def calculate_approx_height(self, z_contact: float) -> float:
        """Calculate approximate height of the sample based on contact Z position.
        Args:
            z_contact: Z position at first contact
        Returns:
            Approximate height in mm
        """
        # Height = Well depth - distance from well top to first contact
        approx_height = self.WELL_DEPTH - abs(z_contact - self.WELL_TOP_Z)
        # Ensure height is within reasonable bounds
        approx_height = max(0.1, min(approx_height, self.WELL_DEPTH))
        return approx_height
    
    def adjust_force(self, depths: List[float], forces: List[float], p_ratio: float, approx_height: float) -> List[float]:
        """Add simulation-based adjustment factor since samples are not ideal shapes.
        Only applies adjustment to depths within the analysis range (0.24-0.5mm).
        Args:
            depths: List of indentation depths
            forces: List of forces
            p_ratio: Poisson's ratio of the sample
            approx_height: Approximate height of the sample
        Returns:
            List of adjusted forces
        """
        new_array = []
        for i in range(len(depths)): 
            # Only adjust forces for depths within the analysis range (0.24-0.5mm)
            if 0.24 <= depths[i] <= 0.5:
                if p_ratio < 0.325:
                    if approx_height >= 9.5:
                        b = 0.13
                        c = 1.24
                    elif approx_height >= 8.5 and approx_height < 9.5:
                        b = 0.131
                        c = 1.24
                    elif approx_height >= 7.5 and approx_height < 8.5:
                        b = 0.133
                        c = 1.25
                    elif approx_height >= 6.5 and approx_height < 7.5:
                        b = 0.132
                        c = 1.24
                    elif approx_height >= 5.5 and approx_height < 6.5:
                        b = 0.132
                        c = 1.24
                    elif approx_height >= 4.5 and approx_height < 5.5:
                        b = 0.139
                        c = 1.27
                    elif approx_height >= 3.5 and approx_height < 4.5:
                        b = 0.149
                        c = 1.3
                    else:
                        b = 0.162
                        c = 1.38
                elif p_ratio >= 0.325 and p_ratio < 0.375:
                    if approx_height >= 9.5:
                        b = 0.132
                        c = 1.25
                    elif approx_height >= 8.5 and approx_height < 9.5:
                        b = 0.132
                        c = 1.25
                    elif approx_height >= 7.5 and approx_height < 8.5:
                        b = 0.134
                        c = 1.25
                    elif approx_height >= 6.5 and approx_height < 7.5:
                        b = 0.136
                        c = 1.26
                    elif approx_height >= 5.5 and approx_height < 6.5:
                        b = 0.126
                        c = 1.25
                    elif approx_height >= 4.5 and approx_height < 5.5:
                        b = 0.133
                        c = 1.27
                    elif approx_height >= 3.5 and approx_height < 4.5:
                        b = 0.144
                        c = 1.32
                    else:
                        b = 0.169
                        c = 1.42
                elif p_ratio >= 0.375 and p_ratio < 0.425:
                    if approx_height >= 9.5:
                        b = 0.181
                        c = 1.33
                    elif approx_height >= 8.5 and approx_height < 9.5:
                        b = 0.182
                        c = 1.34
                    elif approx_height >= 7.5 and approx_height < 8.5:
                        b = 0.183
                        c = 1.34
                    elif approx_height >= 6.5 and approx_height < 7.5:
                        b = 0.183
                        c = 1.34
                    elif approx_height >= 5.5 and approx_height < 6.5:
                        b = 0.194
                        c = 1.38
                    elif approx_height >= 4.5 and approx_height < 5.5:
                        b = 0.198
                        c = 1.4
                    elif approx_height >= 3.5 and approx_height < 4.5:
                        b = 0.203
                        c = 1.44
                    else:
                        b = 0.176
                        c = 1.46
                elif p_ratio >= 0.425 and p_ratio < 0.475:
                    if approx_height >= 9.5:
                        b = 0.156
                        c = 1.35
                    elif approx_height >= 8.5 and approx_height < 9.5:
                        b = 0.152
                        c = 1.34
                    elif approx_height >= 7.5 and approx_height < 8.5:
                        b = 0.156
                        c = 1.35
                    elif approx_height >= 6.5 and approx_height < 7.5:
                        b = 0.161
                        c = 1.37
                    elif approx_height >= 5.5 and approx_height < 6.5:
                        b = 0.153
                        c = 1.37
                    elif approx_height >= 4.5 and approx_height < 5.5:
                        b = 0.166
                        c = 1.42
                    elif approx_height >= 3.5 and approx_height < 4.5:
                        b = 0.179
                        c = 1.47
                    else:
                        b = 0.205
                        c = 1.59
                else:
                    if approx_height >= 9.5:
                        b = 0.203
                        c = 1.58
                    elif approx_height >= 8.5 and approx_height < 9.5:
                        b = 0.207
                        c = 1.6
                    elif approx_height >= 7.5 and approx_height < 8.5:
                        b = 0.212
                        c = 1.62
                    elif approx_height >= 6.5 and approx_height < 7.5:
                        b = 0.217
                        c = 1.65
                    elif approx_height >= 5.5 and approx_height < 6.5:
                        b = 0.21
                        c = 1.64
                    elif approx_height >= 4.5 and approx_height < 5.5:
                        b = 0.22
                        c = 1.68
                    elif approx_height >= 3.5 and approx_height < 4.5:
                        b = 0.17
                        c = 1.58
                    else:
                        b = 0.182
                        c = 1.64
                # Avoid division by zero by adding small epsilon to depth
                depth_with_epsilon = max(depths[i], 1e-6)  # Minimum depth of 1 micron
                adjusted_force = abs(forces[i]) / (c*pow(depth_with_epsilon, b))
                new_array.append(adjusted_force)
            else:
                # For depths outside analysis range, use original force
                new_array.append(abs(forces[i]))
        return new_array
    
    def detect_force_limit_reached(self, filename: str) -> Tuple[bool, float]:
        """If force limit was reached during measurement by checking metadata or final force value."""
        try:
            force_limit = 45.0  # Default force limit
            final_force = None
            
            with open(filename, 'r') as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and len(row) >= 2:
                        if row[0] == 'Force_Limit(N)':
                            force_limit = float(row[1])
                        elif row[0] == 'Force_Exceeded':
                            force_exceeded = row[1].lower() == 'true'
                            return force_exceeded, force_limit
                        # Check if this is a data row (numeric first column)
                        elif len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                            # This is a data row, get the corrected force (4th column)
                            try:
                                final_force = abs(float(row[3]))  # Corrected_Force(N)
                            except (ValueError, IndexError):
                                pass
            
            # If Force_Exceeded not found, check if final force exceeds the limit
            if final_force is not None:
                force_exceeded = final_force >= force_limit * 0.95  # Within 5% of limit
                return force_exceeded, force_limit
            
            return False, force_limit  # Default values
        except Exception as e:
            print(f"⚠️ Could not detect force limit status: {e}")
            return False, 45.0  # Default values
    
    def determine_poisson_ratio(self, filename: str) -> Tuple[float, str]:
        """Automatically determine Poisson's ratio based on force limit detection."""
        force_limit_reached, force_limit = self.detect_force_limit_reached(filename)
        
        if force_limit_reached:
            poisson_ratio = 0.3
            material_type = "glassy_polymer"
            print(f"🔍 Force limit reached ({force_limit}N) - Classified as glassy polymer (ν = 0.3)")
        else:
            poisson_ratio = 0.5
            material_type = "gel"
            print(f"🔍 Force limit not reached ({force_limit}N) - Classified as gel (ν = 0.5)")
        
        return poisson_ratio, material_type

    def add_adjusted_force_column(self, filename, poisson_ratio=None):
        """
        Add a column of adjusted force using adjust_force and approx_height to a measurement CSV.
        The new file will be saved as <original>_with_sim.csv in the same folder.
        
        Args:
            filename: Path to the measurement CSV file
            poisson_ratio: Optional Poissons ratio. If None, automatically determined from force limit.
        """
        # Auto-determine Poisson's ratio if not provided
        if poisson_ratio is None:
            poisson_ratio, material_type = self.determine_poisson_ratio(filename)
            print(f"🔍 Auto-detected material type: {material_type} (ν = {poisson_ratio})")
        else:
            material_type = "manual"  # User-specified Poissons ratio
        meta = [] # metadata rows
        data = [] # data rows
        with open(filename, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row or 'Timestamp' in row[0]:
                    meta.append(row)
                elif len(row) >= 4 and all(x.replace('.', '', 1).replace('-', '', 1).isdigit() for x in row[1:4]):
                    data.append(row)
                else:
                    meta.append(row)
        if len(data) < 2:
            print("Not enough data to compute adjusted force.")
            return
        z_positions = [float(row[1]) for row in data]
        corrected_forces = [float(row[3]) for row in data]
        # 1. Find contact point
        first_contact_idx = self.find_contact_point(corrected_forces)
        # 2. Calculate indentation depths
        depths, z_contact, shifted_forces = self.calculate_indentation_depth(z_positions, first_contact_idx, corrected_forces)
        forces = shifted_forces
        # 3. Calculate approx height
        approx_height = self.calculate_approx_height(z_contact)
        # 4. Adjust force
        adjusted_force_sim = self.adjust_force(depths, forces, poisson_ratio, approx_height)
        # Write new file
        outname = filename[:-4] + '_with_sim.csv' if filename.endswith('.csv') else filename + '_with_sim.csv'
        with open(outname, 'w', newline='') as f:
            writer = csv.writer(f)
            for row in meta:
                writer.writerow(row)
            writer.writerow(['Material_Type', material_type])
            writer.writerow(['Poisson_Ratio', f"{poisson_ratio:.3f}"])
            writer.writerow(['Approx_Height(mm)', f"{approx_height:.3f}"])
            writer.writerow([])
            writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)', 'Adjusted_Force_Sim(N)'])
            for i, row in enumerate(data):
                if i >= first_contact_idx and (i - first_contact_idx) < len(adjusted_force_sim):
                    writer.writerow(row[:4] + [f"{adjusted_force_sim[i - first_contact_idx]:.3f}"])
                else:
                    writer.writerow(row[:4] + [""])
        print(f"💾 File with adjusted force column saved to: {outname}")
        print(f"📊 Used Poissons ratio: {poisson_ratio} ({material_type})")
    
    
    def find_E(self, A: float, p_ratio: float) -> float:
        """
        Calculate elastic modulus from Hertzian contact fit parameter.

        This function converts the fitted parameter A from the Hertzian model
        F = A * (d - d0)^1.5 to the elastic modulus E of the sample.

        Args:
            A: Fitted parameter from Hertzian model (N/mm^1.5)
            p_ratio: Poisson's ratio of the sample (dimensionless)

        Returns:
            Elastic modulus in Pa

        Raises:
            ValueError: If input parameters are invalid
        """
        # Input validation
        if A <= 0:
            raise ValueError(f"Fitted parameter A must be positive, got {A}")
        if not (0.1 <= p_ratio <= 0.5):
            raise ValueError(f"Poisson's ratio must be between 0.1 and 0.5, got {p_ratio}")

        # Use class constants for indenter properties
        r_sphere = self.SPHERE_RADIUS  # m
        sphere_p_ratio = self.SPHERE_NU  # dimensionless
        sphere_E = self.SPHERE_E  # Pa

        # Convert A from N/mm^1.5 to N/m^1.5 (SI units)
        A_SI = A * (1000 ** 1.5)  # Convert mm to m

        # From Hertzian model: A = (4/3) * E* * R^0.5
        # Therefore: E* = (3/4) * A / R^0.5
        E_star = (3 / 4) * A_SI / (r_sphere ** 0.5)

        # Calculate sample elastic modulus from reduced modulus
        # Using the relationship: 1/E* = (1-ν_sample²)/E_sample + (1-ν_indenter²)/E_indenter
        # For very stiff indenter, the second term is negligible
        # E_sample = E* * (1 - ν_sample²)
        E_sample = E_star * (1 - p_ratio ** 2)

        # Include indenter contribution for more accurate results
        # This is the full relationship without neglecting indenter properties
        E_inv = (1 - p_ratio ** 2) / E_sample - (1 - sphere_p_ratio ** 2) / sphere_E
        E_final = 1 / E_inv if E_inv != 0 else E_sample

        return E_final
    
    
    def adjust_E(self, E: float) -> float:
        """
        Apply an empirical correction factor for soft samples at small indentation depths.

        This correction is based on observed deviations for E < threshold.
        Reference: [add reference or note if available]

        Args:
            E: Elastic modulus (Pa)

        Returns:
            Corrected elastic modulus (Pa)
        """
        SOFT_THRESHOLD = 660000  # Pa
        CORR_A = 457
        CORR_B = -0.457

        if E < SOFT_THRESHOLD:
            factor = CORR_A * pow(E, CORR_B)
            corrected_E = E / factor
            print(f"⚠️ Empirical correction applied to soft sample: original E={E:.1f} Pa, corrected E={corrected_E:.1f} Pa")
            return corrected_E
        return E
    
    def fit_hertz_model(self, depths: np.ndarray, forces: np.ndarray, bounds=None, raise_on_fail=False) -> FitResult:
        """
        Fit data to the Hertzian contact model: F = A * (d - d0)^1.5

        Args:
            depths: np.ndarray of indentation depths (mm)
            forces: np.ndarray of forces (N)
            bounds: Optional tuple of (lower_bounds, upper_bounds) for (A, d0)
            raise_on_fail: If True, raise exception on failure

        Returns:
            FitResult namedtuple (params, covariance, model_func)
        """
        def hertz_func(depth, A, d0):
            return A * np.power(np.maximum(depth - d0, 0), 1.5)

        # Input validation
        if len(depths) < 5 or len(forces) < 5:
            raise ValueError("Not enough data points for fitting.")
        if np.any(np.isnan(depths)) or np.any(np.isnan(forces)):
            raise ValueError("Input data contains NaNs.")

        try:
            if bounds is not None:
                params, covariance = curve_fit(hertz_func, depths, forces, p0=[2, 0.03], bounds=bounds)
            else:
                params, covariance = curve_fit(hertz_func, depths, forces, p0=[2, 0.03])
            return FitResult(params, covariance, hertz_func)
        except Exception as e:
            print(f"❌ Curve fitting failed: {e}")
            if raise_on_fail:
                raise
            return FitResult(None, None, hertz_func)
    
    def analyze_well(self, well: str, poisson_ratio: Optional[float] = None, filename: Optional[str] = None) -> Optional[AnalysisResult]:
        """Complete analysis for a single well with full correction factors"""
        print(f"\n🔬 Analyzing well {well}...")
        
        # Auto-determine Poisson's ratio if not provided
        if poisson_ratio is None and filename:
            poisson_ratio, material_type = self.determine_poisson_ratio(filename)
        elif poisson_ratio is None:
            print("❌ Poisson's ratio must be provided if filename is not available")
            return None
        else:
            material_type = "manual"  # User-specified Poissons ratio
        
        if self.data is None:
            print("❌ No data loaded")
            return None
        
        # Parse CSV data - skip metadata rows and get data rows
        data_rows = []
        for row in self.data:
            # Skip metadata rows (those with non-numeric first column)
            if len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                data_rows.append(row)
        
        if len(data_rows) < 10:
            print("❌ Not enough data points for analysis")
            return None
        
        # Extract Z positions and corrected forces
        z_positions = [float(row[1]) for row in data_rows]  # Z_Position(mm)
        corrected_forces = [float(row[3]) for row in data_rows]  # Corrected_Force(N)
        
        # Find true contact point by extrapolating back to zero force
        first_contact_idx = self.find_true_contact_point(z_positions, corrected_forces)
        print(f"🔍 True contact detected at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm, Force={corrected_forces[first_contact_idx]:.3f}N")
        
        # Generate contact detection plot for validation
        if filename:
            # Extract run folder name from filename for plot organization
            run_folder = None
            for part in filename.split(os.sep):
                if part.startswith("run_"):
                    run_folder = part
                    break
            
            self.plot_contact_detection(z_positions, corrected_forces, first_contact_idx, well, save_plot=True, run_folder=run_folder)
        
        # Calculate indentation depths relative to first contact
        depths, z_contact, shifted_forces = self.calculate_indentation_depth(z_positions, first_contact_idx, corrected_forces)
        forces = shifted_forces
        
        # Filter data to analysis range
        depth_in_range = []
        force_in_range = []
        for d, f in zip(depths, forces):
            if 0 <= d <= self.INDENTATION_DEPTH_THRESHOLD:
                depth_in_range.append(d)
                force_in_range.append(f)
        
        if len(depth_in_range) < 5:
            print("❌ Not enough data points in analysis range")
            return None
        
        # Remove the last point if it exceeds force limit (check if final force is close to 45N)
        if len(force_in_range) > 1 and abs(force_in_range[-1]) > self.FORCE_LIMIT:  # Close to 45N limit
            print(f"⚠️ Removing last point (force: {force_in_range[-1]:.2f}N) as it exceeds force limit")
            depth_in_range = depth_in_range[:-1]
            force_in_range = force_in_range[:-1]
        
        # Calculate approximate height
        approx_height = self.calculate_approx_height(z_contact)
        
        # Apply simulation-based force adjustment
        adjusted_forces = self.adjust_force(depth_in_range, force_in_range, float(poisson_ratio), approx_height)
        
        # Fit Hertzian model (both A and d0)
        depths_array = np.array(depth_in_range)
        adjusted_forces_array = np.array(adjusted_forces)
        
        params, covariance, hertz_func = self.fit_hertz_model(depths_array, adjusted_forces_array)
        if params is None:
            print("❌ Curve fitting failed")
            return None
        
        fit_A = float(params[0])
        fit_d0 = float(params[1])
        
        # Calculate elastic modulus
        E = self.find_E(fit_A, poisson_ratio)
        E = self.adjust_E(E)  # Apply empirical correction
        E = round(E)
        
        # Calculate uncertainty
        if covariance is not None:
            err = np.sqrt(np.diag(covariance))
            E_uncertainty = round(self.find_E(err[0], poisson_ratio))
        else:
            E_uncertainty = 0
        
        # Calculate fit quality (R-squared) - only for valid depth points
        valid_mask = depths_array > fit_d0  # Only points where (d - d0) > 0
        if np.sum(valid_mask) > 5:  # Need at least 5 valid points
            valid_depths = depths_array[valid_mask]
            valid_forces = adjusted_forces_array[valid_mask]
            predicted = fit_A * (valid_depths - fit_d0) ** 1.5
            ss_res = np.sum((valid_forces - predicted) ** 2)
            ss_tot = np.sum((valid_forces - np.mean(valid_forces)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0  # Not enough valid points for R² calculation
        # Check depth range
        if depth_in_range and round(max(depth_in_range), 2) < 0.4:
            print("⚠️ Sample was not indented far enough")
            print(f"The range the measurement was made with was {round(min(depth_in_range), 2)} mm to {round(max(depth_in_range), 2)} mm")
        
        return AnalysisResult(
            well=well,
            elastic_modulus=E,
            uncertainty=E_uncertainty,
            poisson_ratio=poisson_ratio,
            sample_height=round(approx_height, 1),
            fit_quality=float(round(r_squared, 3)),
            depth_range=(min(depth_in_range), max(depth_in_range)),
            fit_A=fit_A,
            fit_d0=fit_d0 if fit_d0 is not None else 0.0,
            adjusted_forces=adjusted_forces,
            depth_in_range=depth_in_range,
            material_type=material_type, # Use the determined material type
            contact_z=round(z_contact, 3),  # Z position at first contact
            contact_force=round(corrected_forces[first_contact_idx], 3)  # Force at first contact
        )
    
    def plot_raw_data_all_wells(self, run_folder: str, save_plot: bool = True):
        """Plot raw data (absolute values) for all wells in a single plot"""
        import os
        from datetime import datetime
        import matplotlib.pyplot as plt
        
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
            # Extract well name from filename
            well_name = well_file.split('_')[1]  # well_A6_xxx.csv -> A6
            
            filepath = os.path.join(run_path, well_file)
            
            try:
                # Load data
                if not self.load_data(filepath):
                    print(f"⚠️ Could not load data from {well_file}")
                    continue
                
                # Parse data rows
                data_rows = []
                if self.data is not None:
                    for row in self.data:
                        if len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
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
        plt.title(f'Raw Indentation Data - {run_folder}')
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
        
        # plt.show()  # Commented out to prevent interruption in automated process

    def plot_raw_force_individual_wells(self, run_folder: str, save_plot: bool = True):
        """Generate individual raw force plots for each well in a run"""
        import os
        import matplotlib.pyplot as plt
        
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
                if not self.load_data(filepath):
                    print(f"⚠️ Could not load data from {well_file}")
                    continue
                
                # Parse data rows
                data_rows = []
                if self.data is not None:
                    for row in self.data:
                        if len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                            data_rows.append(row)
                
                if len(data_rows) < 2:
                    print(f"⚠️ Not enough data in {well_file}")
                    continue
                
                # Extract Z positions and forces
                z_positions = [float(row[1]) for row in data_rows]  # Z_Position(mm)
                raw_forces = [float(row[2]) for row in data_rows]  # Raw_Force(N)
                corrected_forces = [float(row[3]) for row in data_rows]  # Corrected_Force(N)
                
                # Create individual plot for this well
                plt.figure(figsize=(10, 6))
                
                # Plot raw force vs Z position
                plt.subplot(2, 1, 1)
                plt.plot(z_positions, raw_forces, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Raw Force')
                plt.xlabel('Z Position (mm)')
                plt.ylabel('Raw Force (N)')
                plt.title(f'Well {well_name} - Raw Force Data')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Plot corrected force vs Z position
                plt.subplot(2, 1, 2)
                plt.plot(z_positions, corrected_forces, 'r-o', alpha=0.7, markersize=3, linewidth=1, label='Corrected Force')
                plt.xlabel('Z Position (mm)')
                plt.ylabel('Corrected Force (N)')
                plt.title(f'Well {well_name} - Corrected Force Data')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                if save_plot:
                    plot_filename = os.path.join(run_folder_plots, f"{well_name}_raw_force.png")
                    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
                    print(f"💾 Raw force plot for {well_name} saved to: {plot_filename}")
                
                plt.close()  # Close the figure to free memory
                
            except Exception as e:
                print(f"⚠️ Error processing {well_file}: {e}")
                continue
        
        print(f"✅ Generated individual raw force plots for {len(well_files)} wells")

    def plot_contact_detection(self, z_positions: List[float], corrected_forces: List[float], contact_idx: int, well_name: str = "Unknown", save_plot: bool = True, run_folder: Optional[str] = None):
        """Plot force data with contact point highlighted for validation"""
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        plt.figure(figsize=(12, 8))
        
        # Plot corrected force data
        plt.plot(z_positions, corrected_forces, 'b-o', alpha=0.7, markersize=3, linewidth=1, label='Corrected Force')
        
        # Highlight contact point
        if 0 <= contact_idx < len(z_positions):
            contact_z = z_positions[contact_idx]
            contact_force = corrected_forces[contact_idx]
            plt.plot(contact_z, contact_force, 'ro', markersize=8, label=f'Contact Point (Z={contact_z:.3f}mm, F={contact_force:.3f}N)')
            
            # Add vertical line at contact point
            plt.axvline(x=contact_z, color='red', linestyle='--', alpha=0.5)
        
        # Add threshold lines
        plt.axhline(y=self.FORCE_THRESHOLD, color='green', linestyle=':', alpha=0.7, label=f'Threshold (+{self.FORCE_THRESHOLD}N)')
        plt.axhline(y=-self.FORCE_THRESHOLD, color='green', linestyle=':', alpha=0.7, label=f'Threshold (-{self.FORCE_THRESHOLD}N)')
        
        plt.xlabel('Z Position (mm)')
        plt.ylabel('Corrected Force (N)')
        plt.title(f'Well {well_name} - Contact Point Detection')
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
            plot_filename = os.path.join(run_folder_plots, f"{well_name}_contact_detection.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Contact detection plot saved to: {plot_filename}")
        
        plt.close()

    def plot_results(self, result: AnalysisResult, save_plot: bool = True, run_folder: Optional[str] = None):
        """Plot analysis results with corrected data and fit, and save to results/plots/"""
        if not result.depth_in_range or not result.adjusted_forces:
            print("❌ No data available for plotting")
            return
        import os
        from datetime import datetime
        plt.figure(figsize=(10, 6))
        
        # Shift data so that d0 becomes (0,0)
        depths_array = np.array(result.depth_in_range)
        forces_array = np.array(result.adjusted_forces)
        
        # Only plot data from d0 onwards
        valid_mask = depths_array >= result.fit_d0
        if np.sum(valid_mask) > 0:
            shifted_depths = depths_array[valid_mask] - result.fit_d0  # Shift so d0 becomes 0
            shifted_forces = forces_array[valid_mask]
            
            # Plot shifted corrected data
            plt.scatter(shifted_depths, shifted_forces, alpha=0.6, label='Corrected Data (shifted)')
            
            # Plot shifted fit - extend to origin (0,0)
            max_depth = max(shifted_depths) if len(shifted_depths) > 0 else 2.0
            fit_depths = np.linspace(0, max_depth, 100)  # Start from 0 to include origin
            fit_forces = result.fit_A * (fit_depths) ** 1.5  # d0 is now 0
            plt.plot(fit_depths, fit_forces, 'r-', label=f'Hertzian Fit (A={result.fit_A:.3f})')
        else:
            plt.scatter([], [], alpha=0.6, label='Corrected Data (shifted)')
            plt.plot([], [], 'r-', label=f'Hertzian Fit (A={result.fit_A:.3f})')
        
        plt.xlabel('Indentation Depth (mm)')
        plt.ylabel('Force (N)')
        plt.title(f'Well {result.well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}')
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
            plot_filename = os.path.join(run_folder_plots, f"{result.well}_analysis.png")
            plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
            print(f"💾 Plot saved to: {plot_filename}")
            # Save summary
            summary_filename = os.path.join(run_folder_plots, f"{result.well}_summary.txt")
            with open(summary_filename, 'w') as f:
                f.write(f"ASMI Analysis Results for Well {result.well}\n")
                f.write("=" * 50 + "\n")
                f.write(f"Elastic Modulus: {result.elastic_modulus} Pa\n")
                f.write(f"Uncertainty: ±{result.uncertainty} Pa\n")
                f.write(f"Poisson's Ratio: {result.poisson_ratio}\n")
                f.write(f"Sample Height: {result.sample_height} mm\n")
                f.write(f"Fit Quality (R²): {result.fit_quality}\n")
                f.write(f"Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm\n")
                f.write(f"Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}\n")
                f.write(f"Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            print(f"💾 Summary saved to: {summary_filename}")
        # plt.show()  # Commented out to prevent interruption in automated process



def main():
    """Main analysis function (non-interactive, argument-based)"""
    print("🔬 ASMI Indentation Analysis (Full Version)")
    
    if len(sys.argv) < 3:
        print("Usage: python analysis.py <datafile.csv> <well> [poisson_ratio]")
        print("Example: python analysis.py results/measurements/run_001/well_A6_20250710_215330.csv A6")
        print("Example: python analysis.py results/measurements/run_001/well_A6_20250710_215330.csv A6 0.33")
        print("Note: If poisson_ratio is not provided, it will be automatically determined based on force limit")
        return
    
    datafile = sys.argv[1]
    well = sys.argv[2].upper()

    # Optional Poisson's ratio argument
    poisson_ratio = None
    if len(sys.argv) >= 4:
        try:
            poisson_ratio = float(sys.argv[3])
            if not (0.3 <= poisson_ratio <= 0.5):
                raise ValueError("Poisson's ratio out of range")
        except ValueError as e:
            print(f"❌ Invalid Poisson's ratio: {e}")
            return

    # Determine data directory and filename
    data_dir, filename = os.path.split(datafile)
    analyzer = IndentationAnalyzer(data_dir or ".")
    if not analyzer.load_data(filename):
        return
    
    # Use full filepath for automatic Poisson's ratio detection
    full_filepath = os.path.join(data_dir, filename) if data_dir else filename
    result = analyzer.analyze_well(well, poisson_ratio, full_filepath)
    if result:
        print(f"\n📊 Results for Well {result.well}:")
        print(f"   Elastic Modulus: {result.elastic_modulus} Pa")
        print(f"   Uncertainty: ±{result.uncertainty} Pa")
        print(f"   Sample Height: {result.sample_height} mm")
        print(f"   Fit Quality (R²): {result.fit_quality}")
        print(f"   Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm")
        print(f"   Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}")
        print(f"   Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N")
        print(f"   Material Type: {result.material_type}")

        # Save summary in same folder as data file
        summary_filename = os.path.join(data_dir, f"{well}_analysis_summary.txt")
        with open(summary_filename, 'w') as f:
            f.write(f"ASMI Analysis Results for Well {result.well}\n")
            f.write("=" * 50 + "\n")
            f.write(f"Elastic Modulus: {result.elastic_modulus} Pa\n")
            f.write(f"Uncertainty: ±{result.uncertainty} Pa\n")
            f.write(f"Poisson's Ratio: {result.poisson_ratio}\n")
            f.write(f"Sample Height: {result.sample_height} mm\n")
            f.write(f"Fit Quality (R²): {result.fit_quality}\n")
            f.write(f"Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm\n")
            f.write(f"Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}\n")
            f.write(f"Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N\n")
            from datetime import datetime
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        print(f"💾 Summary saved to: {summary_filename}")

        # Save plot in corresponding plots folder
        # Mirror the run folder structure
        # e.g. results/measurements/run_001/well_A6_xxx.csv -> results/plots/run_001/well_A6_analysis.png
        run_folder = None
        for part in data_dir.split(os.sep):
            if part.startswith("run_"):
                run_folder = part
                break
        if run_folder:
            plots_dir = os.path.join("results", "plots", run_folder)
            os.makedirs(plots_dir, exist_ok=True)
            plot_filename = os.path.join(plots_dir, f"{well}_analysis.png")
            analyzer.plot_results(result, save_plot=True, run_folder=run_folder)
            # Move the plot to the correct location if needed
            import shutil
            # Find the most recent plot file in results/plots/run_*
            from glob import glob
            import time
            plot_candidates = glob(os.path.join("results", "plots", "run_*", f"{well}_analysis.png"))
            if plot_candidates:
                latest_plot = max(plot_candidates, key=os.path.getctime)
                shutil.move(latest_plot, plot_filename)
                print(f"💾 Plot saved to: {plot_filename}")
        else:
            analyzer.plot_results(result, save_plot=True, run_folder=run_folder)
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
