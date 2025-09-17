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
    FORCE_LIMIT = 25.0         # N - force limit for filtering data
    
     # Well geometry constants
    WELL_DEPTH = 10.9 # mm - total depth of the well
    WELL_TOP_Z = -8.0  # mm - Z position when indenter is at well top
    
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
    
    def find_contact_point(self, raw_forces: List[float], baseline: float, baseline_std: float) -> int:
        """Find the first contact point using multiple detection strategies.
        
        Args:
            raw_forces: List of raw force values (not baseline-corrected)
            baseline: Baseline force value
            baseline_std: Standard deviation of baseline force
        Returns:
            Index of the first contact point
        """
        if len(raw_forces) < 10:
            return 0  # Not enough data points
        
        # Strategy 1: Sustained force above threshold (abs(corrected force) > threshold)
        corrected_forces = [f - baseline for f in raw_forces]
        threshold = self.FORCE_THRESHOLD
        
        for i in range(len(corrected_forces) - 5):
            # Check if we have 5 consecutive points above threshold (absolute value)
            if all(abs(corrected_forces[j]) > threshold for j in range(i, i + 5)):
                print(f"🔍 Contact detected (strategy 1) at index {i}: sustained force above threshold")
                return i
        
        # Strategy 2: Smoothed force detection
        window_size = 3
        smoothed_forces = []
        for i in range(len(corrected_forces)):
            start = max(0, i - window_size // 2)
            end = min(len(corrected_forces), i + window_size // 2 + 1)
            smoothed_forces.append(sum(corrected_forces[start:end]) / (end - start))
        
        for i in range(len(smoothed_forces) - 3):
            if all(abs(smoothed_forces[j]) > threshold for j in range(i, i + 3)):
                print(f"🔍 Contact detected (strategy 2) at index {i}: smoothed force above threshold")
                return i
        
        # Strategy 3: Force derivative detection
        force_derivatives = []
        for i in range(1, len(corrected_forces)):
            derivative = corrected_forces[i] - corrected_forces[i-1]
            force_derivatives.append(derivative)
        
        # Look for sustained positive derivatives
        for i in range(len(force_derivatives) - 3):
            if all(force_derivatives[j] > 0.1 for j in range(i, i + 3)):
                print(f"🔍 Contact detected (strategy 3) at index {i}: force derivative")
                return i
        
        # Strategy 4: Simple threshold (fallback) - using absolute value
        for i, force in enumerate(corrected_forces):
            if abs(force) > threshold:
                print(f"🔍 Contact detected (strategy 4) at index {i}: simple threshold (abs)")
                return i
        
        return 0  # Default to first point if no contact detected

    def find_true_contact_point(self, z_positions: List[float], raw_forces: List[float], baseline: float, baseline_std: float) -> int:
        """Find the true contact point by extrapolating back to zero force from the initial threshold contact point.
        
        Args:
            z_positions: List of Z positions
            raw_forces: List of raw force values
            baseline: Baseline force value
            baseline_std: Standard deviation of baseline force
        Returns:
            Index of the true contact point
        """
        # Find initial contact point using threshold detection
        initial_contact_idx = self.find_contact_point(raw_forces, baseline, baseline_std)
        
        if initial_contact_idx == 0:
            return initial_contact_idx
        
        # Get the force and position at initial contact
        initial_contact_force = raw_forces[initial_contact_idx] - baseline  # Corrected force
        initial_contact_z = z_positions[initial_contact_idx]
        
        print(f"🔍 Initial contact detected at index {initial_contact_idx}: Z={initial_contact_z:.3f}mm, Corrected_Force={initial_contact_force:.3f}N")
        
        # If the initial contact force is already very small, use it directly
        if abs(initial_contact_force) < 0.1:
            print(f"🔍 True contact point is the same as initial contact point (force already small)")
            return initial_contact_idx
        
        # Extrapolate back to find where force would be zero
        # Use linear extrapolation from the initial contact point
        # Find the next few points to establish a trend
        if initial_contact_idx + 3 < len(raw_forces):
            # Use points around the initial contact for extrapolation
            z1 = z_positions[initial_contact_idx]
            f1 = raw_forces[initial_contact_idx] - baseline
            
            z2 = z_positions[initial_contact_idx + 2]
            f2 = raw_forces[initial_contact_idx + 2] - baseline
            
            # Linear extrapolation: f = m*z + b
            if abs(z2 - z1) > 1e-6:  # Avoid division by zero
                m = (f2 - f1) / (z2 - z1)  # Slope
                b = f1 - m * z1  # Intercept
                
                # Find z where f = 0: 0 = m*z + b => z = -b/m
                if abs(m) > 1e-6:  # Avoid division by zero
                    true_contact_z = -b / m
                    
                    # Find the closest Z position to this extrapolated point
                    min_distance = float('inf')
                    true_contact_idx = initial_contact_idx
                    
                    for i, z in enumerate(z_positions):
                        distance = abs(z - true_contact_z)
                        if distance < min_distance:
                            min_distance = distance
                            true_contact_idx = i
                    
                    true_contact_force = raw_forces[true_contact_idx] - baseline
                    print(f"🔍 True contact extrapolated to index {true_contact_idx}: Z={z_positions[true_contact_idx]:.3f}mm, Corrected_Force={true_contact_force:.3f}N")
                    return true_contact_idx
        
        # If extrapolation fails, return the initial contact point
        print(f"🔍 Using initial contact point as true contact point (extrapolation failed)")
        return initial_contact_idx
    
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
        
        Sample Height = 10.9 (well depth) + 8.5 (abs value of initial Z position at surface) - abs(z_contact)
        Where:
        - Well depth = 10.9mm (WELL_DEPTH)
        - Initial Z position at surface = -8.5mm (WELL_TOP_Z)
        - Contact Z position = z_contact (negative value)
        
        Args:
            z_contact: Z position at first contact (negative value)
        Returns:
            Approximate height in mm
        """
        # Sample Height = 10.9 + 8.5 - abs(z_contact) = 19.4 - abs(z_contact)
        approx_height = 19.4 - abs(z_contact)
        
        # Ensure height is within reasonable bounds (0.1mm to 50mm)
        approx_height = max(0.1, min(approx_height, 50.0))
        return approx_height
    
    def adjust_force(self, depths: List[float], forces: List[float], p_ratio: float, approx_height: float) -> List[float]:
        """Add simulation-based adjustment factor since samples are not ideal shapes.
        Only applies adjustment to depths within the analysis range (0.1-1.0mm).
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
            # Only adjust forces for depths within the analysis range (0.1-1.0mm)
            if 0.1 <= depths[i] <= 1.0:
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
            force_limit = self.FORCE_LIMIT  # Default force limit
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
        first_contact_idx = self.find_contact_point(corrected_forces, 0.0, 0.0) # Pass dummy values for baseline/std
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
        
        # Extract Z positions, raw forces, and corrected forces
        z_positions = [float(row[1]) for row in data_rows]  # Z_Position(mm)
        raw_forces = [float(row[2]) for row in data_rows]  # Raw_Force(N)
        corrected_forces = [float(row[3]) for row in data_rows]  # Corrected_Force(N)
        
        # Extract baseline and baseline_std from metadata
        baseline = 0.0
        baseline_std = 0.0
        for row in self.data:
            if len(row) >= 2:
                if row[0] == 'Baseline_Force(N)':
                    baseline = float(row[1])
                elif row[0] == 'Baseline_Std(N)':
                    baseline_std = float(row[1])
        
        # Find true contact point using new criteria
        first_contact_idx = self.find_true_contact_point(z_positions, raw_forces, baseline, baseline_std)
        print(f"🔍 True contact detected at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm, Raw_Force={raw_forces[first_contact_idx]:.3f}N")
        
        # Generate contact detection plot for validation
        if filename:
            # Extract run folder name from filename for plot organization
            run_folder = None
            for part in filename.split(os.sep):
                if part.startswith("run_"):
                    run_folder = part
                    break
            
            self.plot_contact_detection(z_positions, raw_forces, first_contact_idx, well, save_plot=True, run_folder=run_folder, baseline=baseline, baseline_std=baseline_std)
        
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
    
    def analyze_well_original_method(self, depths: List[float], forces: List[float], p_ratio: float, well_name: str = "Unknown") -> Optional[Dict]:
        """Analyze well data using the original script's comprehensive method.
        
        This implements the complete analysis pipeline from the original script:
        1. Calculate approximate sample height
        2. Apply force corrections
        3. Perform iterative depth adjustment
        4. Calculate elastic modulus
        5. Apply empirical corrections
        
        Args:
            depths: List of indentation depths
            forces: List of forces
            p_ratio: Poisson ratio
            well_name: Name of the well
        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        if len(depths) < 10:
            print(f"❌ Not enough data points for well {well_name}")
            return None
        
        # Calculate approximate sample height (assuming contact at first depth)
        if depths:
            approx_height = self.calculate_approx_height(-depths[0])  # Convert depth to Z position
        else:
            approx_height = 5.0  # Default height
        
        print(f"📏 Approximate sample height: {approx_height:.1f} mm")
        
        # Get data in analysis range
        depth_in_range, force_in_range = self.find_d_and_f_in_range(depths, forces)
        
        if len(depth_in_range) < 5:
            print(f"❌ Not enough data points in analysis range for well {well_name}")
            return None
        
        # Check force range
        if max(force_in_range) - min(force_in_range) < 0.04:
            print(f"❌ Force range too small for well {well_name}")
            return None
        
        # Perform iterative depth adjustment
        print(f"🔄 Performing iterative depth adjustment for well {well_name}...")
        final_A, final_d0, converged = self.iterative_depth_adjustment(
            depths, forces, p_ratio, approx_height
        )
        
        if not converged:
            print(f"⚠️ Depth adjustment did not converge for well {well_name}")
        
        # Calculate elastic modulus
        E = self.find_E(final_A, p_ratio)
        E = self.adjust_E(E)
        E = round(E)
        
        # Calculate uncertainty (simplified)
        uncertainty = round(E * 0.1)  # 10% uncertainty estimate
        
        # Check depth range
        max_depth = max(depth_in_range)
        if max_depth < 0.4:
            print(f"⚠️ Sample was not indented far enough (max depth: {max_depth:.2f} mm)")
        
        # Prepare results
        results = {
            'well': well_name,
            'elastic_modulus': E,
            'uncertainty': uncertainty,
            'poisson_ratio': p_ratio,
            'sample_height': approx_height,
            'fit_A': final_A,
            'fit_d0': final_d0,
            'converged': converged,
            'depth_range': (min(depth_in_range), max(depth_in_range)),
            'force_range': (min(force_in_range), max(force_in_range)),
            'data_points': len(depth_in_range)
        }
        
        print(f"✅ Analysis complete for well {well_name}: E = {E} Pa")
        return results
    
    def find_d_and_f_in_range(self, depths: List[float], forces: List[float], min_depth: float = 0.1, max_depth: float = 1.0) -> Tuple[List[float], List[float]]:
        """Select data within specified depth range for analysis.
        
        Args:
            depths: List of indentation depths
            forces: List of forces
            min_depth: Minimum depth for analysis
            max_depth: Maximum depth for analysis
        Returns:
            Tuple of (depths_in_range, forces_in_range)
        """
        depths_in_range = []
        forces_in_range = []
        
        for i in range(len(depths)):
            if min_depth <= depths[i] <= max_depth:
                depths_in_range.append(depths[i])
                forces_in_range.append(forces[i])
        
        return depths_in_range, forces_in_range
    
    def iterative_depth_adjustment(self, depths: List[float], forces: List[float], p_ratio: float, approx_height: float, max_iterations: int = 100, d0_tolerance: float = 0.01) -> Tuple[float, float, bool]:
        """Perform iterative depth adjustment to refine contact point.
        
        This implements the iterative algorithm from the original script:
        1. Fit Hertzian model to current data
        2. Extract d0 parameter
        3. Adjust depths by subtracting d0
        4. Repeat until |d0| < tolerance
        
        Args:
            depths: List of indentation depths
            forces: List of forces
            p_ratio: Poisson ratio
            approx_height: Approximate sample height
            max_iterations: Maximum number of iterations
            d0_tolerance: Tolerance for d0 convergence
        Returns:
            Tuple of (final_A, final_d0, converged)
        """
        current_depths = depths.copy()
        current_forces = forces.copy()
        
        # Get data in analysis range
        depth_in_range, force_in_range = self.find_d_and_f_in_range(current_depths, current_forces)
        
        if len(depth_in_range) < 5:
            print("❌ Not enough data points in analysis range")
            return 0.0, 0.0, False
        
        # Apply force correction
        adjusted_forces = self.adjust_force(depth_in_range, force_in_range, p_ratio, approx_height)
        
        # Convert to numpy arrays for fitting
        depth_array = np.array(depth_in_range)
        force_array = np.array(adjusted_forces)
        
        # Define Hertzian function
        def hertz_func(depth, A, d0):
            return A * (depth - d0) ** 1.5
        
        # Initial fit
        try:
            parameters, covariance = curve_fit(hertz_func, depth_array, force_array, p0=[2, 0.03])
            fit_A = parameters[0]
            fit_d0 = parameters[1]
        except:
            print("❌ Initial fit failed")
            return 0.0, 0.0, False
        
        # Iterative refinement
        count = 0
        min_d0 = 100
        converged = False
        
        while count < max_iterations and not converged:
            count += 1
            old_d0 = fit_d0
            
            # Adjust depths
            current_depths = [d - fit_d0 for d in current_depths]
            
            # Get new data in range
            depth_in_range, force_in_range = self.find_d_and_f_in_range(current_depths, current_forces)
            
            if len(depth_in_range) < 5:
                print("❌ Not enough data points after adjustment")
                return 0.0, 0.0, False
            
            # Recalculate force correction with new depths
            adjusted_forces = self.adjust_force(depth_in_range, force_in_range, p_ratio, approx_height)
            
            # Convert to numpy arrays
            depth_array = np.array(depth_in_range)
            force_array = np.array(adjusted_forces)
            
            # Fit again
            try:
                parameters, covariance = curve_fit(hertz_func, depth_array, force_array, p0=[2, 0.03])
                fit_A = parameters[0]
                fit_d0 = parameters[1]
            except:
                print(f"❌ Fit failed at iteration {count}")
                return 0.0, 0.0, False
            
            # Check convergence
            if abs(fit_d0) < min_d0:
                min_d0 = abs(fit_d0)
            
            if abs(fit_d0) < d0_tolerance:
                converged = True
                break
            
            # Handle convergence issues
            if abs(round(old_d0, 5)) == abs(round(fit_d0, 5)):
                fit_d0 = -0.75 * fit_d0
            elif count > 100 and count < 200:
                if abs(round(fit_d0, 2)) == round(min_d0, 2):
                    break
            elif count >= 200 and count < 300:
                if abs(round(fit_d0, 1)) == round(min_d0, 1):
                    break
            elif count >= 300:
                print("❌ Maximum iterations reached without convergence")
                return 0.0, 0.0, False
        
        return fit_A, fit_d0, converged
        
    # === PLOTTING FUNCTIONS ===
    
    def plot_raw_data_all_wells(self, run_folder: str, save_plot: bool = True):
        """Plot raw data (absolute values) for all wells in a single plot"""
        from .plot import plotter
        plotter.plot_raw_data_all_wells(run_folder, save_plot)

    def plot_raw_force_individual_wells(self, run_folder: str, save_plot: bool = True):
        """Generate individual raw force plots for each well in a run"""
        from .plot import plotter
        plotter.plot_raw_force_individual_wells(run_folder, save_plot)

    def plot_contact_detection(self, z_positions: List[float], raw_forces: List[float], contact_idx: int, well_name: str = "Unknown", save_plot: bool = True, run_folder: Optional[str] = None, baseline: float = 0.0, baseline_std: float = 0.0):
        """Plot force data with contact point highlighted for validation using new detection criteria"""
        from .plot import plotter
        plotter.plot_contact_detection(z_positions, raw_forces, contact_idx, well_name, save_plot, run_folder, baseline, baseline_std)

    def plot_results(self, result: AnalysisResult, save_plot: bool = True, run_folder: Optional[str] = None):
        """Plot analysis results with corrected data and fit, and save to results/plots/"""
        from .plot import plotter
        plotter.plot_results(result, save_plot, run_folder)




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
        # Calculate total indentation depth from contact point to max depth
        total_indentation_depth = result.depth_range[1]  # Max depth in analysis range
        print(f"   Total Indentation Depth: {total_indentation_depth:.3f} mm")
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
            # Calculate total indentation depth from contact point to max depth
            total_indentation_depth = result.depth_range[1]  # Max depth in analysis range
            f.write(f"Total Indentation Depth: {total_indentation_depth:.3f} mm\n")
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
            analyzer.plot_results(result, save_plot=True, run_folder=run_folder)
        else:
            analyzer.plot_results(result, save_plot=True, run_folder=run_folder)
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()

    def analyze_well_with_contact_method(self, well: str, contact_method: str = "true_contact", poisson_ratio: Optional[float] = None, filename: Optional[str] = None, contact_force_threshold: float = 2.0, retrospective_threshold: float = 0.05) -> Optional[AnalysisResult]:
        """
        Complete analysis for a single well with selectable contact detection method.
        
        Args:
            well: Well identifier
            contact_method: Contact detection method:
                - "true_contact": Use find_true_contact_point (default sophisticated method)
                - "simple_threshold": Use simple force threshold
                - "retrospective": Use retrospective contact detection (find lowest force point)
                - "metadata": Use contact point from CSV metadata (if available)
            poisson_ratio: Poisson's ratio (auto-determined if None)
            filename: CSV file path for auto-determining Poisson's ratio
            contact_force_threshold: Force threshold for simple_threshold method
            retrospective_threshold: Force threshold for retrospective method
            
        Returns:
            AnalysisResult object or None if analysis fails
        """
        print(f"\n🔬 Analyzing well {well} using {contact_method} contact detection...")
        
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
        metadata = {}
        for row in self.data:
            if len(row) >= 2:
                # Collect metadata
                if len(row) == 2 and not row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                    metadata[row[0]] = row[1]
            # Data rows (those with numeric first column)
            if len(row) >= 4 and row[0].replace('.', '', 1).replace('-', '', 1).isdigit():
                data_rows.append(row)
        
        if len(data_rows) < 10:
            print("❌ Not enough data points for analysis")
            return None
        
        # Extract Z positions, raw forces, and corrected forces
        z_positions = [float(row[1]) for row in data_rows]  # Z_Position(mm)
        raw_forces = [float(row[2]) for row in data_rows]  # Raw_Force(N)
        corrected_forces = [float(row[3]) for row in data_rows]  # Corrected_Force(N)
        
        # Extract baseline and baseline_std from metadata
        baseline = float(metadata.get('Baseline_Force(N)', 0.0))
        baseline_std = float(metadata.get('Baseline_Std(N)', 0.1))
        
        # Choose contact detection method
        first_contact_idx = None
        
        if contact_method == "metadata":
            # Try to use contact point from metadata first
            if 'Contact_Z(mm)' in metadata:
                contact_z_meta = float(metadata['Contact_Z(mm)'])
                # Find the index closest to this Z position
                for i, z in enumerate(z_positions):
                    if abs(z - contact_z_meta) < 0.01:  # Within 0.01mm
                        first_contact_idx = i
                        print(f"🔍 Using metadata contact point at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm")
                        break
                if first_contact_idx is None:
                    print("⚠️ Metadata contact point not found in data, falling back to true_contact method")
                    contact_method = "true_contact"
            else:
                print("⚠️ No metadata contact point found, falling back to true_contact method")
                contact_method = "true_contact"
        
        if contact_method == "true_contact":
            # Use the sophisticated true contact detection
            first_contact_idx = self.find_true_contact_point(z_positions, raw_forces, baseline, baseline_std)
            print(f"🔍 True contact detected at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm, Raw_Force={raw_forces[first_contact_idx]:.3f}N")
        
        elif contact_method == "simple_threshold":
            # Use simple force threshold detection
            first_contact_idx = self.find_simple_contact_point(corrected_forces, contact_force_threshold)
            print(f"�� Simple threshold contact detected at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm, Force_threshold={contact_force_threshold}N")
        
        elif contact_method == "retrospective":
            # Use retrospective contact detection (find first point below threshold, going backwards)
            first_contact_idx = self.find_retrospective_contact_point(corrected_forces, retrospective_threshold)
            print(f"🔍 Retrospective contact detected at index {first_contact_idx}: Z={z_positions[first_contact_idx]:.3f}mm, Force_threshold={retrospective_threshold}N")
        
        if first_contact_idx is None:
            print("❌ Contact detection failed")
            return None
        
        # Generate contact detection plot for validation
        if filename:
            # Extract run folder name from filename for plot organization
            run_folder = None
            for part in filename.split(os.sep):
                if part.startswith("run_"):
                    run_folder = part
                    break
            
            self.plot_contact_detection(z_positions, raw_forces, first_contact_idx, well, save_plot=True, run_folder=run_folder, baseline=baseline, baseline_std=baseline_std)
        
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
        
        # Remove the last point if it exceeds force limit
        if len(force_in_range) > 1 and abs(force_in_range[-1]) > self.FORCE_LIMIT:
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
        uncertainty = self.calculate_uncertainty(covariance, len(depth_in_range)) if covariance is not None else E * 0.1
        uncertainty = round(uncertainty)
        
        # Calculate R-squared
        y_predicted = [hertz_func(d) for d in depths_array]
        r_squared = self.calculate_r_squared(adjusted_forces_array, y_predicted)
        
        # Return analysis result with contact method info
        return AnalysisResult(
            well=well,
            elastic_modulus=E,
            uncertainty=uncertainty,
            poisson_ratio=poisson_ratio,
            sample_height=approx_height,
            fit_quality=r_squared,
            depth_range=(min(depth_in_range), max(depth_in_range)),
            fit_A=fit_A,
            fit_d0=fit_d0,
            adjusted_forces=adjusted_forces,
            depth_in_range=depth_in_range,
            material_type=f"{material_type}_{contact_method}",  # Include contact method in material type
            contact_z=z_contact,
            contact_force=raw_forces[first_contact_idx]
        )

    def find_simple_contact_point(self, corrected_forces: List[float], threshold: float) -> int:
        """Find contact using simple force threshold detection"""
        for i, force in enumerate(corrected_forces):
            if abs(force) > threshold:
                return i
        return 0  # Default to first point if no contact detected

    def find_retrospective_contact_point(self, corrected_forces: List[float], threshold: float) -> int:
        """Find contact using retrospective detection (go backwards from end)"""
        # Start from the end and work backwards to find first point below threshold
        for i in range(len(corrected_forces) - 1, -1, -1):
            if abs(corrected_forces[i]) <= threshold:
                return i
        return 0  # Default to first point if all forces exceed threshold
