"""
ASMI Data Analysis v2 - Elastic Modulus Calculation from Indentation Data
Hertzian contact mechanics with robust contact detection and plotting wrappers.

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import csv
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import namedtuple

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from .version import __version__

FitResult = namedtuple("FitResult", ["params", "covariance", "model_func"])


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
    spring_constant: Optional[float] = None
    linear_fit_quality: Optional[float] = None
    linear_intercept: Optional[float] = None
    corrected_depths: Optional[list] = None  # For Hertzian fits with system compliance correction
    original_elastic_modulus: Optional[float] = None  # Original E before system correction
    original_fit_quality: Optional[float] = None  # Original R² before system correction
    depth_full: Optional[list] = None  # For plotting: 0 to max_depth (when min_depth > 0)
    forces_full: Optional[list] = None  # For plotting: forces corresponding to depth_full


class IndentationAnalyzer:
    """Analyze indentation data to calculate elastic modulus using Hertzian fitting."""
    
    # Physical constants
    SPHERE_RADIUS = 0.0025  # m
    SPHERE_E = 1.8e11       # Pa
    SPHERE_NU = 0.28
    
    # Analysis parameters
    INDENTATION_DEPTH_THRESHOLD = 2.5  # mm
    FORCE_THRESHOLD = 2.0              # N
    FORCE_LIMIT = 25.0                 # N (filtering)
    RETROSPECTIVE_THRESHOLD = 1.0    # N (retrospective contact detection) (13N for measuring the spring constant of the system)
    
    # System compliance correction for Hertzian fitting
    K_SYSTEM = 64.27  # N/mm - system stiffness for depth correction
    
     # Well geometry constants
    WELL_DEPTH = 10.9  # mm
    WELL_TOP_Z = -9.0  # mm
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.data = None
        self._spring_constant_map = None  # Cache for well-specific spring constants
    
    # ----------------------------- Data loading -----------------------------
    def load_data(self, filename: str) -> bool:
        try:
            filepath = os.path.join(self.data_dir, filename)
            with open(filepath, "r") as f:
                reader = csv.reader(f)
                self.data = [r for r in reader if r]
            print(f"✅ Loaded {len(self.data)} rows from {filename}")
            return True
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return False
    
    # ----------------------- Contact detection methods ----------------------
    def find_contact_point(self, raw_forces: List[float], baseline: float, baseline_std: float) -> int:
        if len(raw_forces) < 10:
            return 0
        corrected = [f - baseline for f in raw_forces]

        # Strategy 1: sustained abs(force) above threshold
        thr = self.FORCE_THRESHOLD
        for i in range(len(corrected) - 5):
            if all(abs(corrected[j]) > thr for j in range(i, i + 5)):
                return i
        
        # Strategy 2: smoothed force
        w = 3
        smoothed = []
        for i in range(len(corrected)):
            s = max(0, i - w // 2)
            e = min(len(corrected), i + w // 2 + 1)
            smoothed.append(sum(corrected[s:e]) / (e - s))
        for i in range(len(smoothed) - 3):
            if all(abs(smoothed[j]) > thr for j in range(i, i + 3)):
                return i
        
        # Strategy 3: derivative
        deriv = [corrected[i] - corrected[i - 1] for i in range(1, len(corrected))]
        for i in range(len(deriv) - 3):
            if all(deriv[j] > 0.1 for j in range(i, i + 3)):
                return i
        
        # Fallback: simple threshold
        for i, v in enumerate(corrected):
            if abs(v) > thr:
                return i
        return 0

    def find_extraploation_contact_point(
        self,
        z_positions: List[float],
        raw_forces: List[float],
        baseline: float,
        baseline_std: float,
        extrapolation_threshold: float = 1.0 / 3.0,
        min_consecutive_points: int = 5,
    ) -> int:
        """Extrapolate back to f=0 using linear fit on points above a fraction of max force."""
        if not z_positions or not raw_forces or len(z_positions) != len(raw_forces):
            return 0
        corrected = [f - baseline for f in raw_forces]
        abs_f = [abs(v) for v in corrected]
        if len(abs_f) < 3:
            return 0
        max_f = max(abs_f)
        if max_f <= 0:
            return 0

        thr = extrapolation_threshold * max_f
        thr_idx = None
        for i, v in enumerate(abs_f):
            if v >= thr:
                thr_idx = i
                break

        if thr_idx is not None:
            idxs = []
            for k in range(thr_idx, len(abs_f)):
                if abs_f[k] >= thr:
                    idxs.append(k)
                    if len(idxs) >= min_consecutive_points:
                        break
                else:
                    break
            if len(idxs) >= 2:
                zs = np.array([z_positions[k] for k in idxs])
                fs = np.array([corrected[k] for k in idxs])
                try:
                    m, b = np.polyfit(zs, fs, 1)
                    if abs(m) > 1e-12:
                        z0 = -b / m
                        nearest = min(range(len(z_positions)), key=lambda k: abs(z_positions[k] - z0))
                        print(f"🔍 Extrapolated f=0 at Z={z0:.3f}mm → idx {nearest}")
                        return nearest
                except Exception:
                    pass

        # Fallback to threshold detection then local extrapolation
        i0 = self.find_contact_point(raw_forces, baseline, baseline_std)
        if i0 < 0 or i0 >= len(raw_forces):
            return 0
        if (i0 + 2) < len(raw_forces):
            z1, f1 = z_positions[i0], corrected[i0]
            z2, f2 = z_positions[i0 + 2], corrected[i0 + 2]
            if abs(z2 - z1) > 1e-9:
                m = (f2 - f1) / (z2 - z1)
                b = f1 - m * z1
                if abs(m) > 1e-12:
                    z0 = -b / m
                    nearest = min(range(len(z_positions)), key=lambda k: abs(z_positions[k] - z0))
                    print(f"🔍 Fallback extrapolated f=0 at Z={z0:.3f}mm → idx {nearest}")
                    return nearest
        return i0

    # Backward-compatibility wrapper
    def find_true_contact_point(
        self, z_positions: List[float], raw_forces: List[float], baseline: float, baseline_std: float
    ) -> int:
        return self.find_extraploation_contact_point(z_positions, raw_forces, baseline, baseline_std)

    def find_baseline_threshold_contact_point(
        self, raw_forces: List[float], baseline: float, baseline_std: float
    ) -> int:
        """Find contact using original KABlab threshold: threshold = -baseline + 2*baseline_std.
        No contact when raw_force > threshold; contact when raw_force <= threshold.
        Returns first contact index = last no-contact index + 1.
        """
        if len(raw_forces) < 2:
            return 0
        threshold = -baseline + 2 * baseline_std
        last_no_contact = -1
        for i, f in enumerate(raw_forces):
            if f > threshold:
                last_no_contact = i
        first_contact_idx = last_no_contact + 1
        if first_contact_idx >= len(raw_forces):
            first_contact_idx = 0
        return first_contact_idx

    def find_retrospective_contact_point(self, corrected_forces: List[float], threshold: float, z_positions: List[float], use_gradient: bool = False) -> int:
        if len(corrected_forces) < 3:
            return 0
        max_depth_idx = max(range(len(corrected_forces)), key=lambda i: abs(z_positions[i]))

        # Threshold-only mode: walk backward and return the first point under threshold
        if not use_gradient:
            for i in range(max_depth_idx, 0, -1):
                cur = abs(corrected_forces[i])
                if cur <= threshold:
                    return i
            return 0

        # Gradient mode: detect a turning point that sustains for the next two steps (g1 vs g2, with g2 == sign g3)
        # Ensure we don't index out of range when looking ahead
        for i in range(min(max_depth_idx, len(corrected_forces) - 4), 0, -1):
            cur = abs(corrected_forces[i])
            # Forward differences from i
            f1 = abs(corrected_forces[i + 1])
            f2 = abs(corrected_forces[i + 2])
            f3 = abs(corrected_forces[i + 3])
            g1 = f1 - cur
            g2 = f2 - f1
            g3 = f3 - f2
            if np.sign(g1) != np.sign(g2) and np.sign(g2) == np.sign(g3):
                return i
            # Fallback to threshold at this position
            if cur <= threshold:
                return i
        return 0
    
    # ------------------- Indentation depth and height calc ------------------
    def calculate_indentation_depth(
        self, z_positions: List[float], first_contact_idx: int, corrected_forces: Optional[List[float]] = None
    ) -> Tuple[List[float], float, List[float]]:
        """
        Calculate indentation depths relative to contact point and zero forces at contact.
        
        Implements "Force Zeroing" to satisfy Hertzian requirement: F(0) = 0.
        This eliminates S-shape fitting bias by ensuring force is zero at zero indentation depth.
        
        Args:
            z_positions: List of absolute Z positions
            first_contact_idx: Index where contact is detected
            corrected_forces: Baseline-corrected forces (optional)
            
        Returns:
            Tuple of (depths, zc, forces) where:
            - depths: Relative indentation depths (mm), starting at 0.0 at contact
            - zc: Absolute Z position of contact point (mm)
            - forces: Forces zeroed at contact point (N), ensuring F(0) = 0
        """
        zc = z_positions[first_contact_idx]  # Absolute contact position
        depths = [abs(z - zc) for z in z_positions[first_contact_idx:]]  # Relative depths from contact
        
        if corrected_forces is not None and len(corrected_forces) > first_contact_idx:
            # Force Zeroing: Subtract force at contact to ensure F(0) = 0
            f_at_contact = corrected_forces[first_contact_idx]
            forces = [f - f_at_contact for f in corrected_forces[first_contact_idx:]]
        else:
            forces = [0.0] * len(depths)
        
        return depths, zc, forces
    
    def calculate_approx_height(self, z_contact: float, well_bottom_z: float = -85.0) -> float:
        """Sample height = |z_contact - well_bottom_z| (distance from well bottom to contact/sample top)."""
        h = abs(z_contact - well_bottom_z)
        return max(0.1, min(h, 50.0)) # limit the height to 0.1-50.0 mm

    def calculate_approx_height_legacy(self, depths: List[float], step_mm: float = 0.02) -> float:
        """Original KABlab batch script formula: approx_height = 15 - (num*step + 3).
        num = index of minimum |depth|. Used for (b,c) lookup compatibility with original."""
        if not depths:
            return 10.0
        abs_depths = [abs(d) for d in depths]
        zero = min(abs_depths)
        num = abs_depths.index(zero)
        z_pos = (num * step_mm) + 3
        approx_height = 15 - z_pos
        return max(0.1, min(approx_height, 50.0))
    
    def _load_spring_constant_map(self) -> Dict[str, float]:
        """Load well-specific spring constants from CSV file in src folder.
        
        IMPORTANT: The CSV should contain SYSTEM-ONLY spring constants (k_system),
        measured on rigid surfaces (e.g., aluminum plate or well bottom).
        Do NOT use total spring constants (k_total = system + sample) measured on samples.
        
        Returns:
            Dictionary mapping well names to spring constant values (N/mm)
        """
        if self._spring_constant_map is not None:
            return self._spring_constant_map
        
        # Look for CSV file in src folder
        csv_path = os.path.join("src", "well_heatmap_spring_constant_data.csv")
        if not os.path.exists(csv_path):
            # Fallback to default K_SYSTEM for all wells
            print(f"⚠️ Spring constant CSV not found at {csv_path}, using default K_SYSTEM = {self.K_SYSTEM} N/mm")
            self._spring_constant_map = {}
            return self._spring_constant_map
        
        try:
            df = pd.read_csv(csv_path)
            
            # Check which column contains the spring constant
            # Prefer 'SpringConstant_k_Corrected' if available (system-only, corrected from total)
            # Otherwise use 'SpringConstant_k' (should be system-only measured on rigid surface)
            k_col = None
            for col in ['SpringConstant_k_Corrected', 'SpringConstant_k']:
                if col in df.columns:
                    k_col = col
                    break
            
            if k_col is None:
                print(f"⚠️ No spring constant column found in {csv_path}, using default K_SYSTEM = {self.K_SYSTEM} N/mm")
                self._spring_constant_map = {}
                return self._spring_constant_map
            
            # Build mapping from well name to spring constant
            spring_map = {}
            for _, row in df.iterrows():
                well = str(row.get('Well', '')).upper()
                k_val = row.get(k_col)
                
                if well and pd.notna(k_val) and k_val != '':
                    try:
                        k_float = float(k_val)
                        # Warn if value seems too low (might be total instead of system-only)
                        if k_float < 30.0:
                            print(f"⚠️ Well {well}: k={k_float:.2f} N/mm seems low. Ensure this is system-only (measured on rigid surface), not total (system+sample).")
                        spring_map[well] = k_float
                    except (ValueError, TypeError):
                        continue
            
            self._spring_constant_map = spring_map
            print(f"✅ Loaded {len(spring_map)} well-specific spring constants from {csv_path}")
            if len(spring_map) > 0:
                k_values = list(spring_map.values())
                print(f"📊 Spring constant range: {min(k_values):.2f} - {max(k_values):.2f} N/mm (mean: {np.mean(k_values):.2f} N/mm)")
            return spring_map
            
        except Exception as e:
            print(f"⚠️ Error loading spring constant CSV: {e}, using default K_SYSTEM = {self.K_SYSTEM} N/mm")
            self._spring_constant_map = {}
            return self._spring_constant_map
    
    def _get_spring_constant_for_well(self, well: str) -> float:
        """Get spring constant for a specific well.
        
        IMPORTANT: This should return the SYSTEM-ONLY spring constant (k_system),
        not the total spring constant (k_total = system + sample).
        
        Args:
            well: Well identifier (e.g., 'A1', 'B2')
            
        Returns:
            Spring constant in N/mm (uses well-specific value if available, otherwise default)
        """
        spring_map = self._load_spring_constant_map()
        
        # Normalize well name (remove _down/_up suffix if present)
        well_normalized = well.upper()
        if well_normalized.endswith('_DOWN'):
            well_normalized = well_normalized[:-5]
        elif well_normalized.endswith('_UP'):
            well_normalized = well_normalized[:-3]
        
        # Look up in map, fallback to default
        k_system = spring_map.get(well_normalized, self.K_SYSTEM)
        
        if well_normalized in spring_map:
            return k_system
        else:
            # Use default if not found
            return self.K_SYSTEM
    
    def diagnose_correction_issue(self, summary_csv: str) -> Dict:
        """Diagnose potential issues with system correction that might cause increased scatter.
        
        Checks if spring constants in CSV are appropriate for depth correction.
        
        Args:
            summary_csv: Path to summary CSV with elastic modulus data
            
        Returns:
            Dictionary with diagnostic information
        """
        import pandas as pd
        
        diagnostics = {
            'csv_found': False,
            'spring_constants_loaded': False,
            'k_system_values': [],
            'k_system_range': None,
            'k_system_mean': None,
            'potential_issue': None,
            'recommendation': None
        }
        
        # Check if spring constant CSV exists
        csv_path = os.path.join("src", "well_heatmap_spring_constant_data.csv")
        if os.path.exists(csv_path):
            diagnostics['csv_found'] = True
            spring_map = self._load_spring_constant_map()
            if spring_map:
                diagnostics['spring_constants_loaded'] = True
                k_values = list(spring_map.values())
                diagnostics['k_system_values'] = k_values
                diagnostics['k_system_range'] = (min(k_values), max(k_values))
                diagnostics['k_system_mean'] = np.mean(k_values)
                
                # Check for potential issues
                if min(k_values) < 30.0:
                    diagnostics['potential_issue'] = "Spring constants are very low (<30 N/mm). These might be total spring constants (system+sample) rather than system-only constants."
                    diagnostics['recommendation'] = "Ensure CSV contains system-only spring constants measured on rigid surfaces (e.g., aluminum plate or well bottom), not total spring constants measured on samples."
                elif max(k_values) - min(k_values) > 20.0:
                    diagnostics['potential_issue'] = f"Large variation in spring constants across wells (range: {max(k_values)-min(k_values):.2f} N/mm). This variation might be causing increased scatter."
                    diagnostics['recommendation'] = "Verify that spring constants are measured consistently. Large variations might indicate measurement errors or actual system compliance variations."
        
        # Check summary CSV for scatter metrics
        if os.path.exists(summary_csv):
            try:
                df = pd.read_csv(summary_csv)
                if 'ElasticModulus' in df.columns and 'ElasticModulus_Original' in df.columns:
                    valid = df.dropna(subset=['ElasticModulus', 'ElasticModulus_Original'])
                    valid = valid[(valid['ElasticModulus'] > 0) & (valid['ElasticModulus_Original'] > 0)]
                    
                    if len(valid) > 0:
                        orig = valid['ElasticModulus_Original'] / 1e6  # Convert to MPa
                        corr = valid['ElasticModulus'] / 1e6
                        
                        orig_cv = (np.std(orig) / np.mean(orig) * 100) if np.mean(orig) > 0 else 0
                        corr_cv = (np.std(corr) / np.mean(corr) * 100) if np.mean(corr) > 0 else 0
                        
                        diagnostics['original_cv'] = orig_cv
                        diagnostics['corrected_cv'] = corr_cv
                        diagnostics['cv_change'] = corr_cv - orig_cv
                        
                        if corr_cv > orig_cv:
                            diagnostics['scatter_increased'] = True
                            diagnostics['recommendation'] = (diagnostics.get('recommendation', '') + 
                                f"\n⚠️ Scatter increased after correction (CV: {orig_cv:.2f}% → {corr_cv:.2f}%). "
                                "This suggests the correction might be incorrect. Check if spring constants in CSV are system-only (measured on rigid surfaces) or total (measured on samples).")
            except Exception as e:
                diagnostics['error'] = str(e)
        
        return diagnostics
    
    def correct_depth_for_system_compliance(self, depths: np.ndarray, forces: np.ndarray, well: Optional[str] = None) -> np.ndarray:
        """Correct measured depths for system compliance: d_true = d_measure - force / k_system
        
        Uses well-specific spring constant from CSV if available, otherwise uses default K_SYSTEM.
        
        Args:
            depths: Measured indentation depths (mm)
            forces: Corresponding forces (N)
            well: Well identifier (e.g., 'A1', 'B2') for well-specific correction
            
        Returns:
            Corrected depths (mm)
        """
        if well is not None:
            k_system = self._get_spring_constant_for_well(well)
        else:
            k_system = self.K_SYSTEM
        
        return depths - forces / k_system

    def _get_force_correction_params(self, p_ratio: float, approx_height: float) -> Tuple[float, float]:
        """Return (b, c) for geometry correction: F_corrected = F_raw / (c * d^b).
        Based on simulation data for non-ideal sample shapes (finite height, well geometry).
        """
        # p_ratio bands: <0.325, [0.325,0.375), [0.375,0.425), [0.425,0.475), >=0.475
        # height bands: >=9.5, [8.5,9.5), [7.5,8.5), [6.5,7.5), [5.5,6.5), [4.5,5.5), [3.5,4.5), else
        def h_idx(h: float) -> int:
            if h >= 9.5: return 0
            if h >= 8.5: return 1
            if h >= 7.5: return 2
            if h >= 6.5: return 3
            if h >= 5.5: return 4
            if h >= 4.5: return 5
            if h >= 3.5: return 6
            return 7 # h < 3.5

        # Table: [p_band][h_idx] = (b, c)
        if p_ratio < 0.325:
            tbl = [(0.13, 1.24), (0.131, 1.24), (0.133, 1.25), (0.132, 1.24), (0.132, 1.24), (0.139, 1.27), (0.149, 1.3), (0.162, 1.38)] #tbl means table values, each tuple is (b, c) corresponding to the height and p_ratio band
        elif p_ratio < 0.375:
            tbl = [(0.132, 1.25), (0.132, 1.25), (0.134, 1.25), (0.136, 1.26), (0.126, 1.25), (0.133, 1.27), (0.144, 1.32), (0.169, 1.42)]
        elif p_ratio < 0.425:
            tbl = [(0.181, 1.33), (0.182, 1.34), (0.183, 1.34), (0.183, 1.34), (0.194, 1.38), (0.198, 1.4), (0.203, 1.44), (0.176, 1.46)]
        elif p_ratio < 0.475:
            tbl = [(0.156, 1.35), (0.152, 1.34), (0.156, 1.35), (0.161, 1.37), (0.153, 1.37), (0.166, 1.42), (0.179, 1.47), (0.205, 1.59)]
        else: # p_ratio >= 0.475
            tbl = [(0.203, 1.58), (0.207, 1.6), (0.212, 1.62), (0.217, 1.65), (0.21, 1.64), (0.22, 1.68), (0.17, 1.58), (0.182, 1.64)]
        b, c = tbl[h_idx(approx_height)]
        return b, c

    def correct_force_for_geometry(self, depths: np.ndarray, forces: np.ndarray, p_ratio: float, approx_height: float) -> np.ndarray:
        """Apply geometry correction: F_corrected = F_raw / (c * d^b). For Hertzian fit on non-ideal samples."""
        b, c = self._get_force_correction_params(p_ratio, approx_height)
        # Avoid division by zero for d near 0
        d_safe = np.maximum(depths, 0.01)
        return np.abs(forces) / (c * np.power(d_safe, b))

    # ---------------------- Material property detection ---------------------
    def detect_force_limit_reached(self, filename: str) -> Tuple[bool, float]:
        try:
            force_limit = self.FORCE_LIMIT
            final_force = None
            with open(filename, "r") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row and len(row) >= 2:
                        if row[0] == "Force_Limit(N)":
                            force_limit = float(row[1])
                        elif row[0] == "Force_Exceeded":
                            return row[1].lower() == "true", force_limit
                        elif len(row) >= 4 and row[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                            try:
                                final_force = abs(float(row[3]))
                            except Exception:
                                pass
            if final_force is not None:
                return final_force >= force_limit * 0.95, force_limit
            return False, force_limit
        except Exception as e:
            print(f"⚠️ Could not detect force limit status: {e}")
            return False, 45.0
    
    def determine_poisson_ratio(self, filename: str) -> Tuple[float, str]:
        reached, lim = self.detect_force_limit_reached(filename)
        if reached:
            print(f"🔍 Force limit reached ({lim}N) → glassy polymer (ν=0.3)")
            return 0.3, "glassy_polymer"
        print(f"🔍 Force limit not reached ({lim}N) → gel (ν=0.5)")
        return 0.5, "gel"

    # ----------------------- Hertzian model and fitting ---------------------
    def find_E(self, A: float, poisson_ratio: float) -> float:
        if A <= 0:
            raise ValueError(f"Fitted parameter A must be positive, got {A}")
        if not (0.1 <= poisson_ratio <= 0.5):
            raise ValueError(f"Poisson's ratio must be between 0.1 and 0.5, got {poisson_ratio}")

        R = self.SPHERE_RADIUS
        A_SI = A * (1000 ** 1.5) # Convert A to SI units (N/mm^1.5) to (N/m^1.5)
        E_star = (3.0 / 4.0) * A_SI / (R ** 0.5) # E* is the reduced elastic modulus
        # Include indenter contribution (full relationship)
        E_sample = E_star * (1 - poisson_ratio ** 2)
        E_inv = (1 - poisson_ratio ** 2) / E_sample - (1 - self.SPHERE_NU ** 2) / self.SPHERE_E
        return 1 / E_inv if E_inv != 0 else E_sample
    
    def adjust_E(self, E: float) -> float:
        SOFT_THRESHOLD = 660000  # Pa
        CORR_A = 457
        CORR_B = -0.457
        if E < SOFT_THRESHOLD:
            factor = CORR_A * pow(E, CORR_B)
            corrected = E / factor
            print(f"⚠️ Empirical correction applied: {E:.1f} → {corrected:.1f} Pa")
            return corrected
        return E
    
    def fit_hertz_model(self, depths: np.ndarray, forces: np.ndarray, bounds=None, raise_on_fail=False) -> FitResult:
        """
        Fit Hertzian contact model: F = A * (depth - d0)^1.5
        
        Optimized bounds for d0 parameter to eliminate S-shape fitting bias.
        With force zeroing, d0 should be close to 0, allowing fine-tuning of contact position.
        
        Args:
            depths: Indentation depths (mm), relative to contact point
            forces: Forces (N), zeroed at contact point (F(0) = 0)
            bounds: Optional custom bounds [(lb_A, lb_d0), (ub_A, ub_d0)]
            raise_on_fail: Whether to raise exception on fitting failure
            
        Returns:
            FitResult with fit parameters [A, d0] and covariance matrix
        """
        def hertz(depth, A, d0):
            return A * np.power(np.maximum(depth - d0, 0), 1.5)

        if len(depths) < 5 or len(forces) < 5:
            raise ValueError("Not enough data points for fitting.")
        if np.any(np.isnan(depths)) or np.any(np.isnan(forces)):
            raise ValueError("Input data contains NaNs.")
        
        # Optimized bounds: A > 0, d0 in [-0.5, 0.5] mm for fine-tuning contact position
        # Initial guess: A = 2, d0 = 0.0 (since forces are zeroed at contact)
        default_bounds = ([0.0, -0.5], [np.inf, 0.5])
        default_p0 = [2.0, 0.0]
        
        try:
            if bounds is not None:
                p, cov = curve_fit(hertz, depths, forces, p0=default_p0, bounds=bounds)
            else:
                p, cov = curve_fit(hertz, depths, forces, p0=default_p0, bounds=default_bounds)
            return FitResult(p, cov, hertz)
        except Exception as e:
            print(f"❌ Curve fitting failed: {e}")
            if raise_on_fail:
                raise
            return FitResult(None, None, hertz)

    def fit_linear_model(self, depths: np.ndarray, forces: np.ndarray, raise_on_fail=False) -> FitResult:
        """Fit a linear model F = k * d + b to the data for spring constant calculation.
        
        Args:
            depths: Array of indentation depths (mm)
            forces: Array of forces (N)
            raise_on_fail: Whether to raise exception on fitting failure
            
        Returns:
            FitResult with spring constant k, intercept b, and R² quality
        """
        def linear(depth, k, b):
            return k * depth + b

        if len(depths) < 2 or len(forces) < 2:
            if raise_on_fail:
                raise ValueError("Not enough data points for linear fitting (need at least 2).")
            return FitResult(None, None, linear)
        
        if np.any(np.isnan(depths)) or np.any(np.isnan(forces)):
            if raise_on_fail:
                raise ValueError("Input data contains NaNs.")
            return FitResult(None, None, linear)
        
        try:
            # Linear fit: F = k * d + b
            p, cov = curve_fit(linear, depths, forces, p0=[1.0, 0.0])
            k = float(p[0])
            b = float(p[1])
            
            # Calculate R²
            predicted = linear(depths, k, b)
            ss_res = np.sum((forces - predicted) ** 2)
            ss_tot = np.sum((forces - np.mean(forces)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            print(f"📏 Linear fit: F = {k:.3f}*d + {b:.3f} N, R² = {r2:.3f}")
            return FitResult(p, cov, linear)
            
        except Exception as e:
            print(f"❌ Linear fitting failed: {e}")
            if raise_on_fail:
                raise
            return FitResult(None, None, linear)

    # ----------------------------- Main analysis ----------------------------
    def analyze_well(
        self,
        well: str,
        poisson_ratio: Optional[float] = None,
        filename: Optional[str] = None,
        contact_method: str = "true_contact",
        fit_method: str = "hertzian",  # "hertzian" or "linear"
        apply_system_correction: bool = True,
        retrospective_threshold: Optional[float] = None,
        max_depth: float = 0.5,  # Maximum depth (mm) to use for analysis (default: 0.5 mm)
        min_depth: float = 0.25,  # Hertzian only: min depth; 0.25 = legacy 0.25–0.5 mm. Linear always uses 0–max_depth.
        apply_force_correction: bool = False,  # Hertzian only: geometry correction (F/(c*d^b)) before fit
        iterative_d0_refinement: bool = False,  # Hertzian only: KABlab iterative d0 refinement until |d0|<0.01 mm
        well_bottom_z: float = -85.0,  # Well bottom Z (mm); sample height = |contact_z - well_bottom_z|
        use_legacy_height: bool = False,  # Use original batch script approx_height formula for (b,c) lookup
        legacy_height_step_mm: float = 0.02,  # Step size (mm) for legacy height formula
    ) -> Optional[AnalysisResult]:
        print(f"\n🔬 Analyzing well {well}...")
        if poisson_ratio is None and filename:
            poisson_ratio, material_type = self.determine_poisson_ratio(filename) # auto-detect from file
        elif poisson_ratio is None:
            print("❌ Poisson's ratio must be provided if filename is not available")
            return None
        else:
            material_type = "manual"
        
        if self.data is None:
            print("❌ No data loaded")
            return None
        
        # Extract data rows and optional direction labels
        rows = []
        directions: List[Optional[str]] = []
        for r in self.data:
            if len(r) >= 4 and r[0].replace(".", "", 1).replace("-", "", 1).isdigit():
                rows.append(r)
                directions.append(r[4] if len(r) >= 5 and r[4] in ("down", "up") else None)
        if len(rows) < 10:
            print("❌ Not enough data points for analysis")
            return None
        
        z_positions = [float(r[1]) for r in rows]
        raw_forces = [float(r[2]) for r in rows]
        corrected_forces = [float(r[3]) for r in rows]

        # Extract baseline values from metadata
        baseline = 0.0
        baseline_std = 0.0
        for r in self.data:
            if len(r) >= 2:
                if r[0] == "Baseline_Force(N)":
                    baseline = float(r[1])
                elif r[0] == "Baseline_Std(N)":
                    baseline_std = float(r[1])
        
        # Choose contact detection method
        if contact_method in ("true_contact", "extrapolation"):
            first_idx = self.find_extraploation_contact_point(z_positions, raw_forces, baseline, baseline_std)
            label_method = "extrapolation"
        elif contact_method == "retrospective":
            thr = self.RETROSPECTIVE_THRESHOLD if retrospective_threshold is None else float(retrospective_threshold)
            first_idx = self.find_retrospective_contact_point(corrected_forces, threshold = thr, z_positions = z_positions)
            label_method = "retrospective"
        elif contact_method == "simple_threshold":
            first_idx = self.find_contact_point(raw_forces, baseline, baseline_std)
            label_method = "simple_threshold"
        elif contact_method == "baseline_threshold":
            first_idx = self.find_baseline_threshold_contact_point(raw_forces, baseline, baseline_std)
            label_method = "baseline_threshold"
        else:
            first_idx = self.find_extraploation_contact_point(z_positions, raw_forces, baseline, baseline_std)
            label_method = "extrapolation"

        if filename:
            # Derive run folder
            run_folder = None
            for part in filename.split(os.sep):
                if part.startswith("run_"):
                    run_folder = part
                    break
            # Direction label from file name if split
            dir_label = None
            low = filename.lower()
            if low.endswith("_down.csv"):
                dir_label = "down"
            elif low.endswith("_up.csv"):
                dir_label = "up"
            self.plot_contact_detection(
                z_positions,
                raw_forces,
                first_idx,
                well,
                save_plot=True,
                run_folder=run_folder,
                baseline=baseline,
                baseline_std=baseline_std,
                method=label_method,
                directions=directions,
                direction_label=dir_label,
            )

            # Export well, depth, force CSV to plot folder
            if run_folder:
                plots_dir = "results/plots"
                run_folder_plots = os.path.join(plots_dir, run_folder)
                os.makedirs(run_folder_plots, exist_ok=True)
                csv_path = os.path.join(run_folder_plots, f"{well}_depth_force.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    for i in range(len(z_positions)):
                        w.writerow([well, z_positions[i], raw_forces[i]])
                print(f"📄 Saved depth-force CSV: {csv_path}")

        # Depths and forces from contact
        depths, zc, forces_from_contact = self.calculate_indentation_depth(z_positions, first_idx, corrected_forces)

        # Filter to analysis range (min_depth/max_depth apply to Hertzian only; linear uses 0 to max_depth)
        d_max = max_depth
        is_hertzian = fit_method.lower() != "linear"
        if is_hertzian:
            d_min = min_depth
            if d_max <= d_min:
                d_max = max(d_min + 0.25, 0.5)
                print(f"⚠️ max_depth <= min_depth; using max_depth={d_max:.3f} mm")
            print(f"📏 Using depth range: {d_min:.3f}–{d_max:.3f} mm for analysis (Hertzian)")
        else:
            d_min = 0.0  # Linear: always use full range from contact
            print(f"📏 Using depth range: 0–{d_max:.3f} mm for analysis (Linear)")
        
        d_in, f_in = [], []
        d_full, f_full = [], []  # 0 to d_max for plotting when min_depth > 0 (Hertzian only)
        for d, f in zip(depths, forces_from_contact):
            if 0 <= d <= d_max:
                d_full.append(d)
                f_full.append(abs(f))
            if d_min <= d <= d_max:
                d_in.append(d)
                f_in.append(f)
        if len(d_in) < 5:
            print(f"❌ Not enough data points in analysis range ({d_min:.3f}–{d_max:.3f} mm)")
            return None
        
        # Remove last point if exceeds force limit
        if len(f_in) > 1 and abs(f_in[-1]) > self.FORCE_LIMIT:
            d_in = d_in[:-1]
            f_in = f_in[:-1]

        if use_legacy_height:
            approx_h = self.calculate_approx_height_legacy(depths, step_mm=legacy_height_step_mm)
            print(f"📐 Using legacy approx_height = {approx_h:.2f} mm (original batch formula)")
        else:
            approx_h = self.calculate_approx_height(zc, well_bottom_z)

        # Choose fitting method: Hertzian or Linear
        d_arr = np.array(d_in)
        f_arr = np.abs(np.array(f_in))

        # Apply geometry-based force correction for Hertzian (legacy KABlab method)
        if apply_force_correction and fit_method.lower() != "linear":
            f_arr = self.correct_force_for_geometry(d_arr, f_arr, poisson_ratio, approx_h)
            print(f"📐 Applied geometry force correction (b, c from simulation lookup)")

        # Full range (0 to d_max) for plotting when min_depth > 0
        depth_full_list = None
        forces_full_list = None
        if d_min > 0 and len(d_full) > 0 and fit_method.lower() != "linear":
            d_full_arr = np.array(d_full)
            f_full_arr = np.array(f_full)
            if apply_force_correction:
                f_full_arr = self.correct_force_for_geometry(d_full_arr, f_full_arr, poisson_ratio, approx_h)
            depth_full_list = list(d_full_arr)
            forces_full_list = list(f_full_arr)
        
        if fit_method.lower() == "linear":
            # Linear fitting: F = k * d
            print("📏 Using linear fitting (F = k * d)")
            linear_fit = self.fit_linear_model(d_arr, f_arr)
            if linear_fit.params is None:
                print("❌ Linear fitting failed")
                return None
        
            k = float(linear_fit.params[0])
            b = float(linear_fit.params[1])
            spring_constant = k
            linear_intercept = b
            
            # Calculate R² for linear fit
            predicted = k * d_arr + b
            ss_res = np.sum((f_arr - predicted) ** 2)
            ss_tot = np.sum((f_arr - np.mean(f_arr)) ** 2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            linear_r2 = r2
            
            # For linear fit, set Hertzian parameters to None/0
            E = 0  # No elastic modulus for linear fit
            E_unc = 0
            fit_A = 0
            fit_d0 = 0
            corrected_depths = None  # No depth correction for linear fit
            
        else:
            # Hertzian fitting (default)
            original_E = None
            original_r2 = None
            fit = None
            fit_A = 0.0
            fit_d0 = 0.0
            d_corrected = None
            d_arr_final = d_arr
            f_arr_final = f_arr

            if iterative_d0_refinement:
                # KABlab legacy: iterative d0 refinement until |d0| < 0.01 mm
                print("🔄 Using iterative d0 refinement (KABlab legacy)")
                depths_full = np.array(depths, dtype=float)
                forces_full = np.abs(np.array(forces_from_contact, dtype=float))
                lb, ub = [0.0, -0.5], [np.inf, 0.5]
                max_iter = 300
                for count in range(max_iter):
                    # Filter to depth range
                    mask = (depths_full >= d_min) & (depths_full <= d_max)
                    d_arr = depths_full[mask]
                    f_arr = forces_full[mask]
                    if len(d_arr) < 5:
                        print(f"❌ Iterative refinement: too few points at iter {count}")
                        fit = None
                        break
                    # Remove last point if exceeds force limit
                    if len(f_arr) > 1 and abs(f_arr[-1]) > self.FORCE_LIMIT:
                        d_arr = d_arr[:-1]
                        f_arr = f_arr[:-1]
                    if len(d_arr) < 5:
                        print(f"❌ Iterative refinement: too few points after filter at iter {count}")
                        fit = None
                        break
                    # Apply geometry force correction
                    if apply_force_correction:
                        f_arr = self.correct_force_for_geometry(d_arr, f_arr, poisson_ratio, approx_h)
                    # Apply system compliance
                    if apply_system_correction:
                        d_corr = self.correct_depth_for_system_compliance(d_arr, f_arr, well=well)
                    else:
                        d_corr = d_arr
                    # Fit
                    fit = self.fit_hertz_model(d_corr, f_arr, bounds=(lb, ub))
                    if fit.params is None:
                        print(f"❌ Iterative refinement: fit failed at iter {count}")
                        break
                    fit_A = float(fit.params[0])
                    fit_d0 = float(fit.params[1])
                    if abs(fit_d0) < 0.01:
                        print(f"✅ Iterative d0 refinement converged at iter {count + 1}")
                        d_corrected = d_corr
                        d_arr_final = d_arr.copy()
                        f_arr_final = f_arr.copy()
                        d_in = list(d_arr_final)
                        f_arr = f_arr_final
                        break
                    min_d0 = abs(fit_d0) if count == 0 else min(min_d0, abs(fit_d0))
                    if count > 0 and abs(round(old_d0, 5)) == abs(round(fit_d0, 5)):
                        fit_d0 = -0.75 * fit_d0
                    old_d0 = fit_d0
                    depths_full = depths_full - fit_d0
                    if count > 100 and count < 200 and abs(round(fit_d0, 2)) == round(min_d0, 2):
                        d_corrected = d_corr
                        d_arr_final = d_arr.copy()
                        f_arr_final = f_arr.copy()
                        d_in = list(d_arr_final)
                        f_arr = f_arr_final
                        break
                    if count >= 200 and count < 300 and abs(round(fit_d0, 1)) == round(min_d0, 1):
                        d_corrected = d_corr
                        d_arr_final = d_arr.copy()
                        f_arr_final = f_arr.copy()
                        d_in = list(d_arr_final)
                        f_arr = f_arr_final
                        break
                    if count == max_iter - 1:
                        print("⚠️ Iterative d0 refinement reached max iterations (300)")
                        d_corrected = d_corr
                        d_arr_final = d_arr.copy()
                        f_arr_final = f_arr.copy()
                        d_in = list(d_arr_final)
                        f_arr = f_arr_final
                if fit is None or fit.params is None:
                    print("❌ Hertzian fitting failed (iterative)")
                    return None
                if d_corrected is None:
                    d_corrected = d_corr
                    d_arr_final = d_arr.copy()
                    f_arr_final = f_arr.copy()
                    d_in = list(d_arr_final)
                    f_arr = f_arr_final
            elif apply_system_correction:
                print("🔬 Using Hertzian fitting with system compliance correction")
                # First, fit original (uncorrected) data to get original E
                # Use optimized bounds: d0 in [-0.5, 0.5] mm for fine-tuning contact position
                lb = [0.0, -0.5]
                ub = [np.inf, 0.5]
                fit_original = self.fit_hertz_model(d_arr, f_arr, bounds=(lb, ub))
                if fit_original.params is not None:
                    A_original = float(fit_original.params[0])
                    d0_original = float(fit_original.params[1])
                    original_E = round(self.adjust_E(self.find_E(A_original, poisson_ratio)))
                    # Calculate original R²
                    mask_orig = d_arr > d0_original
                    if np.sum(mask_orig) > 5:
                        vd_orig = d_arr[mask_orig]
                        vf_orig = f_arr[mask_orig]
                        pred_orig = A_original * (vd_orig - d0_original) ** 1.5
                        ss_res_orig = np.sum((vf_orig - pred_orig) ** 2)
                        ss_tot_orig = np.sum((vf_orig - np.mean(vf_orig)) ** 2)
                        original_r2 = 1 - (ss_res_orig / ss_tot_orig) if ss_tot_orig > 0 else 0
                    else:
                        original_r2 = 0
                    print(f"📊 Original (uncorrected) E = {original_E} Pa, R² = {original_r2:.3f}")
                
                # Apply system compliance correction: d_true = d_measure - force / k_system
                k_system_used = self._get_spring_constant_for_well(well)
                d_corrected = self.correct_depth_for_system_compliance(d_arr, f_arr, well=well)
                print(f"🔧 Applied system compliance correction (k_system = {k_system_used:.2f} N/mm for well {well})")
            else:
                print("🔬 Using Hertzian fitting (no system compliance correction)")
                d_corrected = d_arr

            if not iterative_d0_refinement:
                # Single fit (non-iterative)
                lb = [0.0, -0.5]
                ub = [np.inf, 0.5]
                fit = self.fit_hertz_model(d_corrected, f_arr, bounds=(lb, ub))
                if fit.params is None:
                    print("❌ Hertzian fitting failed")
                    return None
                fit_A = float(fit.params[0])
                fit_d0 = float(fit.params[1])
                d_arr_final = d_arr
                f_arr_final = f_arr

            E = round(self.adjust_E(self.find_E(fit_A, poisson_ratio)))
            if fit.covariance is not None:
                err = np.sqrt(np.diag(fit.covariance))
                E_unc = round(self.find_E(err[0], poisson_ratio))
            else:
                E_unc = 0

            # R^2 on valid region for Hertzian (use corrected depths)
            mask = d_corrected > fit_d0
            if np.sum(mask) > 5:
                vd = d_corrected[mask]
                vf = f_arr[mask]
                pred = fit_A * (vd - fit_d0) ** 1.5
                ss_res = np.sum((vf - pred) ** 2)
                ss_tot = np.sum((vf - np.mean(vf)) ** 2)
                r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            else:
                r2 = 0
            
            # No linear parameters for Hertzian fit
            spring_constant = None
            linear_r2 = None
            linear_intercept = None
            # Only populate corrected_depths when system correction is applied
            corrected_depths = list(d_corrected) if apply_system_correction else None

        result = AnalysisResult(
            well=well,
            elastic_modulus=E,
            uncertainty=E_unc,
            poisson_ratio=poisson_ratio,
            sample_height=float(round(approx_h, 1)),
            fit_quality=float(round(r2, 3)),
            depth_range=(min(d_in), max(d_in)),
            fit_A=fit_A,
            fit_d0=fit_d0,
            adjusted_forces=list(f_arr),
            depth_in_range=d_in,
            material_type=material_type,
            contact_z=float(round(zc, 3)),
            contact_force=float(round(corrected_forces[first_idx], 3)),
            spring_constant=spring_constant,
            linear_fit_quality=linear_r2,
            linear_intercept=linear_intercept,
            corrected_depths=corrected_depths if fit_method.lower() != "linear" else None,
            original_elastic_modulus=original_E if apply_system_correction and fit_method.lower() != "linear" else None,
            original_fit_quality=float(round(original_r2, 3)) if apply_system_correction and fit_method.lower() != "linear" and original_r2 is not None else None,
            depth_full=depth_full_list,
            forces_full=forces_full_list,
        )

        # Optional per-direction subset plots when direction info exists
        try:
            if filename and any((d in ("down", "up")) for d in directions if d is not None):
                post_z = z_positions[first_idx:]
                post_f = corrected_forces[first_idx:]
                post_dir = directions[first_idx:]
                z_down, f_down, z_up, f_up = [], [], [], []
                for z, f, d in zip(post_z, post_f, post_dir):
                    if d == "up":
                        z_up.append(z); f_up.append(abs(f))
                    elif d == "down":
                        z_down.append(z); f_down.append(abs(f))

                def plot_subset(z_sub, f_sub, dir_label: str):
                    if len(z_sub) < 5:
                        return
                    depths_sub = [abs(z - zc) for z in z_sub]
                    order = np.argsort(depths_sub)
                    d_sorted = np.array([depths_sub[i] for i in order])
                    f_sorted = np.array([f_sub[i] for i in order])
                    # Use the same depth range as the main analysis
                    mask = (d_sorted >= d_min) & (d_sorted <= d_max)
                    d_use = d_sorted[mask]
                    f_use = np.array(f_sorted[mask], dtype=float)
                    if len(d_use) < 5:
                        return
                    if apply_force_correction:
                        f_use = self.correct_force_for_geometry(np.array(d_use), f_use, poisson_ratio, approx_h)
                    sub_fit = self.fit_hertz_model(d_use, f_use, bounds=(lb, ub))
                    if sub_fit.params is None:
                        return
                    A_sub, d0_sub = float(sub_fit.params[0]), float(sub_fit.params[1])
                    r2_mask = d_use > d0_sub
                    if np.sum(r2_mask) > 5:
                        vd = d_use[r2_mask]
                        vf = f_use[r2_mask]
                        pred = A_sub * (vd - d0_sub) ** 1.5
                        ss_res = np.sum((vf - pred) ** 2)
                        ss_tot = np.sum((vf - np.mean(vf)) ** 2)
                        r2_sub = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
                    else:
                        r2_sub = 0
                    
                    subset = AnalysisResult(
                        well=well,
                        elastic_modulus=round(self.adjust_E(self.find_E(A_sub, poisson_ratio))),
                        uncertainty=0,
                        poisson_ratio=poisson_ratio,
                        sample_height=float(round(approx_h, 1)),
                        fit_quality=float(round(r2_sub, 3)),
                        depth_range=(float(d_use.min()), float(d_use.max())),
                        fit_A=A_sub,
                        fit_d0=d0_sub,
                        adjusted_forces=list(f_use),
                        depth_in_range=list(map(float, d_use)),
                        material_type=material_type,
                        contact_z=float(round(zc, 3)),
                        contact_force=float(round(corrected_forces[first_idx], 3)),
                    )
                    self.plot_results(
                        subset,
                        save_plot=True,
                        run_folder=self._extract_run_folder(filename),
                        method=label_method,
                        direction_label=dir_label,
                    )

                plot_subset(z_down, f_down, "down")
                plot_subset(z_up, f_up, "up")
        except Exception:
            pass

        return result

    # ------------------------------ Plotting API ----------------------------
    def plot_raw_data_all_wells(self, run_folder: str, save_plot: bool = True):
        from .plot import plotter
        plotter.plot_raw_data_all_wells(run_folder, save_plot)

    def plot_raw_force_individual_wells(self, run_folder: str, save_plot: bool = True):
        from .plot import plotter
        plotter.plot_raw_force_individual_wells(run_folder, save_plot)

    def plot_contact_detection(
        self,
        z_positions: List[float],
        raw_forces: List[float],
        contact_idx: int,
        well_name: str = "Unknown",
        save_plot: bool = True,
        run_folder: Optional[str] = None,
        baseline: float = 0.0,
        baseline_std: float = 0.0,
        method: str = "unknown",
        directions: Optional[List[str]] = None,
        direction_label: Optional[str] = None,
    ):
        from .plot import plotter
        plotter.plot_contact_detection(
            z_positions,
            raw_forces,
            contact_idx,
            well_name,
            save_plot,
            run_folder,
            baseline,
            baseline_std,
            method,
            directions,
            direction_label,
        )

    def plot_results(self, result: AnalysisResult, save_plot: bool = True, run_folder: Optional[str] = None, method: Optional[str] = None):
        from .plot import plotter
        try:
            plotter.plot_results(result, save_plot, run_folder, method)
        except TypeError:
            plotter.plot_results(result, save_plot, run_folder)

    # ------------------------------ Utilities -------------------------------
    def split_up_down_post_contact(
        self,
        z_positions: List[float],
        raw_forces: List[float],
        corrected_forces: List[float],
        directions: Optional[List[Optional[str]]],
        first_contact_idx: int,
        sort_up_by_abs_z: bool = True,
    ) -> Dict[str, Dict[str, List[float]]]:
        n = min(len(z_positions), len(raw_forces), len(corrected_forces))
        if not directions or len(directions) != n:
            return {
                "down": {
                    "z": list(z_positions[first_contact_idx:n]),
                    "raw": list(raw_forces[first_contact_idx:n]),
                    "corr": list(corrected_forces[first_contact_idx:n]),
                    "dir": ["down"] * max(0, n - first_contact_idx),
                },
                "up": {"z": [], "raw": [], "corr": [], "dir": []},
            }

        post_z = z_positions[first_contact_idx:n]
        post_raw = raw_forces[first_contact_idx:n]
        post_corr = corrected_forces[first_contact_idx:n]
        post_dir = directions[first_contact_idx:n]

        down = {"z": [], "raw": [], "corr": [], "dir": []}
        up = {"z": [], "raw": [], "corr": [], "dir": []}
        for z, rf, cf, d in zip(post_z, post_raw, post_corr, post_dir):
            if d == "up":
                up["z"].append(z); up["raw"].append(rf); up["corr"].append(cf); up["dir"].append("up")
            else:
                down["z"].append(z); down["raw"].append(rf); down["corr"].append(cf); down["dir"].append("down")

        if sort_up_by_abs_z and up["z"]:
            order = sorted(range(len(up["z"])), key=lambda i: abs(up["z"][i]))
            up = {
                "z": [up["z"][i] for i in order],
                "raw": [up["raw"][i] for i in order],
                "corr": [up["corr"][i] for i in order],
                "dir": [up["dir"][i] for i in order],
            }
        return {"down": down, "up": up}
    
    def calculate_uncertainty(self, covariance: np.ndarray, n_points: int) -> float:
        if covariance is not None and len(covariance) > 0:
            err = np.sqrt(np.diag(covariance))
            return float(err[0]) if len(err) > 0 else 0.0
        return 0.0
    
    def calculate_r_squared(self, y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
        if len(y_actual) == 0 or len(y_predicted) == 0:
            return 0.0
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        if ss_tot == 0:
            return 0.0
        return 1 - (ss_res / ss_tot)

    # Helper
    @staticmethod
    def _extract_run_folder(path: Optional[str]) -> Optional[str]:
        if not path:
            return None
        for part in path.split(os.sep):
            if part.startswith("run_"):
                return part
        return None


def main(contact_method: str = "retrospective"):
    import sys
    print("🔬 ASMI Indentation Analysis (v2)")
    if len(sys.argv) < 3:
        print("Usage: python analysis_2.py <datafile.csv> <well> [poisson_ratio]")
        return
    datafile = sys.argv[1]
    well = sys.argv[2].upper()
    p = None
    if len(sys.argv) >= 4:
        try:
            p = float(sys.argv[3])
            if not (0.3 <= p <= 0.5):
                raise ValueError("Poisson's ratio out of range")
        except ValueError as e:
            print(f"❌ Invalid Poisson's ratio: {e}")
            return
    data_dir, filename = os.path.split(datafile)
    analyzer = IndentationAnalyzer(data_dir or ".")
    if not analyzer.load_data(filename):
        return
    fullpath = os.path.join(data_dir, filename) if data_dir else filename
    res = analyzer.analyze_well(well, p, fullpath, contact_method=contact_method)
    if not res:
        print("❌ Analysis failed")
        return
    print(f"\n📊 Results for {res.well}: E={res.elastic_modulus} Pa, R²={res.fit_quality}")


if __name__ == "__main__":
    main("retrospective")