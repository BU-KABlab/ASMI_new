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
    RETROSPECTIVE_THRESHOLD = 0.05     # N (retrospective contact detection) (13N for measuring the spring constant of the system)
    
    # System compliance correction for Hertzian fitting
    K_SYSTEM = 16.09  # N/mm - system stiffness for depth correction
    
     # Well geometry constants
    WELL_DEPTH = 10.9  # mm
    WELL_TOP_Z = -9.0  # mm
    
    def __init__(self, data_dir: str = "."):
        self.data_dir = data_dir
        self.data = None
    
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
        zc = z_positions[first_contact_idx]
        depths = [abs(z - zc) for z in z_positions[first_contact_idx:]]
        forces = corrected_forces[first_contact_idx:] if corrected_forces is not None else [0.0] * len(depths)
        return depths, zc, forces
    
    def calculate_approx_height(self, z_contact: float) -> float:
        # Sample Height ≈ 10.9 + 8.5 - |z_contact| = 19.4 - |z_contact|
        h = 19.4 - abs(z_contact)
        return max(0.1, min(h, 50.0))
    
    def correct_depth_for_system_compliance(self, depths: np.ndarray, forces: np.ndarray) -> np.ndarray:
        """Correct measured depths for system compliance: d_true = d_measure - force / k_system
        
        Args:
            depths: Measured indentation depths (mm)
            forces: Corresponding forces (N)
            
        Returns:
            Corrected depths (mm)
        """
        return depths - forces / self.K_SYSTEM

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
    def find_E(self, A: float, p_ratio: float) -> float:
        if A <= 0:
            raise ValueError(f"Fitted parameter A must be positive, got {A}")
        if not (0.1 <= p_ratio <= 0.5):
            raise ValueError(f"Poisson's ratio must be between 0.1 and 0.5, got {p_ratio}")

        R = self.SPHERE_RADIUS
        A_SI = A * (1000 ** 1.5)
        E_star = (3.0 / 4.0) * A_SI / (R ** 0.5) # E* is the reduced elastic modulus
        # Include indenter contribution (full relationship)
        E_sample = E_star * (1 - p_ratio ** 2)
        E_inv = (1 - p_ratio ** 2) / E_sample - (1 - self.SPHERE_NU ** 2) / self.SPHERE_E
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
        def hertz(depth, A, d0):
            return A * np.power(np.maximum(depth - d0, 0), 1.5)

        if len(depths) < 5 or len(forces) < 5:
            raise ValueError("Not enough data points for fitting.")
        if np.any(np.isnan(depths)) or np.any(np.isnan(forces)):
            raise ValueError("Input data contains NaNs.")
        try:
            if bounds is not None:
                p, cov = curve_fit(hertz, depths, forces, p0=[2, 0.03], bounds=bounds)
            else:
                p, cov = curve_fit(hertz, depths, forces, p0=[2, 0.03])
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
    ) -> Optional[AnalysisResult]:
        print(f"\n🔬 Analyzing well {well}...")
        if poisson_ratio is None and filename:
            poisson_ratio, material_type = self.determine_poisson_ratio(filename)
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
            first_idx = self.find_retrospective_contact_point(corrected_forces, threshold = self.RETROSPECTIVE_THRESHOLD, z_positions = z_positions)
            label_method = "retrospective"
        elif contact_method == "simple_threshold":
            first_idx = self.find_contact_point(raw_forces, baseline, baseline_std)
            label_method = "simple_threshold"
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

        # Depths and forces from contact
        depths, zc, forces_from_contact = self.calculate_indentation_depth(z_positions, first_idx, corrected_forces)

        # Filter to analysis range
        d_in, f_in = [], []
        for d, f in zip(depths, forces_from_contact):
            if 0 <= d <= self.INDENTATION_DEPTH_THRESHOLD:
                d_in.append(d)
                f_in.append(f)
        if len(d_in) < 5:
            print("❌ Not enough data points in analysis range")
            return None
        
        # Remove last point if exceeds force limit
        if len(f_in) > 1 and abs(f_in[-1]) > self.FORCE_LIMIT:
            d_in = d_in[:-1]
            f_in = f_in[:-1]

        approx_h = self.calculate_approx_height(zc)

        # Choose fitting method: Hertzian or Linear
        d_arr = np.array(d_in)
        f_arr = np.abs(np.array(f_in))
        
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
            print("🔬 Using Hertzian fitting with system compliance correction")
            # Apply system compliance correction: d_true = d_measure - force / k_system
            d_corrected = self.correct_depth_for_system_compliance(d_arr, f_arr)
            print(f"🔧 Applied system compliance correction (k_system = {self.K_SYSTEM} N/mm)")
            lb = [0.0, -0.1]
            ub = [np.inf, 2.0]
            fit = self.fit_hertz_model(d_corrected, f_arr, bounds=(lb, ub))
            if fit.params is None:
                print("❌ Hertzian fitting failed")
                return None
            
            fit_A = float(fit.params[0])
            fit_d0 = float(fit.params[1])

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
            corrected_depths = list(d_corrected)  # Store corrected depths for plotting

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
                    mask = (d_sorted >= 0) & (d_sorted <= self.INDENTATION_DEPTH_THRESHOLD)
                    d_use = d_sorted[mask]
                    f_use = f_sorted[mask]
                    if len(d_use) < 5:
                        return
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
        from .Plot import plotter
        plotter.plot_raw_data_all_wells(run_folder, save_plot)

    def plot_raw_force_individual_wells(self, run_folder: str, save_plot: bool = True):
        from .Plot import plotter
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
        from .Plot import plotter
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
        from .Plot import plotter
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