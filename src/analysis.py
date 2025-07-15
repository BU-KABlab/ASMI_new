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

@dataclass
class AnalysisResult:
    """Container for analysis results"""
    well: str
    elastic_modulus: float
    uncertainty: float
    poisson_ratio: float
    sample_height: float
    fit_quality: float
    depth_range: Tuple[float, float]
    fit_A: float
    fit_d0: float
    adjusted_forces: List[float]
    depth_in_range: List[float]

class IndentationAnalyzer:
    """Analyzes indentation data to calculate elastic modulus with full correction factors"""
    
    # Physical constants
    SPHERE_RADIUS = 0.0025  # m
    SPHERE_E = 1.8e11       # Pa
    SPHERE_NU = 0.28
    
    # Analysis parameters
    DEPTH_RANGE = (0.24, 0.5)  # mm
    MIN_FORCE_RANGE = 0.04     # N
    
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
    
    def get_available_files(self) -> List[str]:
        """Get list of available CSV files"""
        return [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
    
    def extract_well_data(self, well: str) -> Optional[List]:
        """Extract data for specific well"""
        if not self.data:
            print("❌ No data loaded")
            return None
        
        well_data = [row for row in self.data if row[0] == well]
        if not well_data:
            print(f"❌ No data found for well {well}")
            return None
        
        print(f"✅ Found {len(well_data)} measurements for well {well}")
        return well_data
    
    def collect_run_data(self, data: List, well: str, stiff: bool = False) -> List:
        """Collect data for specific run from csv file - from measure.py"""
        well_data = []
        no_contact = []
        run_array = []
        forces = []
        
        if stiff:
            return run_array
            
        for i in range(len(data)):
            if data[i][0] == well:  # collect data from most recent well
                values = [data[i][1], data[i][2]]
                well_data.append(values)
        
        for l in range(1, len(well_data)):
            if float(well_data[l][1]) <= -1*float(well_data[0][0]) + 2*float(well_data[0][1]):  # determine which measurements correspond to contact
                no_contact.append(l)
            run_array.append([well_data[l][0], well_data[l][1]])
        
        if len(run_array) - int(no_contact[len(no_contact) - 1]) <= 10:  # check if no or not enough data was collected for well
            print("Either well was not tested or no data was collected, either because sample was too short or too soft")
            run_array = []
            return run_array
            
        if len(no_contact) > 0:  # find index of first continuous contact measurement
            start_val = int(no_contact[len(no_contact)-1]+1)
        else:
            start_val = 0
            
        for k in range(len(run_array)):  # format data for analysis
            run_array[k][0] = round(-1*(float(run_array[k][0]) - float(well_data[start_val][0])), 2)  # set indentation depths relative to initial contact height
            run_array[k][1] = float(run_array[k][1]) + float(well_data[0][0])  # zero forces
            forces.append(run_array[k][1])
            
        if forces == [] or max(forces)-min(forces) < 0.04:  # check that force measurements were large enough to make proper measurement
            print("Either well was not tested or no data was collected, either because sample was too short or too soft")
            run_array = []
            return run_array
            
        return run_array
    
    def split(self, run_array: List) -> Tuple[List[float], List[float]]:
        """Splits data from well into separate depth and force arrays - from measure.py"""
        depths = []
        forces = []
        for i in range(len(run_array)):
            depths.append(run_array[i][0])
            forces.append(run_array[i][1])
        return depths, forces
    
    def find_d_and_f_in_range(self, run_array: List) -> Tuple[List[float], List[float]]:
        """Select data within desired depth range to determine elastic modulus - from measure.py"""
        forces = []
        depths = []
        for i in range(len(run_array)):
            if run_array[i][0] >= 0.24 and run_array[i][0] <= 0.5:  # .04, .3
                forces.append(run_array[i][1])
                depths.append(run_array[i][0])
        return depths, forces
    
    def approximate_height(self, run_array: List) -> float:
        """Find height of sample to determine correction equation used"""
        depths = []
        for i in range(len(run_array)):
            depths.append(run_array[i][0])
        for j in range(len(depths)):
            depths[j] = abs(depths[j])
        zero = min(depths)
        num = depths.index(zero)
        
        # For continuous reading starting from Z=0
        # Based on your actual measurements:
        # - Well depth: 10.9 mm
        # - Distance from indenter at Z=0 to top of well: 11.5 mm
        # - Total working distance: 11.5 + 10.9 = 22.4 mm
        WELL_DEPTH = 10.9  # mm - your actual well depth
        AIR_GAP = 11.5  # mm - distance from indenter at Z=0 to top of well
        TOTAL_WORKING_DISTANCE = AIR_GAP + WELL_DEPTH  # mm = 22.4 mm
        
        # Calculate Z position based on measurement index
        # Assuming continuous reading with small steps
        z_pos = (num * 0.02)  # Remove the +3 offset for Z=0 starting position
        approx_height = TOTAL_WORKING_DISTANCE - z_pos
        return approx_height
    
    def correct_force(self, depths: List[float], forces: List[float], p_ratio: float, approx_height: float) -> List[float]:
        """Add correction factor based on simulation data since samples are not ideal shapes - from measure.py"""
        new_array = []
        for i in range(len(depths)):
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
            val = (forces[i])/(c*pow(depths[i], b))
            new_array.append(val)
        return new_array
    
    def adjust_depth(self, run_array: List, d0: float) -> List:
        """Using curve fit, adjust depth so that at zero force, depth is 0 - from measure.py"""
        for i in range(len(run_array)):
            run_array[i][0] = run_array[i][0]-d0
        return run_array
    
    def find_E(self, A: float, p_ratio: float) -> float:
        """Determine elastic modulus from curve fit - from measure.py"""
        r_sphere = 0.0025
        sphere_p_ratio = 0.28
        sphere_E = 1.8 * pow(10, 11)
        polymer_p_ratio = p_ratio
        actual_A = A * pow(1000, 1.5)
        E_star = (actual_A * 0.75)/pow(r_sphere, 0.5)
        E_inv = 1/(E_star * (1 - pow(polymer_p_ratio, 2))) - (1 - pow(sphere_p_ratio, 2))/(sphere_E * (1 - pow(polymer_p_ratio, 2)))
        E_polymer = 1/E_inv
        return E_polymer
    
    def adjust_E(self, E: float) -> float:
        """An empirical correction factor for softer samples which causes issues with getting proper data at small indentation depths - from measure.py"""
        if E < 660000:
            factor = 457*pow(E, -0.457)
            E = E/factor
        return E
    
    def fit_hertz_model(self, depths: np.ndarray, forces: np.ndarray) -> Tuple:
        """Fit data to Hertzian contact model with iterative depth adjustment"""
        def hertz_func(depth, A, d0):
            return A * (depth - d0) ** 1.5
        
        try:
            params, covariance = curve_fit(hertz_func, depths, forces, p0=[2, 0.03])
            return params, covariance, hertz_func
        except Exception as e:
            print(f"❌ Curve fitting failed: {e}")
            return None, None, None
    
    def iterative_curve_fitting(self, run_array: List, p_ratio: float, height: float) -> Tuple[Optional[float], Optional[float], Optional[float], bool]:
        """Perform iterative curve fitting with depth adjustment - from measure.py logic"""
        depth_in_range, force_in_range = self.find_d_and_f_in_range(run_array)
        if not depth_in_range:
            return None, None, None, True
            
        adjusted_forces = self.correct_force(depth_in_range, force_in_range, p_ratio, height)
        depth_in_range = np.asarray(depth_in_range)
        adjusted_forces = np.asarray(adjusted_forces)
        
        # Initial fit
        params, covariance, hertz_func = self.fit_hertz_model(depth_in_range, adjusted_forces)
        if params is None:
            return None, None, None, True
            
        fit_A = float(params[0])
        fit_d0 = float(params[1])
        
        # Iterative adjustment
        count = 0
        continue_to_adjust = True
        if abs(fit_d0) < 0.01:
            continue_to_adjust = False
        min_d0 = 100
        error = False
        
        while continue_to_adjust:
            count = count + 1
            old_d0 = fit_d0
            run_array = self.adjust_depth(run_array, fit_d0)
            depth_in_range, force_in_range = self.find_d_and_f_in_range(run_array)
            if not depth_in_range:
                return None, None, None, True
                
            height = self.approximate_height(run_array)
            adjusted_forces = self.correct_force(depth_in_range, force_in_range, p_ratio, height)
            depth_in_range = np.asarray(depth_in_range)
            adjusted_forces = np.asarray(adjusted_forces)
            
            try:
                params, covariance = curve_fit(hertz_func, depth_in_range, adjusted_forces, p0=[2, 0.03])
            except:
                print("Data could not be analyzed")
                error = True
                break
            else:
                fit_A = float(params[0])
                fit_d0 = float(params[1])
                
                if abs(fit_d0) < min_d0:
                    min_d0 = abs(fit_d0)
                    
                if abs(round(old_d0, 5)) == abs(round(fit_d0, 5)):  # if fit continues to converge to improper value
                    fit_d0 = -0.75 * fit_d0
                elif abs(fit_d0) < 0.01:
                    continue_to_adjust = False
                    break
                elif count > 100 and count < 200:
                    if abs(round(fit_d0, 2)) == round(min_d0, 2):
                        break
                elif count >= 200 and count < 300:
                    if abs(round(fit_d0, 1)) == round(min_d0, 1):
                        break
                elif count == 300:
                    print("Error in data analysis")
                    error = True
                    break
        
        return fit_A, fit_d0, covariance, error
    
    def analyze_well(self, well: str, poisson_ratio: float) -> Optional[AnalysisResult]:
        """Complete analysis for a single well with full correction factors"""
        print(f"\n🔬 Analyzing well {well}...")
        
        # Extract and process data
        well_data = self.extract_well_data(well)
        if not well_data:
            return None
        
        # Use the comprehensive data collection from measure.py
        if self.data is None:
            print("❌ No data loaded")
            return None
        run_array = self.collect_run_data(self.data, well, stiff=False)
        if not run_array:
            print("❌ No valid data collected for well")
            return None
        
        # Get sample height
        height = self.approximate_height(run_array)
        
        # Perform iterative curve fitting
        fit_A, fit_d0, covariance, error = self.iterative_curve_fitting(run_array, poisson_ratio, height)
        if error or fit_A is None:
            print("❌ Analysis failed during curve fitting")
            return None
        
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
        
        # Get final data for plotting
        depth_in_range, force_in_range = self.find_d_and_f_in_range(run_array)
        adjusted_forces = self.correct_force(depth_in_range, force_in_range, poisson_ratio, height)
        
        # Calculate fit quality (R-squared)
        if depth_in_range and adjusted_forces:
            depths_array = np.array(depth_in_range)
            forces_array = np.array(adjusted_forces)
            predicted = fit_A * (depths_array - fit_d0) ** 1.5
            ss_res = np.sum((forces_array - predicted) ** 2)
            ss_tot = np.sum((forces_array - np.mean(forces_array)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        else:
            r_squared = 0
        
        # Check depth range
        if depth_in_range and round(max(depth_in_range), 2) < 0.4:
            print("⚠️ Sample was not indented far enough")
            print(f"The range the measurement was made with was {round(min(depth_in_range), 2)} mm to {round(max(depth_in_range), 2)} mm")
        
        return AnalysisResult(
            well=well,
            elastic_modulus=E,
            uncertainty=E_uncertainty,
            poisson_ratio=poisson_ratio,
            sample_height=round(height, 1),
            fit_quality=float(round(r_squared, 3)),
            depth_range=(min(depth_in_range) if depth_in_range else 0, max(depth_in_range) if depth_in_range else 0),
            fit_A=fit_A,
            fit_d0=fit_d0 if fit_d0 is not None else 0.0,
            adjusted_forces=adjusted_forces,
            depth_in_range=depth_in_range
        )
    
    def plot_results(self, result: AnalysisResult):
        """Plot analysis results with corrected data and fit"""
        if not result.depth_in_range or not result.adjusted_forces:
            print("❌ No data available for plotting")
            return
            
        plt.figure(figsize=(10, 6))
        
        # Plot corrected data
        plt.scatter(result.depth_in_range, result.adjusted_forces, alpha=0.6, label='Corrected Data')
        
        # Plot fit
        depths_array = np.array(result.depth_in_range)
        fit_forces = result.fit_A * (depths_array - result.fit_d0) ** 1.5
        plt.plot(depths_array, fit_forces, 'r-', label=f'Hertzian Fit (A={result.fit_A:.3f}, d0={result.fit_d0:.3f})')
        
        plt.xlabel('Indentation Depth (mm)')
        plt.ylabel('Force (N)')
        plt.title(f'Well {result.well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

def main():
    """Main analysis function"""
    print("🔬 ASMI Indentation Analysis (Full Version)")
    
    # Initialize analyzer
    analyzer = IndentationAnalyzer()
    
    # Show available files
    files = analyzer.get_available_files()
    if not files:
        print("❌ No CSV files found in current directory")
        return
    
    print(f"📁 Available files: {files}")
    
    # Load data
    filename = input("Enter filename (without .csv): ") + ".csv"
    if not analyzer.load_data(filename):
        return
    
    # Get well and Poisson's ratio
    well = input("Enter well (e.g., A1): ").upper()
    try:
        poisson_ratio = float(input("Enter Poisson's ratio (0.3-0.5): "))
        if not (0.3 <= poisson_ratio <= 0.5):
            raise ValueError("Poisson's ratio out of range")
    except ValueError as e:
        print(f"❌ Invalid input: {e}")
        return
    
    # Analyze
    result = analyzer.analyze_well(well, poisson_ratio)
    if result:
        print(f"\n📊 Results for Well {result.well}:")
        print(f"   Elastic Modulus: {result.elastic_modulus} Pa")
        print(f"   Uncertainty: ±{result.uncertainty} Pa")
        print(f"   Sample Height: {result.sample_height} mm")
        print(f"   Fit Quality (R²): {result.fit_quality}")
        print(f"   Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm")
        print(f"   Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}")
        
        # Plot results
        plot_choice = input("\nPlot results? (y/n): ").strip().lower()
        if plot_choice == 'y':
            analyzer.plot_results(result)
    else:
        print("❌ Analysis failed")

if __name__ == "__main__":
    main()
