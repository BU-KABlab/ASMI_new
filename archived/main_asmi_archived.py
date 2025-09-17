#!/usr/bin/env python3
"""
ASMI - Automated Sample Measurement Interface
Main entry point for the ASMI system
"""

from operator import truediv
from tkinter import TRUE
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
from src.force_monitoring import test_step_force_measurement, dynamic_indentation_measurement, dynamic_indentation_with_retrospective_contact
from src.analysis import IndentationAnalyzer
from src.plot import plotter
import time
import os
from datetime import datetime
from typing import Optional, Tuple
import csv

class ASMI:
    def __init__(self):
        print("🔧 Initializing ASMI system...")
        self.cnc = CNCController()
        self.force_sensor = ForceSensor()
        if not self.force_sensor.is_connected():
            raise RuntimeError("Force sensor not connected")
        self.analyzer = IndentationAnalyzer()
        self.current_run_folder = None
        print("✅ ASMI system initialized successfully")

    def create_run_folder(self):
        """Create a new run folder for the current experiment sequence"""
        from src.force_monitoring import get_and_increment_run_count
        run_count = get_and_increment_run_count()
        run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
        os.makedirs(self.current_run_folder, exist_ok=True)
        print(f"📁 Created run folder: {self.current_run_folder}")
        return self.current_run_folder

    def run_dynamic_experiment(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0, contact_detection_depth: float = 1.0, use_sophisticated_detection: bool = False, stop_after_contact: bool = True, contact_force_threshold: float = 2.0):
        """Run experiment with dynamic indentation control"""
        col, row = well_name[0], well_name[1:]
        try:
            print(f"\n🧪 Starting dynamic experiment at well {well_name}")
            self.cnc.move_to_well(col, row)
            self.cnc.move_to_z(-8.0, feedrate=1000)
            self.cnc.wait_for_idle()  # Wait for movement to well top to complete
            
            ok = dynamic_indentation_measurement(
                self.cnc, 
                self.force_sensor, 
                well=well_name, 
                z_target=z_target, 
                step_size=step_size, 
                force_limit=force_limit,
                contact_detection_depth=contact_detection_depth,
                use_sophisticated_detection=use_sophisticated_detection,
                stop_after_contact=stop_after_contact,
                contact_force_threshold=contact_force_threshold,  # ← Add this
                run_folder=self.current_run_folder
            )
            
            if not ok:
                print("⚠️ Dynamic measurement failed")
                return False
            print(f"✅ Dynamic experiment at well {well_name} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error during dynamic experiment at well {well_name}: {e}")
            return False

    def analyze_experiment_data(self, well_name: str, poisson_ratio: Optional[float] = None):
        """Analyze the data from a completed experiment"""
        print(f"\n🔬 Analyzing data for well {well_name}...")
        
        # Find the most recent data file for this well
        data_dir = "results/measurements"
        if not os.path.exists(data_dir):
            print(f"❌ Data directory {data_dir} not found")
            return None
        
        # Look for the most recent run folder
        run_folders = [f for f in os.listdir(data_dir) if f.startswith("run_")]
        if not run_folders:
            print(f"❌ No run folders found in {data_dir}")
            return None
        
        # Get the most recent run folder
        latest_run = max(run_folders, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
        run_path = os.path.join(data_dir, latest_run)
        
        # Find the data file for this well
        well_files = [f for f in os.listdir(run_path) if f.startswith(f"well_{well_name}_") and f.endswith(".csv")]
        if not well_files:
            print(f"❌ No data file found for well {well_name} in {run_path}")
            return None
        
        # Get the most recent file for this well
        latest_file = max(well_files, key=lambda x: os.path.getctime(os.path.join(run_path, x)))
        filepath = os.path.join(run_path, latest_file)
        
        print(f"📁 Analyzing file: {filepath}")
        
        # Load and analyze the data
        if not self.analyzer.load_data(filepath):
            print(f"❌ Failed to load data from {filepath}")
            return None
        
        # Analyze the well
        result = self.analyzer.analyze_well(well_name, poisson_ratio, filepath)
        if result:
            print(f"\n📊 Analysis Results for Well {result.well}:")
            print(f"   Elastic Modulus: {result.elastic_modulus} Pa")
            print(f"   Uncertainty: ±{result.uncertainty} Pa")
            print(f"   Sample Height: {result.sample_height} mm")
            print(f"   Fit Quality (R²): {result.fit_quality}")
            print(f"   Depth Range: {result.depth_range[0]:.2f}-{result.depth_range[1]:.2f} mm")
            print(f"   Fit Parameters: A={result.fit_A:.3f}, d0={result.fit_d0:.3f}")
            print(f"   Contact Point: Z={result.contact_z:.3f} mm, Force={result.contact_force:.3f} N")
            print(f"   Material Type: {result.material_type}")
            
            # Save plot and summary
            run_folder_name = os.path.basename(run_path) if run_path else None
            self.analyzer.plot_results(result, save_plot=True, run_folder=run_folder_name)
            
            # Save summary in the run folder
            summary_filename = os.path.join(run_path, f"{well_name}_analysis_summary.txt")
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
            
            return result
        else:
            print(f"❌ Analysis failed for well {well_name}")
            return None

    def run_dynamic_experiment_with_analysis(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0, poisson_ratio: Optional[float] = None, contact_detection_depth: float = 1.0, use_sophisticated_detection: bool = False, stop_after_contact: bool = True, contact_force_threshold: float = 2.0):
        """Run dynamic experiment and immediately analyze the results"""
        # Run the dynamic experiment
        success = self.run_dynamic_experiment(well_name, z_target, step_size, force_limit, contact_detection_depth, use_sophisticated_detection, stop_after_contact, contact_force_threshold)  # ← Add this
        if success:
            # Wait a moment for file system to update
            time.sleep(0.5)
            result = self.analyze_experiment_data(well_name, poisson_ratio)
            return success, result
        return False, None


    def run_retrospective_contact_experiment(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0, contact_threshold: float = 0.05):
        """Run experiment with retrospective contact detection"""
        col, row = well_name[0], well_name[1:]
        try:
            print(f"\n🧪 Starting retrospective contact experiment at well {well_name}")
            self.cnc.move_to_well(col, row)
            self.cnc.move_to_z(-8.0, feedrate=1000)
            self.cnc.wait_for_idle()
            
            ok = dynamic_indentation_with_retrospective_contact(
                self.cnc, 
                self.force_sensor, 
                well=well_name, 
                z_target=z_target, 
                step_size=step_size, 
                force_limit=force_limit,
                contact_threshold=contact_threshold,
                run_folder=self.current_run_folder
            )
            
            if not ok:
                print("⚠️ Retrospective contact measurement failed")
                return False
            print(f"✅ Retrospective contact experiment at well {well_name} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error during retrospective contact experiment at well {well_name}: {e}")
            return False

    def run_retrospective_contact_experiment_with_analysis(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0, contact_threshold: float = 0.05, poisson_ratio: Optional[float] = None):
        """Run retrospective contact experiment and immediately analyze the results"""
        # Run the retrospective contact measurement
        success = self.run_retrospective_contact_experiment(well_name, z_target, step_size, force_limit, contact_threshold)
        if success:
            # Wait a moment for file system to update
            time.sleep(0.5)
            # Analyze the data
            result = self.analyze_experiment_data(well_name, poisson_ratio)
            return success, result
        else:
            return False, None

    def recover_cnc(self):
        """Recover CNC from alarm state"""
        print("🔄 Attempting to recover CNC from alarm state...")
        
        try:
            # Try to unlock first
            if self.cnc.unlock():
                print("✅ CNC unlocked successfully")
            else:
                print("⚠️ Unlock failed, trying full reset...")
                self.cnc.reset_grbl()
            
            # Wait a moment for reset to complete
            time.sleep(2)
            
            # Check if CNC is responsive
            current_pos = self.cnc.get_current_position()
            if current_pos:
                print(f"✅ CNC recovered! Current position: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
                return True
            else:
                print("❌ CNC still not responsive after recovery attempt")
                return False
                
        except Exception as e:
            print(f"❌ Error during CNC recovery: {e}")
            return False

    def cleanup(self):
        print("\n🧹 Cleaning up ASMI system...")
        try:
            self.cnc.close()
            self.force_sensor.cleanup()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")
            
            
            
            

def main(wells_to_test, auto_analyze=True, poisson_ratio: Optional[float] = None, 
         coarse_step_size: float = 0.5, fine_step_size: float = 0.02, fine_steps: int = 50,
         force_limit: float = 40.0, z_target: float = -18.0, 
         fine_sleep_time: float = 0.01, contact_detection_depth: float = 1.0, 
         use_sophisticated_detection: bool = False, stop_after_contact: bool = True, 
         contact_force_threshold: float = 0.5, find_contact_retrospectively: bool = False):
    asmi = None
    results = []
    
    # Record start time
    import time
    start_time = time.time()
    print(f"🚀 Starting ASMI experiment run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Total wells to test: {len(wells_to_test)}")
    print(f"⚙️  Parameters: step_size={fine_step_size}mm, force_limit={force_limit}N, z_target={z_target}mm")
    print("=" * 60)
    
    try:
        asmi = ASMI()
        # Home once at the beginning of all experiments
        asmi.cnc.home()
        # Create a run folder for the entire experiment sequence
        asmi.create_run_folder()

        for i, well in enumerate(wells_to_test):
            well_start_time = time.time()
            print(f"\n🔬 Processing well {well} ({i+1}/{len(wells_to_test)})...")
            
            if auto_analyze:
                if find_contact_retrospectively:
                    success, result = asmi.run_retrospective_contact_experiment_with_analysis(
                        well_name=well,
                        z_target=z_target,
                        step_size=fine_step_size,
                        force_limit=force_limit,
                        contact_threshold=0.05,  # Threshold for retrospective contact detection
                        poisson_ratio=poisson_ratio
                    )
                if result:
                    results.append(result)
                    print(f"✅ Analysis completed successfully for well {well}")
                    print(f"🔍 Debug: Added result type = {type(result)} to results list")
                elif success:
                    print(f"✅ Measurement completed for well {well} (analysis failed)")
                    print(f"🔍 Debug: result type = {type(result)}, result = {result}")
                else:
                    print(f"❌ Measurement failed for well {well}")
            else:
                # Only run measurement without analysis
                success = asmi.run_dynamic_experiment(
                    well, z_target, fine_step_size, force_limit, 
                    contact_detection_depth, use_sophisticated_detection, stop_after_contact,
                    contact_force_threshold
                )
                
                # Only analyze if measurement succeeded AND we want analysis
                if success and auto_analyze:
                    time.sleep(0.5)
                    result = asmi.analyze_experiment_data(well, poisson_ratio)
                    if result:
                        results.append(result)
            
            well_time = time.time() - well_start_time
            print(f"⏱️  Well {well} completed in {well_time:.1f} seconds")
            
            if not success:
                print(f"⚠️ Stopping due to failure at well {well}")
                break
        
        # NOTE: Retrospective contact detection is already handled in the main experiment loop above
        # The find_contact_retrospectively parameter enables it in each individual experiment
        
        # Generate raw data plot for all wells
        if results and asmi.current_run_folder:
            print(f"\n📊 Generating raw data plot for all wells...")
            try:
                # Use the current run folder
                run_folder_name = os.path.basename(asmi.current_run_folder)
                asmi.analyzer.plot_raw_data_all_wells(run_folder_name, save_plot=True)
            except Exception as e:
                print(f"⚠️ Error generating raw data plot: {e}")
        
        # Generate individual raw force plots for each well
        if results and asmi.current_run_folder:
            print(f"\n📊 Generating individual raw force plots for each well...")
            try:
                # Use the current run folder
                run_folder_name = os.path.basename(asmi.current_run_folder)
                asmi.analyzer.plot_raw_force_individual_wells(run_folder_name, save_plot=True)
            except Exception as e:
                print(f"⚠️ Error generating individual raw force plots: {e}")
        
        # Calculate total time
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        
        # Create summary CSV and heatmap
        if results and asmi.current_run_folder:
            print(f"\n📈 Creating summary CSV and heatmap...")
            try:
                # Import the summary creation function
                from create_summary_csv import extract_analysis_results
                from plot_well_heatmap import plot_well_heatmap
                
                # Add timing information to the run folder
                timing_info = {
                    'total_time_seconds': total_time,
                    'total_time_formatted': f"{hours:02d}:{minutes:02d}:{seconds:05.2f}",
                    'wells_completed': len(results),
                    'wells_total': len(wells_to_test),
                    'avg_time_per_well': total_time / len(results) if len(results) > 0 else 0,
                    'start_time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time)),
                    'end_time': time.strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Save timing information to a separate file
                timing_filename = os.path.join(asmi.current_run_folder, "run_timing.txt")
                with open(timing_filename, 'w') as f:
                    f.write(f"ASMI Run Timing Information\n")
                    f.write("=" * 40 + "\n")
                    f.write(f"Total Run Time: {timing_info['total_time_formatted']}\n")
                    f.write(f"Wells Completed: {timing_info['wells_completed']}/{timing_info['wells_total']}\n")
                    f.write(f"Average Time per Well: {timing_info['avg_time_per_well']:.1f} seconds\n")
                    f.write(f"Start Time: {timing_info['start_time']}\n")
                    f.write(f"End Time: {timing_info['end_time']}\n")
                    f.write(f"Parameters:\n")
                    f.write(f"  Step Size: {fine_step_size} mm\n")
                    f.write(f"  Force Limit: {force_limit} N\n")
                    f.write(f"  Z Target: {z_target} mm\n")
                    f.write(f"  Contact Threshold: {contact_force_threshold} N\n")
                print(f"💾 Timing information saved to: {timing_filename}")
                
                # Get the run folder name
                run_folder_name = os.path.basename(asmi.current_run_folder)
                
                # Create summary CSV
                csv_path = extract_analysis_results(run_folder_name)
                
                if csv_path:
                    # Create plot folder path
                    plots_dir = "results/plots"
                    if not os.path.exists(plots_dir):
                        os.makedirs(plots_dir)
                    
                    plot_folder_path = os.path.join(plots_dir, run_folder_name)
                    
                    # Create plot folder if it doesn't exist
                    if not os.path.exists(plot_folder_path):
                        os.makedirs(plot_folder_path)
                    
                    # Create heatmap filename in plot folder
                    heatmap_filename = f"{run_folder_name}_heatmap.png"
                    heatmap_path = os.path.join(plot_folder_path, heatmap_filename)
                    
                    # Generate heatmap
                    plot_well_heatmap(csv_path, save_path=heatmap_path, convert_to_mpa=True)
                    print(f"🎨 Heatmap saved to: {heatmap_path}")
                else:
                    print("⚠️ Could not create summary CSV")
                    
            except ImportError as e:
                print(f"⚠️ Could not import required modules for heatmap creation: {e}")
            except Exception as e:
                print(f"⚠️ Error creating heatmap: {e}")
        else:
            print(f"\n📈 Skipping summary CSV and heatmap creation (no results)")
            if not results:
                print(f"   Reason: No analysis results available")
            if not asmi.current_run_folder:
                print(f"   Reason: No current run folder")
        
        # Generate UV exposure analysis plot
        if results and asmi.current_run_folder:
            print(f"\n🔬 Generating UV exposure analysis plot...")
            try:
                # Import UV exposure analysis functions
                from plot_uv_exposure_analysis import analyze_uv_exposure_data
                
                # Get the run folder name
                run_folder_name = os.path.basename(asmi.current_run_folder)
                
                # Generate UV exposure plot
                uv_success = analyze_uv_exposure_data(run_folder_name)
                
                if uv_success:
                    print(f"✅ UV exposure analysis completed successfully!")
                else:
                    print(f"⚠️ UV exposure analysis failed or no data available")
                    
            except ImportError as e:
                print(f"⚠️ Could not import UV exposure analysis modules: {e}")
            except Exception as e:
                print(f"⚠️ Error creating UV exposure plot: {e}")
        else:
            print(f"\n🔬 Skipping UV exposure analysis (no results)")
            if not results:
                print(f"   Reason: No analysis results available")
            if not asmi.current_run_folder:
                print(f"   Reason: No current run folder")
        
        # Print summary of all results
        if results:
            print(f"\n📋 Summary of All Results:")
            print("=" * 60)
            for result in results:
                if hasattr(result, 'well'):
                    # AnalysisResult object
                    print(f"Well {result.well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}")
                elif isinstance(result, dict):
                    # Dictionary from original analysis method
                    well_name = result.get('well', 'Unknown')
                    elastic_modulus = result.get('elastic_modulus', 'N/A')
                    print(f"Well {well_name}: E = {elastic_modulus} Pa (Original Method)")
                else:
                    print(f"Unknown result type: {type(result)}")
            print("=" * 60)
        else:
            print(f"\n📋 Summary of All Results:")
            print("=" * 60)
            print("No successful analysis results to display")
            print("=" * 60)
        
        print(f"\n🎉 All experiments completed!")
        print(f"⏱️  Total run time: {hours:02d}:{minutes:02d}:{seconds:05.2f} (HH:MM:SS.ss)")
        print(f"📊 Wells completed: {len(results)}/{len(wells_to_test)}")
        if len(results) > 0:
            avg_time_per_well = total_time / len(results)
            print(f"📈 Average time per well: {avg_time_per_well:.1f} seconds")
        print(f"🏁 Finished at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
    except KeyboardInterrupt:
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"\n🛑 Experiments interrupted by user after {hours:02d}:{minutes:02d}:{seconds:05.2f}")
    except Exception as e:
        total_time = time.time() - start_time
        hours = int(total_time // 3600)
        minutes = int((total_time % 3600) // 60)
        seconds = total_time % 60
        print(f"\n❌ Fatal error after {hours:02d}:{minutes:02d}:{seconds:05.2f}: {e}")
        print("💡 Check if:")
        print("   - CNC machine is connected via USB")
        print("   - No other software is using the CNC port")
        print("   - CNC machine is powered on")
        print("   - Force sensor is connected")
    finally:
        if asmi:
            asmi.cleanup()

if __name__ == "__main__":
    # Analyze existing data from run_440_20250905_103414
    print("🔬 Analyzing existing data from run_440_20250905_103414")
    print("=" * 60)
    
    # Wells available in run_440_20250905_103414
    wells_to_analyze = ['F10', 'F11', 'F12', 'G10', 'G11', 'G12', 'H10']
    
    # Use analyzer directly without hardware initialization
    from src.analysis import IndentationAnalyzer
    analyzer = IndentationAnalyzer()
    
    results = []
    
    try:
        for well in wells_to_analyze:
            print(f"\n🔬 Analyzing well {well}...")
            
            # Find the data file for this well
            data_dir = "results/measurements"
            run_path = os.path.join(data_dir, "run_440_20250905_103414")
            
            well_files = [f for f in os.listdir(run_path) if f.startswith(f"well_{well}_") and f.endswith(".csv")]
            if not well_files:
                print(f"❌ No data file found for well {well}")
                continue
            
            # Get the most recent file for this well
            latest_file = max(well_files, key=lambda x: os.path.getctime(os.path.join(run_path, x)))
            filepath = os.path.join(run_path, latest_file)
            
            print(f"📁 Analyzing file: {filepath}")
            
            # Load and analyze the data
            if not analyzer.load_data(filepath):
                print(f"❌ Failed to load data from {filepath}")
                continue
            
            # Analyze the well using the standard method
            result = analyzer.analyze_well(well, poisson_ratio=0.33, filename=filepath)
            if result:
                results.append(result)
                print(f"✅ Well {well} analyzed successfully")
                print(f"   Elastic Modulus: {result.elastic_modulus} Pa")
                print(f"   Fit Quality (R²): {result.fit_quality:.3f}")
                print(f"   Contact Point: Z={result.contact_z:.3f} mm")
            else:
                print(f"❌ Well {well} analysis failed")
        
        # Print summary
        if results:
            print(f"\n📋 Analysis Summary:")
            print("=" * 60)
            for result in results:
                print(f"Well {result.well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality:.3f}, Contact Z = {result.contact_z:.3f} mm")
        else:
            print("❌ No wells were successfully analyzed")
            
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()