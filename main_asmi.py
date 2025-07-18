#!/usr/bin/env python3
"""
ASMI - Automated Sample Measurement Interface
Main entry point for the ASMI system
"""

from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
from src.force_monitoring import test_step_force_measurement, dynamic_indentation_measurement
from src.analysis import IndentationAnalyzer
import time
import os
from datetime import datetime
from typing import Optional

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

    def run_experiment(self, well_name: str, step_size: float = 0.2, force_limit: float = 45.0, z_target: float = -18.0):
        col, row = well_name[0], well_name[1:]
        try:
            print(f"\n🧪 Starting experiment at well {well_name}")
            self.cnc.move_to_well(col, row)
            self.cnc.move_to_z(-9.5, feedrate=1000)
            ok = test_step_force_measurement(self.cnc, self.force_sensor, well=well_name, target_z=z_target, step_size=step_size, force_limit=force_limit, run_folder=self.current_run_folder)
            if not ok:
                print("⚠️ Measurement failed")
                return False
            print(f"✅ Experiment at well {well_name} completed successfully")
            return True
        except Exception as e:
            print(f"❌ Error during experiment at well {well_name}: {e}")
            return False

    def run_dynamic_experiment(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0):
        """Run experiment with dynamic indentation control"""
        col, row = well_name[0], well_name[1:]
        try:
            print(f"\n🧪 Starting dynamic experiment at well {well_name}")
            self.cnc.move_to_well(col, row)
            self.cnc.move_to_z(-9.5, feedrate=1000)
            
            ok = dynamic_indentation_measurement(
                self.cnc, 
                self.force_sensor, 
                well=well_name, 
                z_target=z_target, 
                step_size=step_size, 
                force_limit=force_limit,
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

    def run_experiment_with_analysis(self, well_name: str, step_size: float = 0.2, force_limit: float = 45.0, poisson_ratio: Optional[float] = None, z_target: float = -18.0):
        """Run experiment and immediately analyze the results"""
        # Run the experiment
        success = self.run_experiment(well_name, step_size, force_limit, z_target)
        if success:
            # Wait a moment for file system to update
            time.sleep(0.5)
            # Analyze the data
            result = self.analyze_experiment_data(well_name, poisson_ratio)
            return success, result
        else:
            return False, None

    def run_dynamic_experiment_with_analysis(self, well_name: str, z_target: float = -18.0, step_size: float = 0.1, force_limit: float = 45.0, poisson_ratio: Optional[float] = None):
        """Run dynamic experiment and immediately analyze the results"""
        # Run the dynamic experiment
        success = self.run_dynamic_experiment(well_name, z_target, step_size, force_limit)
        if success:
            # Wait a moment for file system to update
            time.sleep(0.5)
            # Analyze the data
            result = self.analyze_experiment_data(well_name, poisson_ratio)
            return success, result
        else:
            return False, None

    def cleanup(self):
        print("\n🧹 Cleaning up ASMI system...")
        try:
            self.cnc.close()
            self.force_sensor.cleanup()
            print("✅ Cleanup completed")
        except Exception as e:
            print(f"⚠️ Error during cleanup: {e}")

def main(wells_to_test, auto_analyze=True, poisson_ratio: Optional[float] = None, 
         step_size: float = 0.2, force_limit: float = 40.0, z_target: float = -18.0, 
         use_dynamic: bool = False):
    asmi = None
    results = []
    
    # Record start time
    import time
    start_time = time.time()
    print(f"🚀 Starting ASMI experiment run at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Total wells to test: {len(wells_to_test)}")
    print(f"⚙️  Parameters: step_size={step_size}mm, force_limit={force_limit}N, z_target={z_target}mm")
    print(f"🔧 Mode: {'Dynamic' if use_dynamic else 'Step'} indentation")
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
                if use_dynamic:
                    success, result = asmi.run_dynamic_experiment_with_analysis(
                        well, z_target, step_size, force_limit, poisson_ratio
                    )
                else:
                    success, result = asmi.run_experiment_with_analysis(well, step_size, force_limit, poisson_ratio, z_target)
                if result:
                    results.append(result)
            else:
                if use_dynamic:
                    success = asmi.run_dynamic_experiment(well, z_target, step_size, force_limit)
                else:
                    success = asmi.run_experiment(well, step_size, force_limit, z_target)
                if success:
                    # Wait a moment for file system to update
                    time.sleep(0.5)
                    result = asmi.analyze_experiment_data(well, poisson_ratio)
                    if result:
                        results.append(result)
            
            well_time = time.time() - well_start_time
            print(f"⏱️  Well {well} completed in {well_time:.1f} seconds")
            
            if not success:
                print(f"⚠️ Stopping due to failure at well {well}")
                break
        
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
                    f.write(f"  Step Size: {step_size} mm\n")
                    f.write(f"  Force Limit: {force_limit} N\n")
                    f.write(f"  Z Target: {z_target} mm\n")
                    f.write(f"  Contact Threshold: 2.0 N\n")
                    f.write(f"  Dynamic Mode: {use_dynamic}\n")
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
        
        # Print summary of all results
        if results:
            print(f"\n📋 Summary of All Results:")
            print("=" * 60)
            for result in results:
                print(f"Well {result.well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}")
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
    # Example usage - modify these parameters as needed
    wells_to_test = ['A6', 'B6', 'C6', 'C5', 'B5', 'A5', 'D5', 'D6', 'D7', 'E7', 'E6', 'E5', 'F5', 'F4', 'F3', 'G3', 'G4']
    
    # Run experiments with default parameters
    main(
        wells_to_test=wells_to_test,
        auto_analyze=True,
        poisson_ratio=None,  # Auto-detect based on force limit
        step_size=0.1,
        force_limit=25.0,
        z_target=-15.0,
        use_dynamic=True  # Set to True for dynamic indentation
    )