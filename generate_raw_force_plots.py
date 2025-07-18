#!/usr/bin/env python3
"""
Generate individual raw force plots for each well in a measurement run
This script creates separate plots showing raw and corrected force data for each well
"""

import os
import sys
from src.analysis import IndentationAnalyzer

def generate_raw_force_plots(run_folder=None):
    """
    Generate individual raw force plots for each well in a run
    
    Args:
        run_folder: Specific run folder to process (e.g., 'run_001_20250717_021349')
                   If None, uses the most recent run folder
    """
    
    # Find the run folder
    data_dir = "results/measurements"
    if not os.path.exists(data_dir):
        print(f"❌ Data directory {data_dir} not found")
        return None
    
    if run_folder is None:
        # Get the most recent run folder
        run_folders = [f for f in os.listdir(data_dir) if f.startswith("run_")]
        if not run_folders:
            print(f"❌ No run folders found in {data_dir}")
            return None
        run_folder = max(run_folders, key=lambda x: os.path.getctime(os.path.join(data_dir, x)))
    
    run_path = os.path.join(data_dir, run_folder)
    if not os.path.exists(run_path):
        print(f"❌ Run folder {run_path} not found")
        return None
    
    print(f"📁 Processing run folder: {run_folder}")
    
    # Initialize analyzer and generate plots
    analyzer = IndentationAnalyzer()
    analyzer.plot_raw_force_individual_wells(run_folder, save_plot=True)
    
    print(f"✅ Raw force plots generation completed for {run_folder}")

def main():
    """Main function to generate raw force plots"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate individual raw force plots for each well")
    parser.add_argument('--run-folder', default=None, help='Specific run folder to process (e.g., run_001_20250717_021349)')
    args = parser.parse_args()
    
    # Generate raw force plots
    generate_raw_force_plots(args.run_folder)

if __name__ == "__main__":
    main() 