#!/usr/bin/env python3
"""
Create a summary CSV from ASMI analysis results for heatmap plotting.
This script reads analysis summary files and creates a CSV compatible with plot_well_heatmap.py
"""

import os
import pandas as pd
import glob
from datetime import datetime

def extract_analysis_results(run_folder=None):
    """
    Extract analysis results from summary files and create a CSV.
    
    Args:
        run_folder: Specific run folder to analyze (e.g., 'run_001_20250717_021349')
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
    
    # Find all analysis summary files
    summary_files = glob.glob(os.path.join(run_path, "*_analysis_summary.txt"))
    if not summary_files:
        print(f"❌ No analysis summary files found in {run_path}")
        return None
    
    results = []
    
    for summary_file in summary_files:
        try:
            # Extract well name from filename
            filename = os.path.basename(summary_file)
            well_name = filename.split('_')[0]  # e.g., "A6_analysis_summary.txt" -> "A6"
            
            # Read the summary file
            with open(summary_file, 'r') as f:
                lines = f.readlines()
            
            # Parse the results
            result: dict = {'Well': well_name}
            for line in lines:
                line = line.strip()
                if 'Elastic Modulus:' in line:
                    result['ElasticModulus'] = float(line.split(':')[1].strip().split()[0])
                elif 'Uncertainty:' in line:
                    result['Std'] = float(line.split('±')[1].strip().split()[0])
                elif 'Fit Quality (R²):' in line:
                    result['R2'] = float(line.split(':')[1].strip())
                elif 'Poisson\'s Ratio:' in line:
                    result['PoissonRatio'] = float(line.split(':')[1].strip())
                elif 'Sample Height:' in line:
                    result['SampleHeight'] = float(line.split(':')[1].strip().split()[0])
                elif 'Material Type:' in line:
                    result['MaterialType'] = line.split(':')[1].strip()
            
            results.append(result)
            print(f"✅ Processed {well_name}")
            
        except Exception as e:
            print(f"⚠️ Error processing {summary_file}: {e}")
            continue
    
    if not results:
        print("❌ No valid results found")
        return None
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by well name (A1, A2, ..., H12)
    df['Row'] = df['Well'].str[0]
    df['Col'] = df['Well'].str[1:].astype(int)
    df = df.sort_values(['Row', 'Col'])
    df = df.drop(['Row', 'Col'], axis=1)
    
    # Save CSV
    output_csv = os.path.join(run_path, f"summary_{run_folder}.csv")
    df.to_csv(output_csv, index=False)
    print(f"💾 Summary CSV saved to: {output_csv}")
    print(f"📊 Processed {len(results)} wells")
    
    # Display summary statistics
    if 'ElasticModulus' in df.columns:
        print(f"\n📈 Summary Statistics:")
        print(f"   Mean Elastic Modulus: {df['ElasticModulus'].mean():.0f} Pa")
        print(f"   Std Elastic Modulus: {df['ElasticModulus'].std():.0f} Pa")
        print(f"   Min Elastic Modulus: {df['ElasticModulus'].min():.0f} Pa")
        print(f"   Max Elastic Modulus: {df['ElasticModulus'].max():.0f} Pa")
        print(f"   Mean R²: {df['R2'].mean():.3f}")
    
    return output_csv

def main():
    """Main function to create summary CSV"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Create summary CSV from ASMI analysis results")
    parser.add_argument('--run-folder', default=None, help='Specific run folder to process (e.g., run_001_20250717_021349)')
    parser.add_argument('--create-heatmap', action='store_true', help='Automatically create heatmap after CSV generation')
    args = parser.parse_args()
    
    # Create summary CSV
    csv_path = extract_analysis_results(args.run_folder)
    
    if csv_path and args.create_heatmap:
        print(f"\n🎨 Creating heatmap...")
        try:
            # Import and run the heatmap script
            from plot_well_heatmap import plot_well_heatmap
            
            # Create plot folder path
            plots_dir = "results/plots"
            if not os.path.exists(plots_dir):
                os.makedirs(plots_dir)
            
            # Extract run folder name from CSV path
            run_folder_name = os.path.basename(os.path.dirname(csv_path))
            plot_folder_path = os.path.join(plots_dir, run_folder_name)
            
            # Create plot folder if it doesn't exist
            if not os.path.exists(plot_folder_path):
                os.makedirs(plot_folder_path)
            
            # Create heatmap filename in plot folder
            heatmap_filename = f"{run_folder_name}_heatmap.png"
            heatmap_path = os.path.join(plot_folder_path, heatmap_filename)
            
            plot_well_heatmap(csv_path, save_path=heatmap_path, convert_to_mpa=True)
            print(f"🎨 Heatmap saved to: {heatmap_path}")
            
        except ImportError:
            print("⚠️ Could not import plot_well_heatmap.py")
            print("💡 Run manually: python plot_well_heatmap.py <csv_file> --save <output.png>")
        except Exception as e:
            print(f"❌ Error creating heatmap: {e}")

if __name__ == "__main__":
    main() 