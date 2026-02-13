#!/usr/bin/env python3
"""
ASMI v2 batch analysis runner - uses analysis_batch_2.py (original KABlab batch script).

Analyzes existing measurement data in results/measurements/<run_folder>/ using:
  - convert_measurement_format: converts Z/Raw_Force/Corrected_Force → well/depth/force
  - analysis_batch_2: original Hertzian fitting with baseline_threshold contact detection

Use main() with parameters. Edit parameters below when running as script.
"""

import os
import sys

from src.analysis_batch_2 import run_analysis


def main(
    existing_run_folder: str,
    p_ratio: float = 0.5,
    show_plot: bool = False,
    save_plot: bool = True,
    baseline_points: int = 10,
    save_heatmap: bool = True,
):
    """
    Analyze existing run folder using analysis_batch_2 (original batch script pipeline).

    Parameters
    ----------
    existing_run_folder : str
        Run folder name (e.g., "run_774_20260206_133925").
        Data is expected in results/measurements/<existing_run_folder>/.
    p_ratio : float, default 0.5
        Poisson's ratio (0.3 to 0.5)
    show_plot : bool, default False
        Whether to show plots (interactive)
    save_plot : bool, default True
        Whether to save per-well plots to results/plots/<folder>/Well.png
    baseline_points : int, default 10
        Auto-compute baseline from first N points per well (for data without baseline row).
        Set to None if data already has baseline row.
    save_heatmap : bool, default True
        Whether to save 96-well E heatmap to results/plots/<folder>/heatmap.png
    """
    if p_ratio < 0.3 or p_ratio > 0.5:
        print("Error: Poisson's ratio must be between 0.3 and 0.5")
        sys.exit(1)

    folder_name = os.path.basename(existing_run_folder.strip(os.sep))
    run_path = os.path.join("results", "measurements", folder_name)

    if not os.path.isdir(run_path):
        print(f"❌ Run folder not found: {run_path}")
        print("   Expected: results/measurements/<run_folder>/ with well_*.csv files")
        sys.exit(1)

    run_analysis(
        folder_name=folder_name,
        p_ratio=p_ratio,
        show_plot=show_plot,
        save_plot=save_plot,
        baseline_points=baseline_points,
        save_heatmap=save_heatmap,
    )


if __name__ == "__main__":
    try:
        main(
            existing_run_folder="run_774_20260206_133925",
            p_ratio=0.5,
            show_plot=False,
            save_plot=True,
            baseline_points=10,
            save_heatmap=True,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
