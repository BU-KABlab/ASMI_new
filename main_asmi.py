#!/usr/bin/env python3
"""
ASMI v2 runner (parameter-based, no CLI args) - secondary entrypoint

Supports two workflows:
  1) Measure -> Analyze -> Plot (default measurement: simple_indentation_measurement)
  2) Analyze existing data folder -> Plot

Also supports splitting direction-tagged measurements into _down/_up CSVs and per-direction analysis/plots.

Uses PANDA_CORE for gantry control (Gantry) and force sensing (ASMI).

Author: Hongrui Zhang
Date: 02/2026
License: MIT
"""

import os
import sys
import csv
import time
from datetime import datetime
from typing import Optional

import yaml

# ── Load experiment config and set up PANDA_CORE path ─────────────────────
_CONFIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config')

with open(os.path.join(_CONFIG_DIR, 'experiment.yaml')) as _f:
    _cfg = yaml.safe_load(_f)

_PANDA_CORE_SRC = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    _cfg.get('paths', {}).get('panda_core_src', '../PANDA_CORE/src'),
)
if os.path.isdir(_PANDA_CORE_SRC) and _PANDA_CORE_SRC not in sys.path:
    sys.path.insert(0, _PANDA_CORE_SRC)

_RESULTS_BASE = _cfg.get('paths', {}).get('results_base', 'results/measurements')
_PLOTS_BASE = _cfg.get('paths', {}).get('plots_base', 'results/plots')
_RUN_COUNT_FILE = _cfg.get('paths', {}).get('run_count_file', 'src/run_count.txt')
_SAFE_Z = _cfg.get('gantry', {}).get('safe_z', -50.0)
_K_SYSTEM = _cfg.get('system', {}).get('k_system', 64.27)

from gantry.gantry import Gantry
from instruments.asmi.driver import ASMI
from deck import load_deck_from_yaml

from src.ForceMonitoring import (
    simple_indentation_measurement,
    simple_indentation_with_return_measurement,
    get_and_increment_run_count,
)
from src.analysis import IndentationAnalyzer
from src.plot import plotter
from src.version import get_full_version

# ── Load deck ─────────────────────────────────────────────────────────────
_deck_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    _cfg.get('paths', {}).get('deck_yaml', 'config/deck.yaml'),
)
_deck = load_deck_from_yaml(_deck_path)
_plate = _deck["plate"]


def _resolve_well_xy(well_id: str) -> tuple[float, float]:
    """Look up well XY from the deck's well plate."""
    coord = _plate.get_well_center(well_id)
    return (coord.x, coord.y)


def ensure_run_folder(base: str = None) -> str:
    """Create and return a new run folder path under base."""
    if base is None:
        base = _RESULTS_BASE
    run_count = get_and_increment_run_count(_RUN_COUNT_FILE)
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base, f"run_{run_count:03d}_{run_date}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def plot_results_via_plotter(result, run_folder: str | None, method: str | None = None, direction_label: str | None = None):
    """Thin wrapper to call plotter.plot_results with common defaults."""
    try:
        plotter.plot_results(result, save_plot=True, run_folder=run_folder, method=method, direction_label=direction_label)
    except TypeError:
        plotter.plot_results(result, save_plot=True, run_folder=run_folder)


def split_up_down_csv(orig_csv_path: str) -> tuple[str | None, str | None]:
    """Split a measurement CSV with Direction column into two files: _down and _up.

    - Copies metadata rows unchanged and adds a 'Direction_File' marker row.
    - Preserves original headers; for missing header, writes a default header.
    - Sorts the 'up' subset by increasing absolute Z to align return trajectory.
    """
    import csv as _csv
    try:
        with open(orig_csv_path, 'r') as f:
            reader = _csv.reader(f)
            rows = [r for r in reader if r]
    except Exception as e:
        print(f"Failed to read for splitting: {orig_csv_path}: {e}")
        return None, None

    metadata_rows: list[list[str]] = []
    data_rows: list[list[str]] = []
    header = None
    for r in rows:
        if len(r) >= 4 and r[0].replace('.', '', 1).replace('-', '', 1).isdigit():
            data_rows.append(r)
        elif r and r[0] == 'Timestamp(s)':
            header = r
        else:
            metadata_rows.append(r)

    if not data_rows:
        print("No data rows to split.")
        return None, None

    down_rows: list[list[str]] = []
    up_rows: list[list[str]] = []
    for r in data_rows:
        direction = r[4] if len(r) >= 5 else 'down'
        if direction == 'up':
            up_rows.append(r)
        else:
            down_rows.append(r)

    try:
        up_rows.sort(key=lambda r: abs(float(r[1])))
    except Exception:
        pass

    root, ext = os.path.splitext(orig_csv_path)
    down_path = f"{root}_down{ext}" if down_rows else None
    up_path = f"{root}_up{ext}" if up_rows else None

    def _write_subset(path: str, subset_rows: list[list[str]], label: str):
        with open(path, 'w', newline='') as f:
            w = _csv.writer(f)
            for m in metadata_rows:
                w.writerow(m)
            w.writerow(['Direction_File', label])
            w.writerow([])
            if header:
                w.writerow(header)
            else:
                w.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)', 'Direction'])
            for r in subset_rows:
                w.writerow(r)

    try:
        if down_path:
            _write_subset(down_path, down_rows, 'down')
        if up_path:
            _write_subset(up_path, up_rows, 'up')
    except Exception as e:
        print(f"Failed to write split files: {e}")

    return down_path, up_path


def analyze_file(datafile: str, well: str, contact_method: str = "retrospective", fit_method: str = "hertzian", apply_system_correction: bool = True, retrospective_threshold: float | None = None, max_depth: float = 0.5, min_depth: float = 0.25, apply_force_correction: bool = False, iterative_d0_refinement: bool = False, well_bottom_z: float = -85.0, poisson_ratio: float | None = None, use_legacy_height: bool = False, legacy_height_step_mm: float = 0.02):
    """Analyze a single CSV file and emit plots. Compatible with current src.Analysis."""
    data_dir, filename = os.path.split(datafile)
    analyzer = IndentationAnalyzer(data_dir or ".")
    if not analyzer.load_data(filename):
            return None

    method_key = {
        "extrapolation": "true_contact",
        "retrospective": "retrospective",
        "simple_threshold": "simple_threshold",
        "baseline_threshold": "baseline_threshold",
    }.get(contact_method, "true_contact")

    try:
        result = analyzer.analyze_well(
            well=well,
            poisson_ratio=poisson_ratio,
            filename=datafile,
            contact_method=method_key,
            fit_method=fit_method,
            apply_system_correction=apply_system_correction,
            retrospective_threshold=retrospective_threshold,
            max_depth=max_depth,
            min_depth=min_depth,
            apply_force_correction=apply_force_correction,
            iterative_d0_refinement=iterative_d0_refinement,
            well_bottom_z=well_bottom_z,
            use_legacy_height=use_legacy_height,
            legacy_height_step_mm=legacy_height_step_mm,
        )
    except TypeError:
        result = analyzer.analyze_well(
            well=well,
            poisson_ratio=poisson_ratio,
            filename=datafile,
            fit_method=fit_method,
            apply_system_correction=apply_system_correction,
            retrospective_threshold=retrospective_threshold,
            max_depth=max_depth,
            min_depth=min_depth,
            apply_force_correction=apply_force_correction,
            iterative_d0_refinement=iterative_d0_refinement,
            well_bottom_z=well_bottom_z,
            use_legacy_height=use_legacy_height,
            legacy_height_step_mm=legacy_height_step_mm,
        )

    if not result:
        print("Analysis failed")
        return None

    run_folder = None
    for part in data_dir.split(os.sep):
        if part.startswith("run_"):
            run_folder = part
            break

    dir_label = None
    if well.lower().endswith("_down"):
        dir_label = "down"
    elif well.lower().endswith("_up"):
        dir_label = "up"

    try:
        method_for_plot = {
            "extrapolation": "extrapolation",
            "retrospective": "retrospective",
            "simple_threshold": "simple_threshold",
            "baseline_threshold": "baseline_threshold",
        }.get(contact_method, "extrapolation")
        plot_results_via_plotter(result, run_folder, method=method_for_plot, direction_label=dir_label)
    except Exception:
        plot_results_via_plotter(result, run_folder)
    return result


def run_measure_analyze_plot(
    gantry,
    asmi,
    well: str | None,
    contact_method: str,
    measure_with_return: bool = False,
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float | None = -9.0,
    run_folder: str | None = None,
    fit_method: str = "hertzian",
    apply_system_correction: bool = True,
    retrospective_threshold: float | None = None,
    lock_xy_single_spot: bool = False,
    lock_xy_position: tuple[float, float] | None = None,
    max_depth: float | None = None,
    min_depth: float = 0.25,
    apply_force_correction: bool = False,
    iterative_d0_refinement: bool = False,
    well_bottom_z: float = -85.0,
    poisson_ratio: float | None = None,
    use_legacy_height: bool = False,
    legacy_height_step_mm: float = 0.02,
):
    """Measure a single well or current position, then analyze and plot."""
    run_folder = run_folder or ensure_run_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if well is not None:
        datafile = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")
    else:
        datafile = os.path.join(run_folder, f"indentation_{timestamp}.csv")
        print(f"Measuring at current position (no well specified)")

    # Handle well_top_z=None by using current Z position
    if well_top_z is None:
        try:
            current_pos = gantry.get_coordinates()
            well_top_z = float(current_pos["z"])
            print(f"Using current Z position as well_top_z: {well_top_z:.1f}mm")
        except Exception:
            print("Could not get current position, using default well_top_z=-9.0mm")
            well_top_z = -9.0

    try:
        t0 = time.time()
        well_xy = _resolve_well_xy(well) if well is not None else None
        common = dict(
            gantry=gantry, asmi=asmi,
            well=well, well_xy=well_xy, safe_z=_SAFE_Z,
            filename=datafile, run_folder=run_folder,
            results_base=_RESULTS_BASE, run_count_file=_RUN_COUNT_FILE,
            z_target=z_target, step_size=step_size,
            force_limit=force_limit, well_top_z=well_top_z,
            locked_xy=(lock_xy_position if lock_xy_single_spot else None),
        )
        if measure_with_return:
            ok = simple_indentation_with_return_measurement(**common)
        else:
            ok = simple_indentation_measurement(**common)
        if not ok:
            print("Measurement failed")
            return None, None

        duration_s = time.time() - t0
        print(f"Measurement saved to: {datafile}")
        print(f"Total measurement time: {duration_s:.2f} s")
        try:
            with open(datafile, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Total_Measurement_Time(s)', f"{duration_s:.3f}"])
        except Exception as e:
            print(f"Could not append total time to CSV: {e}")

        per_well_results = []

        if measure_with_return:
            down_csv, up_csv = split_up_down_csv(datafile)
            if well is not None:
                well_down = f"{well}_down"
                well_up = f"{well}_up"
            else:
                well_down = "indentation_down"
                well_up = "indentation_up"
            if down_csv:
                r_down = analyze_file(datafile=down_csv, well=well_down, contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
                if r_down:
                    per_well_results.append(r_down)
            if up_csv:
                r_up = analyze_file(datafile=up_csv, well=well_up, contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
                if r_up:
                    per_well_results.append(r_up)
        else:
            plain_well = well.upper() if well is not None else "indentation"
            r_single = analyze_file(datafile=datafile, well=plain_well, contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
            if r_single:
                per_well_results.append(r_single)

        return per_well_results, os.path.basename(run_folder)
    except KeyboardInterrupt:
        print("Keyboard interrupt received.")
        raise


def write_summary_csv(run_folder_name: str, results: list):
    """Write summary.csv for heatmap plotting under results/plots/<run_folder_name>/."""
    plots_root = _PLOTS_BASE
    out_dir = os.path.join(plots_root, run_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary.csv")

    has_linear = any(getattr(r, 'spring_constant', None) is not None for r in results if r)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        if has_linear:
            w.writerow(["Well", "SpringConstant_k", "Intercept_b", "R2"])
            for r in results:
                if r:
                    name_lower = r.well.lower() if getattr(r, 'well', None) else ""
                    if name_lower.endswith("_down"):
                        well_core = r.well[: -len("_down")]
                    elif name_lower.endswith("_up"):
                        well_core = r.well[: -len("_up")]
                    else:
                        well_core = r.well
                    k_val = getattr(r, 'spring_constant', 0)
                    b_val = getattr(r, 'linear_intercept', 0)
                    r2_val = getattr(r, 'linear_fit_quality', getattr(r, 'fit_quality', 0))
                    w.writerow([well_core.upper(), k_val, b_val, r2_val])
        else:
            has_system_correction = any(getattr(r, 'original_elastic_modulus', None) is not None for r in results if r)
            if has_system_correction:
                w.writerow(["Well", "ElasticModulus", "ElasticModulus_Original", "Std", "R2", "R2_Original"])
                for r in results:
                    if r:
                        name_lower = r.well.lower() if getattr(r, 'well', None) else ""
                        if name_lower.endswith("_down"):
                            well_core = r.well[: -len("_down")]
                        elif name_lower.endswith("_up"):
                            well_core = r.well[: -len("_up")]
                        else:
                            well_core = r.well
                        orig_E = getattr(r, 'original_elastic_modulus', r.elastic_modulus)
                        orig_r2 = getattr(r, 'original_fit_quality', r.fit_quality)
                        w.writerow([well_core.upper(), r.elastic_modulus, orig_E, r.uncertainty, r.fit_quality, orig_r2])
            else:
                w.writerow(["Well", "ElasticModulus", "Std", "R2"])
                for r in results:
                    if r:
                        name_lower = r.well.lower() if getattr(r, 'well', None) else ""
                        if name_lower.endswith("_down"):
                            well_core = r.well[: -len("_down")]
                        elif name_lower.endswith("_up"):
                            well_core = r.well[: -len("_up")]
                        else:
                            well_core = r.well
                        w.writerow([well_core.upper(), r.elastic_modulus, r.uncertainty, r.fit_quality])
    print(f"Summary CSV written: {out_csv}")
    return out_csv


def correct_spring_constant_csv(csv_path: str, k_system: float = None, output_path: Optional[str] = None):
    if k_system is None:
        k_system = _K_SYSTEM
    """Read spring constant CSV and apply system compliance correction."""
    import pandas as pd

    if not os.path.exists(csv_path):
        print(f"CSV file not found: {csv_path}")
        return None

    df = pd.read_csv(csv_path)

    if 'SpringConstant_k' not in df.columns:
        print(f"Column 'SpringConstant_k' not found. Available: {df.columns.tolist()}")
        return None

    df_corrected = df.copy()

    corrected_values = []
    for idx, row in df.iterrows():
        k_measured = row['SpringConstant_k']
        if pd.isna(k_measured) or k_measured == '' or k_measured == 0:
            corrected_values.append('')
            continue
        try:
            k_measured = float(k_measured)
            if abs(1/k_measured - 1/k_system) < 1e-10:
                print(f"Well {row.get('Well', idx)}: k_measured ({k_measured:.3f}) too close to k_system ({k_system:.3f}), skipping")
                corrected_values.append(k_measured)
            else:
                k_corrected = 1 / (1/k_measured - 1/k_system)
                corrected_values.append(k_corrected)
        except (ValueError, ZeroDivisionError) as e:
            print(f"Error correcting well {row.get('Well', idx)}: {e}")
            corrected_values.append('')

    df_corrected['SpringConstant_k_Corrected'] = corrected_values

    if output_path is None:
        base, ext = os.path.splitext(csv_path)
        output_path = f"{base}_corrected{ext}"

    df_corrected.to_csv(output_path, index=False)
    print(f"Corrected spring constant data saved to: {output_path}")

    valid_corrected = [k for k in corrected_values if k != '' and not pd.isna(k)]
    if valid_corrected:
        import numpy as np
        print(f"Statistics for corrected spring constants:")
        print(f"   Count: {len(valid_corrected)}")
        print(f"   Mean: {np.mean(valid_corrected):.3f} N/mm")
        print(f"   Std: {np.std(valid_corrected):.3f} N/mm")
        print(f"   Min: {np.min(valid_corrected):.3f} N/mm")
        print(f"   Max: {np.max(valid_corrected):.3f} N/mm")

    return output_path


def print_linear_statistics(results: list, direction: str = ""):
    """Print statistics for linear fit parameters (k and b)."""
    linear_results = [r for r in results if r and getattr(r, 'spring_constant', None) is not None]
    if not linear_results:
        return

    k_values = [getattr(r, 'spring_constant', 0) for r in linear_results]
    b_values = [getattr(r, 'linear_intercept', 0) for r in linear_results]
    r2_values = [getattr(r, 'linear_fit_quality', 0) for r in linear_results]

    if k_values:
        k_mean = sum(k_values) / len(k_values)
        k_std = (sum((k - k_mean) ** 2 for k in k_values) / len(k_values)) ** 0.5
        b_mean = sum(b_values) / len(b_values)
        b_std = (sum((b - b_mean) ** 2 for b in b_values) / len(b_values)) ** 0.5
        r2_mean = sum(r2_values) / len(r2_values)
        r2_std = (sum((r2 - r2_mean) ** 2 for r2 in r2_values) / len(r2_values)) ** 0.5

        print(f"\nLinear Fit Statistics {direction}:")
        print(f"   Spring Constant k: {k_mean:.3f} +/- {k_std:.3f} N/mm (n={len(k_values)})")
        print(f"   Intercept b: {b_mean:.3f} +/- {b_std:.3f} N (n={len(b_values)})")
        print(f"   R2 Quality: {r2_mean:.3f} +/- {r2_std:.3f} (n={len(r2_values)})")


def print_version():
    """Print version information."""
    print(get_full_version())


def _init_gantry() -> Gantry:
    """Create, connect, and return a PANDA_CORE Gantry instance."""
    gantry = Gantry()
    gantry.connect()
    return gantry


def _init_asmi() -> ASMI:
    """Create, connect, and return a PANDA_CORE ASMI instance."""
    asmi = ASMI()
    asmi.connect()
    return asmi


def main(
    home_before_measure: bool = True,
    gantry: Gantry | None = None,
    asmi: ASMI | None = None,
    do_measure: bool = True,
    wells_to_test: list[str] | None = None,
    contact_method: str = "retrospective",
    existing_run_folder: str | None = None,
    generate_heatmap: bool = True,
    measure_with_return: bool = False,
    z_target: float = -15.0,
    step_size: float = 0.02,
    force_limit: float = 5.0,
    well_top_z: float | None = -9.0,
    well_bottom_z: float = -85.0,
    existing_measured_with_return: bool = True,
    show_version: bool = False,
    move_to_pickup: bool = False,
    pickup_position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    fit_method: str = "hertzian",
    apply_system_correction: bool = True,
    retrospective_threshold: float | None = None,
    lock_xy_single_spot: bool = False,
    lock_xy_position: tuple[float, float] | None = None,
    max_depth: float = 0.5,
    min_depth: float = 0.25,
    apply_force_correction: bool = False,
    iterative_d0_refinement: bool = False,
    poisson_ratio: float | None = None,
    use_legacy_height: bool = False,
    legacy_height_step_mm: float = 0.02,
):
    """Parameter-based entry point.

    Args:
        gantry: PANDA_CORE Gantry instance (auto-created if None and do_measure=True).
        asmi: PANDA_CORE ASMI instrument instance (auto-created if None and do_measure=True).
        do_measure: Whether to perform measurements (True) or analyze existing data (False).
        wells_to_test: List of wells to measure (e.g., ["A1", "A2"]) or [None] for current position.
        contact_method: Contact detection method.
        existing_run_folder: Folder name for existing data analysis.
        generate_heatmap: Generate heatmaps after measurements.
        measure_with_return: Enable return measurements (up/down).
        z_target: Target indentation depth (mm).
        step_size: Movement step size (mm).
        force_limit: Force limit (N).
        well_top_z: Well top position before indentation (mm) or None to use current Z.
        well_bottom_z: Well bottom Z (mm); sample height = |contact_z - well_bottom_z|.
        fit_method: Fitting method ("hertzian" or "linear").
        max_depth: Maximum depth (mm) for analysis.
        min_depth: Minimum depth (mm) for Hertzian fit.
        apply_force_correction: Apply geometry-based force correction before Hertzian fit.
        iterative_d0_refinement: Iterative d0 refinement until |d0|<0.01 mm.
        poisson_ratio: Sample Poisson's ratio (None = auto-detect).
    """

    if show_version:
        print_version()
        return

    results = []
    run_folder_name = None

    if do_measure:
        # Ensure hardware is initialized
        if gantry is None:
            gantry = _init_gantry()
        if asmi is None:
            asmi = _init_asmi()

        # Unlock and home
        try:
            gantry.unlock()
        except Exception as e:
            print(f"Unlock failed: {e}")
        try:
            if home_before_measure:
                gantry.home()
        except Exception as e:
            print(f"Homing error: {e}. Proceeding with caution.")

        # Build iteration list
        wells_iter = wells_to_test if wells_to_test is not None else [None]

        # Resolve locked XY position once per run
        resolved_locked_xy: tuple[float, float] | None = None
        if lock_xy_single_spot:
            if lock_xy_position is not None:
                resolved_locked_xy = (float(lock_xy_position[0]), float(lock_xy_position[1]))
            else:
                try:
                    pos0 = gantry.get_coordinates()
                    resolved_locked_xy = (float(pos0["x"]), float(pos0["y"]))
                    print(f"Lock-XY mode enabled: X={resolved_locked_xy[0]:.3f}, Y={resolved_locked_xy[1]:.3f}")
                except Exception as e:
                    print(f"Error determining locked XY: {e}")
                    lock_xy_single_spot = False

        # Measure the wells
        try:
            for w in wells_iter:
                well_param = w.upper() if w is not None else None
                r, run_folder_name = run_measure_analyze_plot(
                    gantry=gantry,
                    asmi=asmi,
                    well=well_param,
                    contact_method=contact_method,
                    measure_with_return=measure_with_return,
                    z_target=z_target,
                    step_size=step_size,
                    force_limit=force_limit,
                    well_top_z=well_top_z,
                    run_folder=os.path.join(_RESULTS_BASE, run_folder_name) if run_folder_name else None,
                    fit_method=fit_method,
                    apply_system_correction=apply_system_correction,
                    retrospective_threshold=retrospective_threshold,
                    lock_xy_single_spot=lock_xy_single_spot,
                    lock_xy_position=resolved_locked_xy,
                    max_depth=max_depth,
                    min_depth=min_depth,
                    apply_force_correction=apply_force_correction,
                    iterative_d0_refinement=iterative_d0_refinement,
                    well_bottom_z=well_bottom_z,
                    poisson_ratio=poisson_ratio,
                    use_legacy_height=use_legacy_height,
                    legacy_height_step_mm=legacy_height_step_mm,
                )
                if r:
                    if isinstance(r, list):
                        results.extend(r)
                    else:
                        results.append(r)
            if not run_folder_name:
                print("No run folder detected; skipping heatmap")
                return
        finally:
            # End-of-run positioning
            try:
                if move_to_pickup:
                    print(f"Moving to pickup position: {pickup_position}")
                    coords = gantry.get_coordinates()
                    gantry.move_to(coords["x"], coords["y"], _SAFE_Z)
                    gantry.move_to(pickup_position[0], pickup_position[1],
                                   pickup_position[2])
                    print(f"Positioned at pickup location")
                else:
                    gantry.home()
            except Exception as e:
                print(f"Error moving to final position: {e}")
                try:
                    gantry.home()
                except Exception as e2:
                    print(f"Homing fallback also failed: {e2}")
    else:
        if not existing_run_folder:
            print("existing_run_folder must be provided when do_measure=False")
            return
        run_folder_name = os.path.basename(existing_run_folder.strip(os.sep))
        run_path = os.path.join(_RESULTS_BASE, run_folder_name)
        if not os.path.isdir(run_path):
            print(f"Run folder not found: {run_path}")
            return
        for fname in sorted(os.listdir(run_path)):
            if fname.startswith("well_") and fname.endswith(".csv"):
                if existing_measured_with_return and not (fname.endswith("_down.csv") or fname.endswith("_up.csv")):
                    continue
                try:
                    parts = fname.split("_")
                    well_core = parts[1]
                    if existing_measured_with_return:
                        suffix = "_down" if fname.endswith("_down.csv") else ("_up" if fname.endswith("_up.csv") else "")
                        well_name = f"{well_core}{suffix}"
                    else:
                        well_name = well_core
                except Exception:
                    continue
                datafile = os.path.join(run_path, fname)
                if well_name.lower().endswith("_down"):
                    r = analyze_file(datafile=datafile, well=f"{well_core.upper()}_down", contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
                elif well_name.lower().endswith("_up"):
                    r = analyze_file(datafile=datafile, well=f"{well_core.upper()}_up", contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
                else:
                    r = analyze_file(datafile=datafile, well=well_core.upper(), contact_method=contact_method, fit_method=fit_method, apply_system_correction=apply_system_correction, retrospective_threshold=retrospective_threshold, max_depth=max_depth, min_depth=min_depth, apply_force_correction=apply_force_correction, iterative_d0_refinement=iterative_d0_refinement, well_bottom_z=well_bottom_z, poisson_ratio=poisson_ratio, use_legacy_height=use_legacy_height, legacy_height_step_mm=legacy_height_step_mm)
                if r:
                    results.append(r)

    if wells_to_test is not None and generate_heatmap and results and run_folder_name:
        plots_root = os.path.join(_PLOTS_BASE, run_folder_name)
        os.makedirs(plots_root, exist_ok=True)

        wants_split_heatmaps = (do_measure and measure_with_return) or (not do_measure and existing_measured_with_return)

        if wants_split_heatmaps:
            down_results = [r for r in results if r and r.well and r.well.lower().endswith("_down")]
            up_results = [r for r in results if r and r.well and r.well.lower().endswith("_up")]

            def write_subset(name: str, subset: list):
                out_csv = os.path.join(plots_root, f"summary_{name}.csv")
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    has_linear = any(getattr(r, 'spring_constant', None) is not None for r in subset if r)
                    if has_linear:
                        w.writerow(["Well", "SpringConstant_k", "Intercept_b", "R2"])
                        for r in subset:
                            if r:
                                name_lower = r.well.lower()
                                if name_lower.endswith("_down"):
                                    well_core = r.well[: -len("_down")]
                                elif name_lower.endswith("_up"):
                                    well_core = r.well[: -len("_up")]
                                else:
                                    well_core = r.well
                                k_val = getattr(r, 'spring_constant', 0)
                                b_val = getattr(r, 'linear_intercept', 0)
                                r2_val = getattr(r, 'linear_fit_quality', getattr(r, 'fit_quality', 0))
                                w.writerow([well_core.upper(), k_val, b_val, r2_val])
                    else:
                        has_system_correction = any(getattr(r, 'original_elastic_modulus', None) is not None for r in subset if r)
                        if has_system_correction:
                            w.writerow(["Well", "ElasticModulus", "ElasticModulus_Original", "Std", "R2", "R2_Original"])
                            for r in subset:
                                if r:
                                    name_lower = r.well.lower()
                                    if name_lower.endswith("_down"):
                                        well_core = r.well[: -len("_down")]
                                    elif name_lower.endswith("_up"):
                                        well_core = r.well[: -len("_up")]
                                    else:
                                        well_core = r.well
                                    orig_E = getattr(r, 'original_elastic_modulus', r.elastic_modulus)
                                    orig_r2 = getattr(r, 'original_fit_quality', r.fit_quality)
                                    w.writerow([well_core.upper(), r.elastic_modulus, orig_E, r.uncertainty, r.fit_quality, orig_r2])
                        else:
                            w.writerow(["Well", "ElasticModulus", "Std", "R2"])
                            for r in subset:
                                if r:
                                    name_lower = r.well.lower()
                                    if name_lower.endswith("_down"):
                                        well_core = r.well[: -len("_down")]
                                    elif name_lower.endswith("_up"):
                                        well_core = r.well[: -len("_up")]
                                    else:
                                        well_core = r.well
                                    w.writerow([well_core.upper(), r.elastic_modulus, r.uncertainty, r.fit_quality])
                return out_csv

            if down_results:
                down_csv = write_subset("down", down_results)
                has_linear = any(getattr(r, 'spring_constant', None) is not None for r in down_results if r)
                if has_linear:
                    plotter.plot_well_heatmap(down_csv, value_col='SpringConstant_k', save_path=os.path.join(plots_root, "well_heatmap_down_spring_constant.png"), convert_to_mpa=False)
                    plotter.plot_well_heatmap(down_csv, value_col='Intercept_b', save_path=os.path.join(plots_root, "well_heatmap_down_intercept.png"), convert_to_mpa=False)
                    print_linear_statistics(down_results, "(Down)")
                else:
                    has_system_correction = any(getattr(r, 'original_elastic_modulus', None) is not None for r in down_results if r)
                    if has_system_correction:
                        plotter.plot_well_heatmap(down_csv, value_col='ElasticModulus', save_path=os.path.join(plots_root, "well_heatmap_down_corrected.png"), title_suffix=" (System Corrected)")
                        plotter.plot_well_heatmap(down_csv, value_col='ElasticModulus_Original', save_path=os.path.join(plots_root, "well_heatmap_down_original.png"), title_suffix=" (Original)")
                        plotter.plot_correction_comparison(down_csv, save_path=os.path.join(plots_root, "correction_comparison_down.png"), convert_to_mpa=True)
                    else:
                        plotter.plot_well_heatmap(down_csv, save_path=os.path.join(plots_root, "well_heatmap_down.png"))
            if up_results:
                up_csv = write_subset("up", up_results)
                has_linear = any(getattr(r, 'spring_constant', None) is not None for r in up_results if r)
                if has_linear:
                    plotter.plot_well_heatmap(up_csv, value_col='SpringConstant_k', save_path=os.path.join(plots_root, "well_heatmap_up_spring_constant.png"), convert_to_mpa=False)
                    plotter.plot_well_heatmap(up_csv, value_col='Intercept_b', save_path=os.path.join(plots_root, "well_heatmap_up_intercept.png"), convert_to_mpa=False)
                    print_linear_statistics(up_results, "(Up)")
                else:
                    has_system_correction = any(getattr(r, 'original_elastic_modulus', None) is not None for r in up_results if r)
                    if has_system_correction:
                        plotter.plot_well_heatmap(up_csv, value_col='ElasticModulus', save_path=os.path.join(plots_root, "well_heatmap_up_corrected.png"), title_suffix=" (System Corrected)")
                        plotter.plot_well_heatmap(up_csv, value_col='ElasticModulus_Original', save_path=os.path.join(plots_root, "well_heatmap_up_original.png"), title_suffix=" (Original)")
                        plotter.plot_correction_comparison(up_csv, save_path=os.path.join(plots_root, "correction_comparison_up.png"), convert_to_mpa=True)
                    else:
                        plotter.plot_well_heatmap(up_csv, save_path=os.path.join(plots_root, "well_heatmap_up.png"))
        else:
            summary_csv = write_summary_csv(run_folder_name, results)
            has_linear = any(getattr(r, 'spring_constant', None) is not None for r in results if r)
            if has_linear:
                plotter.plot_well_heatmap(summary_csv, value_col='SpringConstant_k', save_path=os.path.join(plots_root, "well_heatmap_spring_constant.png"), convert_to_mpa=False)
                plotter.plot_well_heatmap(summary_csv, value_col='Intercept_b', save_path=os.path.join(plots_root, "well_heatmap_intercept.png"), convert_to_mpa=False)
                print_linear_statistics(results)
            else:
                has_system_correction = any(getattr(r, 'original_elastic_modulus', None) is not None for r in results if r)
                if has_system_correction:
                    plotter.plot_well_heatmap(summary_csv, value_col='ElasticModulus', save_path=os.path.join(plots_root, "well_heatmap_corrected.png"), title_suffix=" (System Corrected)")
                    plotter.plot_well_heatmap(summary_csv, value_col='ElasticModulus_Original', save_path=os.path.join(plots_root, "well_heatmap_original.png"), title_suffix=" (Original)")
                else:
                    plotter.plot_well_heatmap(summary_csv, save_path=os.path.join(plots_root, "well_heatmap.png"))

                if has_system_correction:
                    plotter.plot_correction_comparison(summary_csv, save_path=os.path.join(plots_root, "correction_comparison.png"), convert_to_mpa=True)
                    try:
                        tmp_analyzer = IndentationAnalyzer()
                        diag = tmp_analyzer.diagnose_correction_issue(summary_csv)
                        if diag.get('scatter_increased'):
                            print(f"\nWARNING: Scatter increased after correction!")
                            print(f"   Original CV: {diag.get('original_cv', 0):.2f}%")
                            print(f"   Corrected CV: {diag.get('corrected_cv', 0):.2f}%")
                            print(f"   {diag.get('recommendation', 'Check spring constant values in CSV.')}")
                    except Exception as e:
                        print(f"Could not run correction diagnostics: {e}")

    # Also generate raw data plots for the run folder
    if run_folder_name:
        try:
            tmp_analyzer = IndentationAnalyzer()
            tmp_analyzer.plot_raw_data_all_wells(run_folder_name, save_plot=True)
            tmp_analyzer.plot_raw_force_individual_wells(run_folder_name, save_plot=True)
        except Exception as e:
            print(f"Failed to generate raw data plots: {e}")


def run_main_at_intervals(
    interval_seconds: float,
    cycles: int,
    wells_to_test: list[str],
    contact_method: str = "extrapolation",
    measure_with_return: bool = False,
    z_target: float = -15.0,
    step_size: float = 0.02,
    force_limit: float = 5.0,
    well_top_z: float | None = -9.0,
    generate_heatmap: bool = True,
    start_delay: float = 0.0,
    stop_on_error: bool = False,
    move_to_pickup: bool = False,
    pickup_position: tuple[float, float, float] = (0.0, 140.0, 0.0),
    home_before_measure: bool = True,
    fit_method: str = "hertzian",
    max_depth: float = 0.5,
):
    """Run main measurement cycles at regular intervals."""
    print(f"Starting scheduled measurements: {cycles} cycles every {interval_seconds:.1f}s")
    print(f"Wells: {wells_to_test}")
    print(f"Method: {contact_method}, Return: {measure_with_return}")
    print(f"Z-target: {z_target}mm, Step: {step_size}mm, Force limit: {force_limit}N")

    if start_delay > 0:
        print(f"Initial delay: {start_delay:.1f}s...")
        time.sleep(start_delay)

    start_time = time.time()
    successful_cycles = 0
    failed_cycles = 0

    try:
        for i in range(cycles):
            cycle_num = i + 1
            cycle_start_time = start_time + i * interval_seconds
            current_time = time.time()

            if current_time < cycle_start_time:
                wait_time = cycle_start_time - current_time
                print(f"Waiting {wait_time:.1f}s before cycle {cycle_num}/{cycles}...")
                time.sleep(wait_time)

            cycle_actual_start = time.time()
            print(f"\n{'='*60}")
            print(f"Starting cycle {cycle_num}/{cycles} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}")

            try:
                main(
                    do_measure=True,
                    home_before_measure=home_before_measure,
                    wells_to_test=wells_to_test,
                    contact_method=contact_method,
                    measure_with_return=measure_with_return,
                    z_target=z_target,
                    step_size=step_size,
                    force_limit=force_limit,
                    well_top_z=well_top_z,
                    generate_heatmap=generate_heatmap,
                    move_to_pickup=move_to_pickup,
                    pickup_position=pickup_position,
                    fit_method=fit_method,
                    max_depth=max_depth,
                )

                cycle_duration = time.time() - cycle_actual_start
                successful_cycles += 1
                print(f"Cycle {cycle_num} completed in {cycle_duration:.1f}s")

            except KeyboardInterrupt:
                print(f"\nKeyboard interrupt during cycle {cycle_num}")
                print(f"Completed {successful_cycles}/{cycles} cycles")
                raise

            except Exception as e:
                failed_cycles += 1
                print(f"Cycle {cycle_num} failed: {e}")

                if stop_on_error:
                    print("Stopping due to error (stop_on_error=True)")
                    break
                else:
                    print("Continuing with next cycle...")

            if cycle_num < cycles:
                next_cycle_time = start_time + cycle_num * interval_seconds
                current_time = time.time()
                time_until_next = next_cycle_time - current_time

                if time_until_next > 0:
                    print(f"Waiting {time_until_next:.1f}s until next cycle...")
                    time.sleep(time_until_next)
                else:
                    print(f"Running behind schedule by {abs(time_until_next):.1f}s")

    except KeyboardInterrupt:
        print("\nScheduled measurements interrupted by user")

    finally:
        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("SCHEDULED MEASUREMENTS SUMMARY")
        print(f"{'='*60}")
        print(f"Successful cycles: {successful_cycles}/{cycles}")
        print(f"Failed cycles: {failed_cycles}")
        print(f"Total time: {total_time/60:.1f} minutes")
        print(f"Success rate: {successful_cycles/cycles*100:.1f}%")
        print(f"Average cycle time: {total_time/cycles:.1f}s")
        print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")


if __name__ == "__main__":
    exp = load_experiment()
    m = exp.get('measurement', {})
    w = exp.get('wells', {})
    a = exp.get('analysis', {})
    wf = exp.get('workflow', {})

    do_measure = wf.get('do_measure', False)

    gantry = None
    asmi = None
    if do_measure:
        gantry = _init_gantry()
        asmi = _init_asmi()

    main(
        gantry=gantry,
        asmi=asmi,
        do_measure=do_measure,
        home_before_measure=wf.get('home_before_measure', True),
        wells_to_test=w.get('wells_to_test'),
        contact_method=a.get('contact_method', 'retrospective'),
        retrospective_threshold=a.get('retrospective_threshold'),
        fit_method=a.get('fit_method', 'hertzian'),
        measure_with_return=m.get('measure_with_return', False),
        move_to_pickup=wf.get('move_to_pickup', False),
        pickup_position=tuple(wf.get('pickup_position', [0.0, 0.0, 0.0])),
        step_size=m.get('step_size', 0.01),
        z_target=m.get('z_target', -15.0),
        force_limit=m.get('force_limit', 5.0),
        well_top_z=m.get('well_top_z', -9.0),
        well_bottom_z=m.get('well_bottom_z', -85.0),
        existing_run_folder=wf.get('existing_run_folder'),
        existing_measured_with_return=wf.get('existing_measured_with_return', False),
        apply_system_correction=a.get('apply_system_correction', True),
        max_depth=a.get('max_depth', 0.5),
        min_depth=a.get('min_depth', 0.25),
        poisson_ratio=a.get('poisson_ratio'),
        apply_force_correction=a.get('apply_force_correction', False),
        iterative_d0_refinement=a.get('iterative_d0_refinement', False),
        use_legacy_height=a.get('use_legacy_height', False),
        legacy_height_step_mm=a.get('legacy_height_step_mm', 0.02),
        generate_heatmap=wf.get('generate_heatmap', True),
    )
