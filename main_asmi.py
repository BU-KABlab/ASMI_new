#!/usr/bin/env python3
"""
ASMI v2 runner (parameter-based, no CLI args) - secondary entrypoint

Supports two workflows:
  1) Measure → Analyze → Plot (default measurement: simple_indentation_measurement)
  2) Analyze existing data folder → Plot

Also supports splitting direction-tagged measurements into _down/_up CSVs and per-direction analysis/plots.

Author: [Your Name/Institution]
Date: 2024
License: MIT
"""

import os
import csv
import time
from datetime import datetime

from src.force_monitoring import (
    simple_indentation_measurement,
    simple_indentation_with_return_measurement,
    get_and_increment_run_count,
)
from src.analysis import IndentationAnalyzer
from src.plot import plotter


def ensure_run_folder(base: str = "results/measurements") -> str:
    """Create and return a new run folder path under base."""
    run_count = get_and_increment_run_count(os.path.join("src", "run_count.txt"))
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = os.path.join(base, f"run_{run_count:03d}_{run_date}")
    os.makedirs(run_folder, exist_ok=True)
    return run_folder


def plot_results_via_plotter(result, run_folder: str | None, method: str | None = None, direction_label: str | None = None):
    """Thin wrapper to call plotter.plot_results with common defaults."""
    try:
        plotter.plot_results(result, save_plot=True, run_folder=run_folder, method=method, direction_label=direction_label)
    except TypeError:
        # Backward compatibility if plotter doesn't accept method/direction_label
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
        print(f"⚠️ Failed to read for splitting: {orig_csv_path}: {e}")
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
        print("⚠️ No data rows to split.")
        return None, None

    down_rows: list[list[str]] = []
    up_rows: list[list[str]] = []
    for r in data_rows:
        direction = r[4] if len(r) >= 5 else 'down'
        if direction == 'up':
            up_rows.append(r)
        else:
            down_rows.append(r)

    # Sort 'up' by increasing |Z|
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
        print(f"⚠️ Failed to write split files: {e}")

    return down_path, up_path


def analyze_file(datafile: str, well: str, contact_method: str = "extrapolation"):
    """Analyze a single CSV file and emit plots. Compatible with current src.analysis."""
    data_dir, filename = os.path.split(datafile)
    analyzer = IndentationAnalyzer(data_dir or ".")
    if not analyzer.load_data(filename):
        return None

    # Map friendly names → analyzer keys (best-effort if supported)
    method_key = {
        "extrapolation": "true_contact",
        "retrospective": "retrospective",
        "simple_threshold": "simple_threshold",
    }.get(contact_method, "true_contact")

    try:
        result = analyzer.analyze_well(
            well=well,
            poisson_ratio=None,  # auto-detect from file
            filename=datafile,
            contact_method=method_key,
        )
    except TypeError:
        # Fall back if analyze_well does not accept contact_method
        result = analyzer.analyze_well(
            well=well,
            poisson_ratio=None,
            filename=datafile,
        )

    if not result:
        print("❌ Analysis failed")
        return None

    # Derive run_folder from data path for plotting
    run_folder = None
    for part in data_dir.split(os.sep):
        if part.startswith("run_"):
            run_folder = part
            break

    # Infer direction from well suffix if present
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
        }.get(contact_method, "extrapolation")
        plot_results_via_plotter(result, run_folder, method=method_for_plot, direction_label=dir_label)
    except Exception:
        plot_results_via_plotter(result, run_folder)
    return result


def run_measure_analyze_plot(
    cnc,
    force_sensor,
    well: str,
    contact_method: str,
    measure_with_return: bool = False,
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
    run_folder: str | None = None,
):
    """Measure a single well, then analyze and plot (handles split up/down files automatically)."""
    # Use provided batch run folder or create one if missing
    run_folder = run_folder or ensure_run_folder()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    datafile = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")

    # Move to the requested well (XY) at safety Z before measuring
    col = ''.join([c for c in well if c.isalpha()]).upper()
    row = ''.join([c for c in well if c.isdigit()])
    if col and row:
        try:
            cnc.move_to_well(col, row, z=0.0)
        except Exception as e:
            print(f"⚠️ Could not move to well {well}: {e}")

    try:
        t0 = time.time()
        if measure_with_return:
            ok = simple_indentation_with_return_measurement(
                cnc=cnc,
                force_sensor=force_sensor,
                well=well,
                filename=datafile,
                run_folder=run_folder,
                z_target=z_target,
                step_size=step_size,
                force_limit=force_limit,
                well_top_z=well_top_z,  # Move to well top before indentation
            )
        else:
            ok = simple_indentation_measurement(
                cnc=cnc,
                force_sensor=force_sensor,
                well=well,
                filename=datafile,
                run_folder=run_folder,
                z_target=z_target,
                step_size=step_size,
                force_limit=force_limit,
                well_top_z=well_top_z,  # Move to well top before indentation
            )
        if not ok:
            print("❌ Measurement failed")
            return None, None

        duration_s = time.time() - t0
        print(f"✅ Measurement saved to: {datafile}")
        print(f"⏱️ Total measurement time: {duration_s:.2f} s")
        # Append total measurement time to CSV metadata
        try:
            with open(datafile, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Total_Measurement_Time(s)', f"{duration_s:.3f}"])
        except Exception as e:
            print(f"⚠️ Could not append total time to CSV: {e}")

        # Split into _down and _up CSVs and analyze each with well suffix
        down_csv, up_csv = split_up_down_csv(datafile)
        per_well_results = []
        if down_csv:
            r_down = analyze_file(datafile=down_csv, well=f"{well}_down", contact_method=contact_method)
            if r_down:
                per_well_results.append(r_down)
        if up_csv:
            r_up = analyze_file(datafile=up_csv, well=f"{well}_up", contact_method=contact_method)
            if r_up:
                per_well_results.append(r_up)

        return per_well_results, os.path.basename(run_folder)
    except KeyboardInterrupt:
        print("🛑 Keyboard interrupt received.")
        raise


def write_summary_csv(run_folder_name: str, results: list):
    """Write summary.csv for heatmap plotting under results/plots/<run_folder_name>/."""
    plots_root = os.path.join("results", "plots")
    out_dir = os.path.join(plots_root, run_folder_name)
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "summary.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Well", "ElasticModulus", "Std", "R2"])  # Std = uncertainty
        for r in results:
            if r:
                w.writerow([r.well, r.elastic_modulus, r.uncertainty, r.fit_quality])
    print(f"💾 Summary CSV written: {out_csv}")
    return out_csv


def main(
    do_measure: bool = True,
    wells_to_test: list[str] | None = None,
    contact_method: str = "extrapolation",
    existing_run_folder: str | None = None,
    generate_heatmap: bool = True,
    measure_with_return: bool = False,
    z_target: float = -15.0,
    step_size: float = 0.02,
    force_limit: float = 5.0,
    well_top_z: float = -9.0,
    existing_measured_with_return: bool = True,
):
    """Parameter-based entry point."""

    results = []
    run_folder_name = None

    if do_measure:
        # Setup shared hardware (home once before the batch)
        from src.CNCController import CNCController
        from src.ForceSensor import ForceSensor
        cnc = CNCController()
        # home the cnc first
        try:
            if not cnc.home(zero_after=True):
                print("⚠️ Homing failed or timed out, attempting position sync...")
                cnc.sync_position()
        except Exception as e:
            print(f"⚠️ Homing error: {e}. Proceeding with caution.")
        force_sensor = ForceSensor()

        if not wells_to_test:
            wells_to_test = ["A1"]
        try:
            # move to the well top position
            # cnc.move_to_z(well_top_z)
            # cnc.wait_for_idle()
            for w in wells_to_test:
                r, run_folder_name = run_measure_analyze_plot(
                    cnc=cnc,
                    force_sensor=force_sensor,
                    well=w.upper(),
                    contact_method=contact_method,
                    measure_with_return=measure_with_return,
                    z_target=z_target,
                    step_size=step_size,
                    force_limit=force_limit,
                    well_top_z=well_top_z,
                    run_folder=os.path.join("results", "measurements", run_folder_name) if run_folder_name else None,
                )
                if r:
                    if isinstance(r, list):
                        results.extend(r)
                    else:
                        results.append(r)
            if not run_folder_name:
                print("⚠️ No run folder detected; skipping heatmap")
                return
        finally:
            # Home once after all measurements are completed
            try:
                cnc.home(zero_after=True)
            except Exception as e:
                print(f"⚠️ Homing error after batch: {e}")
    else:
        if not existing_run_folder:
            print("❌ existing_run_folder must be provided when do_measure=False")
            return
        run_folder_name = os.path.basename(existing_run_folder.strip(os.sep))
        run_path = os.path.join("results", "measurements", run_folder_name)
        if not os.path.isdir(run_path):
            print(f"❌ Run folder not found: {run_path}")
            return
        # Analyze all well CSVs
        for fname in sorted(os.listdir(run_path)):
            if fname.startswith("well_") and fname.endswith(".csv"):
                # If data were measured with return, only analyze direction-specific files
                if existing_measured_with_return and not (fname.endswith("_down.csv") or fname.endswith("_up.csv")):
                    continue
                # Parse well name from filename well_<WELL>_*.csv
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
                    r = analyze_file(datafile=datafile, well=f"{well_core.upper()}_down", contact_method=contact_method)
                elif well_name.lower().endswith("_up"):
                    r = analyze_file(datafile=datafile, well=f"{well_core.upper()}_up", contact_method=contact_method)
                else:
                    r = analyze_file(datafile=datafile, well=well_core.upper(), contact_method=contact_method)
                if r:
                    results.append(r)

    if generate_heatmap and results and run_folder_name:
        plots_root = os.path.join("results", "plots", run_folder_name)
        os.makedirs(plots_root, exist_ok=True)

        wants_split_heatmaps = (do_measure and measure_with_return) or (not do_measure and existing_measured_with_return)

        if wants_split_heatmaps:
            down_results = [r for r in results if r and r.well and r.well.lower().endswith("_down")]
            up_results = [r for r in results if r and r.well and r.well.lower().endswith("_up")]

            def write_subset(name: str, subset: list):
                out_csv = os.path.join(plots_root, f"summary_{name}.csv")
                with open(out_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Well", "ElasticModulus", "Std", "R2"])  # Std = uncertainty
                    for r in subset:
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
                plotter.plot_well_heatmap(down_csv, save_path=os.path.join(plots_root, "well_heatmap_down.png"))
            if up_results:
                up_csv = write_subset("up", up_results)
                plotter.plot_well_heatmap(up_csv, save_path=os.path.join(plots_root, "well_heatmap_up.png"))
        else:
            # Legacy data: generate a single combined heatmap
            summary_csv = write_summary_csv(run_folder_name, results)
            plotter.plot_well_heatmap(summary_csv, save_path=os.path.join(plots_root, "well_heatmap.png"))

    # Also generate raw data plots for the run folder
    if run_folder_name:
        try:
            tmp_analyzer = IndentationAnalyzer()
            tmp_analyzer.plot_raw_data_all_wells(run_folder_name, save_plot=True)
            tmp_analyzer.plot_raw_force_individual_wells(run_folder_name, save_plot=True)
        except Exception as e:
            print(f"⚠️ Failed to generate raw data plots: {e}")


def measure_at_intervals(
    interval_seconds: float,
    cycles: int,
    wells_to_test: list[str],
    contact_method: str = "extrapolation",
    measure_with_return: bool = False,
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
):
    """Measure → Analyze → Plot repeatedly at a fixed time gap (homes before first and after last cycle)."""
    from src.CNCController import CNCController
    from src.ForceSensor import ForceSensor

    if not wells_to_test:
        wells_to_test = ["A1"]

    cnc = CNCController()
    try:
        try:
            if not cnc.home(zero_after=True):
                print("⚠️ Homing failed or timed out, attempting position sync...")
                cnc.sync_position()
        except Exception as e:
            print(f"⚠️ Homing error: {e}. Proceeding with caution.")

        force_sensor = ForceSensor()

        start0 = time.time()
        for i in range(cycles):
            cycle_start_target = start0 + i * interval_seconds
            now = time.time()
            if now < cycle_start_target:
                wait_s = cycle_start_target - now
                print(f"⏳ Waiting {wait_s:.1f}s before cycle {i+1}/{cycles}...")
                time.sleep(max(0, wait_s))

            print(f"\n▶️ Starting cycle {i+1}/{cycles} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            results = []
            run_folder_name = None

            for w in [w.upper() for w in wells_to_test]:
                r, run_folder_name = run_measure_analyze_plot(
                    cnc=cnc,
                    force_sensor=force_sensor,
                    well=w,
                    contact_method=contact_method,
                    measure_with_return=measure_with_return,
                    z_target=z_target,
                    step_size=step_size,
                    force_limit=force_limit,
                    well_top_z=well_top_z,
                    run_folder=os.path.join("results", "measurements", run_folder_name) if run_folder_name else None,
                )
                if r:
                    if isinstance(r, list):
                        results.extend(r)
                    else:
                        results.append(r)

            if not run_folder_name:
                print("⚠️ No run folder detected for this cycle; skipping plotting")
                continue

            plots_root = os.path.join("results", "plots", run_folder_name)
            os.makedirs(plots_root, exist_ok=True)
            wants_split_heatmaps = measure_with_return

            if wants_split_heatmaps:
                down_results = [r for r in results if r and r.well and r.well.lower().endswith("_down")]
                up_results = [r for r in results if r and r.well and r.well.lower().endswith("_up")]

                def _write_subset(name: str, subset: list):
                    out_csv = os.path.join(plots_root, f"summary_{name}.csv")
                    with open(out_csv, "w", newline="") as f:
                        w = csv.writer(f)
                        w.writerow(["Well", "ElasticModulus", "Std", "R2"])  # Std = uncertainty
                        for r in subset:
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
                    down_csv = _write_subset("down", down_results)
                    plotter.plot_well_heatmap(down_csv, save_path=os.path.join(plots_root, "well_heatmap_down.png"))
                if up_results:
                    up_csv = _write_subset("up", up_results)
                    plotter.plot_well_heatmap(up_csv, save_path=os.path.join(plots_root, "well_heatmap_up.png"))
            else:
                summary_csv = write_summary_csv(run_folder_name, results)
                plotter.plot_well_heatmap(summary_csv, save_path=os.path.join(plots_root, "well_heatmap.png"))

            try:
                tmp_analyzer = IndentationAnalyzer()
                tmp_analyzer.plot_raw_data_all_wells(run_folder_name, save_plot=True)
                tmp_analyzer.plot_raw_force_individual_wells(run_folder_name, save_plot=True)
            except Exception as e:
                print(f"⚠️ Failed to generate raw data plots for cycle: {e}")
    finally:
        try:
            cnc.home(zero_after=True)
        except Exception as e:
            print(f"⚠️ Homing error after scheduled cycles: {e}")


if __name__ == "__main__":
    # Example usage
    # main(do_measure=True, wells_to_test=["A1", "A2"], contact_method="extrapolation", measure_with_return=True)
    # Or analyze existing run:
    # main(do_measure=False, existing_run_folder="run_460_20250911_062621", existing_measured_with_return=True)
    # from src.CNCController import CNCController
    # cnc = CNCController()
    # cnc.home(zero_after=True)
    wells_to_test = ["B11", "C11"]
    main(do_measure=False, existing_run_folder='run_463_20250917_000017', wells_to_test=wells_to_test, contact_method="retrospective", measure_with_return=True)


