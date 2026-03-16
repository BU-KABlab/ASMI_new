#!/usr/bin/env python3
"""
Force monitoring module for ASMI system.

Provides measurement functions for step-by-step indentation with force
monitoring.  Uses PANDA_CORE's Gantry for motion control and ASMI
instrument driver for force sensing.

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import time
import csv
import os
from datetime import datetime


def _wait_for_idle(gantry, timeout=10.0):
    """Poll gantry status until idle. Returns True if idle reached."""
    start = time.time()
    while time.time() - start < timeout:
        if "Idle" in gantry.get_status():
            return True
        time.sleep(0.02)
    return False


def _move_z(gantry, pos, z):
    """Move gantry to a new Z using current XY from pos tuple, then wait."""
    gantry.move_to(pos[0], pos[1], z)
    _wait_for_idle(gantry)


def _get_position(gantry):
    """Get gantry position as (x, y, z), returning None on error."""
    try:
        coords = gantry.get_coordinates()
        return (coords["x"], coords["y"], coords["z"])
    except Exception:
        return None


def _move_to_safe_z(gantry, safe_z):
    """Move gantry Z to safety height."""
    coords = gantry.get_coordinates()
    gantry.move_to(coords["x"], coords["y"], safe_z)
    _wait_for_idle(gantry)


def get_and_increment_run_count(run_count_file):
    """Get and increment the run count from file."""
    if not os.path.exists(run_count_file):
        with open(run_count_file, 'w') as f:
            f.write('1')
        return 1
    with open(run_count_file, 'r+') as f:
        count = int(f.read().strip() or '0')
        count += 1
        f.seek(0)
        f.write(str(count))
        f.truncate()
    return count


def _generate_filename(well, run_folder, results_base, run_count_file,
                       prefix="indentation"):
    """Generate an output filename, creating the run folder if needed."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_folder is None:
        run_count = get_and_increment_run_count(run_count_file)
        run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_folder = os.path.join(results_base,
                                  f"run_{run_count:03d}_{run_date}")
    os.makedirs(run_folder, exist_ok=True)
    if well is not None:
        filename = os.path.join(run_folder, f"well_{well}_{ts}.csv")
    else:
        filename = os.path.join(run_folder, f"{prefix}_{ts}.csv")
    return filename


def _position_for_measurement(gantry, well_xy, well_top_z, safe_z,
                              locked_xy):
    """Move gantry to the measurement start position.

    Args:
        gantry: PANDA_CORE Gantry instance.
        well_xy: (x, y) for the target well, or None for current position.
        well_top_z: Z to start measurement from.
        safe_z: gantry safety height.
        locked_xy: (x, y) override, or None.

    Returns True on success, False on failure.
    """
    if well_xy is not None:
        if locked_xy is not None:
            try:
                print(f"Locked-XY mode: moving to X={locked_xy[0]:.3f}, "
                      f"Y={locked_xy[1]:.3f}, then Z={well_top_z:.1f}mm...")
                _move_to_safe_z(gantry, safe_z)
                gantry.move_to(locked_xy[0], locked_xy[1], safe_z)
                _wait_for_idle(gantry)
                gantry.move_to(locked_xy[0], locked_xy[1], well_top_z)
                _wait_for_idle(gantry)
                print("Positioned at locked XY")
            except Exception as e:
                print(f"Could not move to locked XY: {e}")
                return False
        else:
            try:
                print(f"Moving to well at X={well_xy[0]:.3f}, "
                      f"Y={well_xy[1]:.3f}, Z={well_top_z:.1f}mm...")
                gantry.move_to(well_xy[0], well_xy[1], safe_z)
                _wait_for_idle(gantry)
                if well_top_z != safe_z:
                    gantry.move_to(well_xy[0], well_xy[1], well_top_z)
                    _wait_for_idle(gantry)
                print("Positioned at well")
            except Exception as e:
                print(f"Could not move to well: {e}")
                return False
    else:
        try:
            print(f"Moving to Z={well_top_z:.1f}mm at current XY...")
            coords = gantry.get_coordinates()
            gantry.move_to(coords["x"], coords["y"], well_top_z)
            _wait_for_idle(gantry)
            print("Positioned at current position")
        except Exception as e:
            print(f"Could not move to well top position: {e}")
            return False
    return True


def simple_indentation_measurement(
    gantry,
    asmi,
    well: str | None = None,
    well_xy: tuple[float, float] | None = None,
    safe_z: float = -50.0,
    filename: str | None = None,
    run_folder: str | None = None,
    results_base: str = "results/measurements",
    run_count_file: str = "src/run_count.txt",
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
    locked_xy: tuple[float, float] | None = None,
):
    """Measure force during downward indentation until z_target or force_limit.

    Args:
        gantry: PANDA_CORE Gantry instance.
        asmi: PANDA_CORE ASMI instrument instance.
        well: Well identifier for CSV metadata (e.g., "A1").
        well_xy: (x, y) coordinates for the well, or None for current pos.
        safe_z: Gantry safety height (mm).
        filename: Output filename (auto-generated if None).
        run_folder: Run folder for saving data.
        results_base: Base directory for results.
        run_count_file: Path to run count file.
        z_target: Target Z position for indentation (mm).
        step_size: Step size for movement (mm).
        force_limit: Force limit in N.
        well_top_z: Z position at well top before indentation (mm).
        locked_xy: Optional (x, y) to lock XY for all wells.
    """
    try:
        pos = _get_position(gantry)
        if not pos:
            print("Could not get current position from gantry")
            return False
        if not asmi.is_connected():
            print("Force sensor not connected")
            return False

        baseline_avg, baseline_std = asmi.get_baseline_force(samples=10)
        print(f"Baseline: {baseline_avg:.3f} +/- {baseline_std:.3f} N")

        if filename is None:
            filename = _generate_filename(well, run_folder, results_base,
                                          run_count_file)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not _position_for_measurement(gantry, well_xy, well_top_z,
                                         safe_z, locked_xy):
            return False

        measurements: list[list[float]] = []
        data_count = 0

        while True:
            current = _get_position(gantry)
            if not current:
                print("Could not get position - stopping measurement")
                break
            current_z = float(current[2])
            if current_z <= z_target:
                print(f"Reached z_target {z_target:.3f}mm")
                break
            next_z = current_z - step_size
            _move_z(gantry, current, next_z)

            current = _get_position(gantry) or (None, None, next_z)
            force = asmi.get_force_reading()
            corrected = force - baseline_avg
            data_count += 1
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected])
            if data_count % 10 == 0:
                try:
                    print(f"Step #{data_count}: Z={float(current[2]):.3f}mm, "
                          f"F={force:.3f}N, dF={corrected:.3f}N")
                except Exception:
                    pass
            if abs(corrected) > force_limit:
                print(f"Force limit exceeded: {corrected:.3f}N > "
                      f"{force_limit:.1f}N")
                break

        _move_to_safe_z(gantry, safe_z)

        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Test_Time',
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                w.writerow(['Well', well])
            w.writerow(['Target_Z(mm)', f"{z_target:.3f}"])
            w.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            w.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            w.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            w.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            w.writerow(['Force_Exceeded',
                         str(bool(measurements and
                                  abs(measurements[-1][3]) > force_limit))])
            w.writerow([])
            w.writerow(['Timestamp(s)', 'Z_Position(mm)',
                         'Raw_Force(N)', 'Corrected_Force(N)'])
            for t, z, rf, cf in measurements:
                w.writerow([f"{t:.3f}", f"{z:.3f}",
                            f"{rf:.3f}", f"{cf:.3f}"])
        print(f"Saved {len(measurements)} points to {filename}")
        return True
    except Exception as e:
        print(f"Error in simple_indentation_measurement: {e}")
        return False


def simple_indentation_with_return_measurement(
    gantry,
    asmi,
    well: str | None = None,
    well_xy: tuple[float, float] | None = None,
    safe_z: float = -50.0,
    filename: str | None = None,
    run_folder: str | None = None,
    results_base: str = "results/measurements",
    run_count_file: str = "src/run_count.txt",
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
    locked_xy: tuple[float, float] | None = None,
):
    """Measure during downward and upward (return) movement.

    Same args as simple_indentation_measurement. Adds 'Direction' column.
    """
    try:
        pos = _get_position(gantry)
        if not pos:
            print("Could not get current position from gantry")
            return False
        if not asmi.is_connected():
            print("Force sensor not connected")
            return False

        baseline_avg, baseline_std = asmi.get_baseline_force(samples=10)
        print(f"Baseline: {baseline_avg:.3f} +/- {baseline_std:.3f} N")

        if filename is None:
            filename = _generate_filename(well, run_folder, results_base,
                                          run_count_file)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        if not _position_for_measurement(gantry, well_xy, well_top_z,
                                         safe_z, locked_xy):
            return False

        measurements: list[list[object]] = []

        # Downward
        while True:
            current = _get_position(gantry)
            if not current:
                print("Could not get position - stopping measurement")
                break
            current_z = float(current[2])
            if current_z <= z_target:
                print(f"Reached z_target {z_target:.3f}mm")
                break
            next_z = current_z - step_size
            _move_z(gantry, current, next_z)
            current = _get_position(gantry) or (None, None, next_z)
            force = asmi.get_force_reading()
            corrected = force - baseline_avg
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected,
                                 'down'])
            if len(measurements) % 10 == 0:
                try:
                    print(f"Down #{len(measurements)}: "
                          f"Z={float(current[2]):.3f}mm, F={force:.3f}N, "
                          f"dF={corrected:.3f}N")
                except Exception:
                    pass
            if abs(corrected) > force_limit:
                print(f"Force limit exceeded: {corrected:.3f}N > "
                      f"{force_limit:.1f}N")
                break

        # Upward return
        while True:
            current = _get_position(gantry)
            if not current:
                break
            current_z = float(current[2])
            if current_z >= well_top_z:
                break
            next_z = min(current_z + step_size, well_top_z)
            _move_z(gantry, current, next_z)
            current = _get_position(gantry) or (None, None, next_z)
            force = asmi.get_force_reading()
            corrected = force - baseline_avg
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected, 'up'])
            if len(measurements) % 10 == 0:
                try:
                    print(f"Up #{len(measurements)}: "
                          f"Z={float(current[2]):.3f}mm, F={force:.3f}N, "
                          f"dF={corrected:.3f}N")
                except Exception:
                    pass

        _move_to_safe_z(gantry, safe_z)

        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Test_Time',
                         datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                w.writerow(['Well', well])
            w.writerow(['Target_Z(mm)', f"{z_target:.3f}"])
            w.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            w.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            w.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            w.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            w.writerow(['Force_Exceeded',
                         str(any(abs(m[3]) > force_limit
                                 for m in measurements))])
            w.writerow([])
            w.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)',
                         'Corrected_Force(N)', 'Direction'])
            for t, z, rf, cf, d in measurements:
                w.writerow([f"{t:.3f}", f"{z:.3f}",
                            f"{rf:.3f}", f"{cf:.3f}", d])
        print(f"Saved {len(measurements)} points (down+up) to {filename}")
        return True
    except Exception as e:
        print(f"Error in simple_indentation_with_return_measurement: {e}")
        return False
