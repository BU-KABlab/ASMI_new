#!/usr/bin/env python3
"""
Force monitoring module for ASMI system
Provides functionality to test force monitoring at multiple wells with force limits

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import time
import csv
import os
from datetime import datetime
from .version import __version__
from .CNCController import CNCController
from .ForceSensor import ForceSensor
from .Analysis import IndentationAnalyzer


def get_and_increment_run_count(run_count_file='src/run_count.txt'):
    """Get and increment the run count from file"""
    if not os.path.exists(run_count_file):
        with open(run_count_file, 'w') as f:
            f.write('1')
        return 1
    with open(run_count_file, 'r+') as f:
        count = int(f.read().strip() or '0')
        count += 1
        f.seek(0) # reset file pointer to the beginning
        f.write(str(count))
        f.truncate() # truncate the file to the current length
    return count


def simple_indentation_measurement(
    cnc,
    force_sensor,
    well: str | None = None,
    filename: str | None = None,
    run_folder: str | None = None,
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
    locked_xy: tuple[float, float] | None = None,
):
    """Measure force during downward indentation until z_target or force_limit.

    Args:
        cnc: CNCController object
        force_sensor: ForceSensor object
        well: Well identifier (e.g., "A1")
        filename: Output filename (auto-generated if None)
        run_folder: Run folder for saving data
        z_target: Target Z position for indentation (default: -17.0 mm)
        step_size: Step size for movement (default: 0.01 mm)
        force_limit: Force limit in N (default: 15.0 N)
        well_top_z: Z position at well top before indentation (default: -9.0 mm)
        locked_xy: Optional (x, y) to lock XY for all wells at well_top_z

    Writes CSV with metadata and columns: Timestamp(s), Z_Position(mm), Raw_Force(N), Corrected_Force(N).
    """
    try:
        # Connectivity
        pos = cnc.get_current_position()
        if not pos:
            print("❌ Could not get current position from CNC")
            return False
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return False

        # Baseline
        baseline_avg, baseline_std = force_sensor.get_baseline_force(samples=10)
        print(f"📊 Baseline: {baseline_avg:.3f} ± {baseline_std:.3f} N")

        # Filename
        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if well is not None:
                if run_folder is None:
                    run_count = get_and_increment_run_count()
                    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"well_{well}_{ts}.csv")
            else:
                run_count = get_and_increment_run_count()
                run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"indentation_{ts}.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Move to well top position first (optionally lock XY)
        if well is not None:
            if locked_xy is not None:
                try:
                    # lock_xy: override well XY with fixed coordinates
                    # 1) raise to safety Z, 2) move XY at safety, 3) go down to well_top_z
                    print(f"📍 Locked-XY mode: moving to safety Z, then X={locked_xy[0]:.3f}, Y={locked_xy[1]:.3f}, then Z={well_top_z:.1f}mm for well {well}...")
                    cnc.move_to_safe_z()
                    cnc.move_to_x_y(locked_xy[0], locked_xy[1])
                    cnc.move_to_z(well_top_z, wait_for_idle=True)
                    print(f"✅ Positioned at locked XY for well {well}")
                except Exception as e:
                    print(f"⚠️ Could not move to locked XY for well {well}: {e}")
                    return False
            else:
                col = ''.join([c for c in well if c.isalpha()]).upper()
                row = ''.join([c for c in well if c.isdigit()])
                if col and row:
                    try:
                        print(f"📍 Moving to well {well} at top position Z={well_top_z:.1f}mm...")
                        cnc.move_to_well(col, row, z=well_top_z)
                        print(f"✅ Positioned at well {well} top")
                    except Exception as e:
                        print(f"⚠️ Could not move to well {well}: {e}")
                        return False
        else: # well is None, so measure at current position
            print(f"📍 Moving to current position Z={well_top_z:.1f}mm...")
            cnc.move_to_z(well_top_z)
            print(f"✅ Positioned at current position")

        measurements: list[list[float]] = []
        data_count = 0

        # Downward loop
        while True:
            current = cnc.get_current_position()
            if not current:
                print("❌ Could not get position - stopping measurement")
                break
            current_z = float(current[2])
            if current_z <= z_target:
                print(f"🎯 Reached z_target {z_target:.3f}mm")
                break
            next_z = current_z - step_size
            # Use low feedrate so each step finishes quickly and precisely
            cnc.move_to_z(next_z, wait_for_idle=True)
            
            current = cnc.get_current_position() or (None, None, next_z)
            force = force_sensor.get_force_reading()
            corrected = force - baseline_avg
            data_count += 1
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected])
            # Progress print every 10 steps
            if data_count % 10 == 0:
                try:
                    print(f"📉 Step #{data_count}: Z={float(current[2]):.3f}mm, F={force:.3f}N, dF={corrected:.3f}N")
                except Exception:
                    pass
            if abs(corrected) > force_limit:
                print(f"🛑 Force limit exceeded: {corrected:.3f}N > {force_limit:.1f}N")
                break

        # Return to safety height before moving to next well
        cnc.move_to_safe_z()

        # Write CSV
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                w.writerow(['Well', well])
            w.writerow(['Target_Z(mm)', f"{z_target:.3f}"])
            w.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            w.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            w.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            w.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            w.writerow(['Force_Exceeded', str(bool(measurements and abs(measurements[-1][3]) > force_limit))])
            w.writerow([])
            w.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)'])
            for t, z, rf, cf in measurements:
                w.writerow([f"{t:.3f}", f"{z:.3f}", f"{rf:.3f}", f"{cf:.3f}"])
        print(f"💾 Saved {len(measurements)} points to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error in simple_indentation_measurement: {e}")
        return False


def simple_indentation_with_return_measurement(
    cnc,
    force_sensor,
    well: str | None = None,
    filename: str | None = None,
    run_folder: str | None = None,
    z_target: float = -17.0,
    step_size: float = 0.01,
    force_limit: float = 15.0,
    well_top_z: float = -9.0,
    locked_xy: tuple[float, float] | None = None,
):
    """Measure during downward and upward (return) movement; include 'Direction' column.

    Args:
        cnc: CNCController object
        force_sensor: ForceSensor object
        well: Well identifier (e.g., "A1")
        filename: Output filename (auto-generated if None)
        run_folder: Run folder for saving data
        z_target: Target Z position for indentation (default: -17.0 mm)
        step_size: Step size for movement (default: 0.01 mm)
        force_limit: Force limit in N (default: 15.0 N)
        well_top_z: Z position at well top before indentation (default: -9.0 mm)
        locked_xy: Optional (x, y) to lock XY for all wells at well_top_z

    Data header: Timestamp(s), Z_Position(mm), Raw_Force(N), Corrected_Force(N), Direction
    Direction is 'down' for indentation and 'up' for return.
    """
    try:
        pos = cnc.get_current_position()
        if not pos:
            print("❌ Could not get current position from CNC")
            return False
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return False

        baseline_avg, baseline_std = force_sensor.get_baseline_force(samples=10)
        print(f"📊 Baseline: {baseline_avg:.3f} ± {baseline_std:.3f} N")

        if filename is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            if well is not None:
                if run_folder is None:
                    run_count = get_and_increment_run_count()
                    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"well_{well}_{ts}.csv")
            else:
                run_count = get_and_increment_run_count()
                run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"indentation_{ts}.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # Move to well top position first (optionally lock XY)
        if well is not None:
            if locked_xy is not None:
                try:
                    # lock_xy: override well XY with fixed coordinates
                    # 1) raise to safety Z, 2) move XY at safety, 3) go down to well_top_z
                    print(f"📍 Locked-XY mode: moving to safety Z, then X={locked_xy[0]:.3f}, Y={locked_xy[1]:.3f}, then Z={well_top_z:.1f}mm for well {well}...")
                    cnc.move_to_safe_z()
                    cnc.move_to_x_y(locked_xy[0], locked_xy[1])
                    cnc.move_to_z(well_top_z, wait_for_idle=True)
                    print(f"✅ Positioned at locked XY for well {well}")
                except Exception as e:
                    print(f"⚠️ Could not move to locked XY for well {well}: {e}")
                    return False
            else:
                col = ''.join([c for c in well if c.isalpha()]).upper()
                row = ''.join([c for c in well if c.isdigit()])
                if col and row:
                    try:
                        print(f"📍 Moving to well {well} at top position Z={well_top_z:.1f}mm...")
                        cnc.move_to_well(col, row, z=well_top_z)
                        print(f"✅ Positioned at well {well} top")
                    except Exception as e:
                        print(f"⚠️ Could not move to well {well}: {e}")
                        return False
        else:
            # For current position measurements, move to well_top_z
            try:
                print(f"📍 Moving to well top position Z={well_top_z:.1f}mm...")
                cnc.move_to_z(well_top_z)
                print(f"✅ Positioned at well top")
            except Exception as e:
                print(f"⚠️ Could not move to well top position: {e}")
                return False

        measurements: list[list[object]] = []

        # Downward
        while True:
            current = cnc.get_current_position()
            if not current:
                print("❌ Could not get position - stopping measurement")
                break
            current_z = float(current[2])
            if current_z <= z_target:
                print(f"🎯 Reached z_target {z_target:.3f}mm")
                break
            next_z = current_z - step_size
            cnc.move_to_z(next_z, wait_for_idle=True)
            current = cnc.get_current_position() or (None, None, next_z)
            force = force_sensor.get_force_reading()
            corrected = force - baseline_avg
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected, 'down'])
            if len(measurements) % 10 == 0:
                try:
                    print(f"📉 Down #{len(measurements)}: Z={float(current[2]):.3f}mm, F={force:.3f}N, dF={corrected:.3f}N")
                except Exception:
                    pass
            if abs(corrected) > force_limit:
                print(f"🛑 Force limit exceeded: {corrected:.3f}N > {force_limit:.1f}N")
                break

        # Upward return
        while True:
            current = cnc.get_current_position()
            if not current:
                break
            current_z = float(current[2])
            if current_z >= well_top_z:
                break
            next_z = min(current_z + step_size, well_top_z)
            cnc.move_to_z(next_z, wait_for_idle=True)
            current = cnc.get_current_position() or (None, None, next_z)
            force = force_sensor.get_force_reading()
            corrected = force - baseline_avg
            t = time.time()
            measurements.append([t, float(current[2]), force, corrected, 'up'])
            if len(measurements) % 10 == 0:
                try:
                    print(f"📈 Up #{len(measurements)}: Z={float(current[2]):.3f}mm, F={force:.3f}N, dF={corrected:.3f}N")
                except Exception:
                    pass

        # Ensure safety height before moving to next well
        cnc.move_to_safe_z()

        # Write CSV with Direction column
        with open(filename, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                w.writerow(['Well', well])
            w.writerow(['Target_Z(mm)', f"{z_target:.3f}"])
            w.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            w.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            w.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            w.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            w.writerow(['Force_Exceeded', str(any(abs(m[3]) > force_limit for m in measurements))])
            w.writerow([])
            w.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)', 'Direction'])
            for t, z, rf, cf, d in measurements:
                w.writerow([f"{t:.3f}", f"{z:.3f}", f"{rf:.3f}", f"{cf:.3f}", d])
        print(f"💾 Saved {len(measurements)} points (down+up) to {filename}")
        return True
    except Exception as e:
        print(f"❌ Error in simple_indentation_with_return_measurement: {e}")
        return False

def test_step_force_measurement(cnc, force_sensor, target_z=-15.0, step_size=0.2, force_limit=45.0, filename=None, well=None, poisson_ratio=0.33, run_folder=None):
    """Test force measurement for step movement with force limit and baseline correction"""
    try:
        print("🔧 Checking CNC connection...")
        current_pos = cnc.get_current_position()
        if not current_pos:
            print("❌ Could not get current position from CNC")
            print("💡 Check if CNC is connected and powered on")
            return False
        print("🔧 Checking force sensor connection...")
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return False
        print("📊 Taking baseline force measurement...")
        baseline_avg, baseline_std = force_sensor.get_baseline_force(samples=5)
        print(f"📊 Baseline force: {baseline_avg:.3f} ± {baseline_std:.3f}N")
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if well is not None:
                if run_folder is None:
                    # Create new run folder if not provided
                    run_count = get_and_increment_run_count()
                    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")
            else:
                run_count = get_and_increment_run_count()
                run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"force_measurements_{timestamp}.csv")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        data_count = 0
        measurements = []
        z_start = None
        # Step movement loop
        while abs(float(current_pos[2])) < abs(target_z):
            next_z = float(current_pos[2]) - step_size # move down
            print(f"🔄 Moving to Z={next_z:.3f}mm...")
            cnc.move_to_z(next_z)
            current_pos = cnc.get_current_position()
            if not current_pos:
                print("❌ Could not get position - stopping test")
                return False
            force = force_sensor.get_force_reading()
            data_count += 1
            if z_start is None:
                z_start = float(current_pos[2])
            corrected_force = force - baseline_avg
            timestamp = time.time()
            measurements.append([timestamp, float(current_pos[2]), force, corrected_force])
            print(f"✅ Z: {current_pos[2]}mm, Force: {force:.3f}N, Corrected: {corrected_force:.3f}N (Data point #{data_count})")
            if abs(corrected_force) > force_limit:
                print(f"🛑 Force limit exceeded! Corrected force: {corrected_force:.3f}N > {force_limit:.1f}N")
                print(f"🛑 Stopping at Z: {current_pos[2]}mm")
                break
        print("🔄 Returning to safety height...")
        cnc.move_to_safe_z()
        print(f"💾 Saving {len(measurements)} measurements to {filename}")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                writer.writerow(['Well', well])
            writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
            writer.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            writer.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            writer.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            writer.writerow([])  # Empty row for separation
            writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)'])
            for timestamp, z_pos, raw_force, corrected_force in measurements:
                writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{raw_force:.3f}", f"{corrected_force:.3f}"])
        if abs(measurements[-1][3]) > force_limit:
            print(f"⚠️ Test stopped due to force limit! Collected {data_count} data points")
            print(f"⚠️ Final position: Z={current_pos[2]}mm, Final corrected force: {measurements[-1][3]:.3f}N")
        else:
            print(f"✅ Test completed successfully! Collected {data_count} data points")
            print(f"✅ Reached target Z: {target_z:.3f}mm")
        return True
    except Exception as e:
        print(f"❌ Error during step force measurement: {e}")
        return False


def dynamic_indentation_measurement(cnc, force_sensor, well=None, z_target=-15.0, step_size=0.1, force_limit=45.0, 
                                   contact_force_threshold=2.0, filename=None, run_folder=None):
    """
    Dynamic indentation measurement that stops based on two conditions:
    1. Reaches force limit
    2. Reaches z_target
    
    Args:
        cnc: CNCController object
        force_sensor: ForceSensor object
        well: Well identifier (e.g., "A1")
        z_target: Target Z position in mm
        step_size: Step size for movement in mm
        force_limit: Force limit in N
        contact_force_threshold: Force threshold to detect contact in N (default: 2.0)
        filename: Output filename (auto-generated if None)
        run_folder: Run folder for saving data
        
    Returns:
        bool: True if measurement completed successfully
    """
    try:
        print(f"🔧 Starting dynamic indentation measurement...")
        
        # Check connections
        print("🔧 Checking CNC connection...")
        current_pos = cnc.get_current_position()
        if not current_pos:
            print("❌ Could not get current position from CNC")
            return False
            
        print("🔧 Checking force sensor connection...")
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return False
        
        # Take baseline force measurement
        print("📊 Taking baseline force measurement...")
        baseline_avg, baseline_std = force_sensor.get_baseline_force(samples=5)
        print(f"📊 Baseline force: {baseline_avg:.3f} ± {baseline_std:.3f}N")
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if well is not None:
                if run_folder is None:
                    run_count = get_and_increment_run_count()
                    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                    run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")
            else:
                run_count = get_and_increment_run_count()
                run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
                run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
                os.makedirs(run_folder, exist_ok=True)
                filename = os.path.join(run_folder, f"dynamic_indentation_{timestamp}.csv")
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Initialize variables
        measurements = []
        data_count = 0
        contact_detected = False
        contact_z = None
        indentation_depth = 0.0
        z_start = float(current_pos[2]) if current_pos[2] is not None else 0.0
        
        print(f"🎯 Starting indentation from Z={z_start:.3f}mm")
        print(f"🎯 Target Z={z_target:.3f}mm, Force limit={force_limit:.1f}N")
        print(f"🎯 Contact threshold={contact_force_threshold:.1f}N")
        
        # Dynamic indentation loop
        while True:
            # Calculate next Z position
            current_z = float(current_pos[2]) if current_pos[2] is not None else z_start
            next_z = current_z - step_size
            
            # Check if we've reached the target Z
            if next_z <= z_target:
                print(f"🎯 Reached target Z={z_target:.3f}mm")
                break
            
            # Move to next position
            print(f"🔄 Moving to Z={next_z:.3f}mm...")
            cnc.move_to_z(next_z, wait_for_idle=False)
            time.sleep(0.01)  # Brief delay for CNC to process
            
            # Get current position and force
            current_pos = cnc.get_current_position()
            if not current_pos:
                print("❌ Could not get position - stopping measurement")
                return False
            
            force = force_sensor.get_force_reading()
            corrected_force = force - baseline_avg
            data_count += 1
            
            # Record measurement
            timestamp = time.time()
            measurements.append([timestamp, float(current_pos[2]), force, corrected_force])
            
            print(f"✅ Z: {current_pos[2]:.3f}mm, Force: {force:.3f}N, Corrected: {corrected_force:.3f}N (Data #{data_count})")
            
            # Check for contact detection
            if not contact_detected and abs(corrected_force) > contact_force_threshold:
                contact_detected = True
                contact_z = float(current_pos[2])
                print(f"🔍 Contact detected at Z={contact_z:.3f}mm (force: {corrected_force:.3f}N)")
            
            # No target force check - continue until force limit or z_target is reached
            
            # Check force limit
            if abs(corrected_force) > force_limit:
                current_z_pos = float(current_pos[2]) if current_pos[2] is not None else z_start
                print(f"🛑 Force limit exceeded! Corrected force: {corrected_force:.3f}N > {force_limit:.1f}N")
                print(f"🛑 Stopping at Z: {current_z_pos:.3f}mm")
                break
        
        # Return to safety height
        print("🔄 Returning to safety height...")
        cnc.move_to_safe_z()
        
        # Save data
        print(f"💾 Saving {len(measurements)} measurements to {filename}")
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
            if well is not None:
                writer.writerow(['Well', well])
            writer.writerow(['Target_Z(mm)', f"{z_target:.3f}"])
            writer.writerow(['Step_Size(mm)', f"{step_size:.3f}"])
            writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
            writer.writerow(['Contact_Force_Threshold(N)', f"{contact_force_threshold:.3f}"])
            writer.writerow(['Baseline_Force(N)', f"{baseline_avg:.3f}"])
            writer.writerow(['Baseline_Std(N)', f"{baseline_std:.3f}"])
            writer.writerow(['Contact_Detected', str(contact_detected)])
            if contact_detected:
                writer.writerow(['Contact_Z(mm)', f"{contact_z:.3f}"])
            writer.writerow(['Force_Limit_Exceeded', str(abs(measurements[-1][3]) > force_limit)])
            current_z_pos = float(current_pos[2]) if current_pos[2] is not None else z_start
            writer.writerow(['Target_Z_Reached', str(current_z_pos <= z_target)])
            writer.writerow([])  # Empty row for separation
            writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Raw_Force(N)', 'Corrected_Force(N)'])
            for timestamp, z_pos, raw_force, corrected_force in measurements:
                writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{raw_force:.3f}", f"{corrected_force:.3f}"])
        
        # Print summary
        final_force = measurements[-1][3]
        final_z = measurements[-1][1]
        
        print(f"\n📊 Measurement Summary:")
        print(f"   Data points collected: {data_count}")
        print(f"   Final Z position: {final_z:.3f}mm")
        print(f"   Final corrected force: {final_force:.3f}N")
        
        if contact_detected:
            print(f"   Contact detected at: {contact_z:.3f}mm")
        
        # Determine stopping reason
        if abs(final_force) > force_limit:
            print(f"   ⚠️ Stopped due to force limit")
        elif final_z <= z_target:
            print(f"   ✅ Stopped due to reaching target Z")
        else:
            print(f"   ✅ Stopped due to reaching target Z")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during dynamic indentation measurement: {e}")
    return False