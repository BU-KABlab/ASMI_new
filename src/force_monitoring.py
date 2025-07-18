#!/usr/bin/env python3
"""
Force monitoring module for ASMI system
Provides functionality to test force monitoring at multiple wells with force limits
"""

import time
import csv
import os
from datetime import datetime
from .CNCController import CNCController
from .ForceSensor import ForceSensor
from .analysis import IndentationAnalyzer


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
            cnc.move_to_z(next_z, wait_for_idle=False)
            time.sleep(0.01)  # 10ms delay for CNC to process
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
        print("🔄 Returning to Z=0...")
        cnc.move_to_z(0)
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
        print("🔄 Returning to Z=0...")
        cnc.move_to_z(0)
        
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


# def test_well_force_monitoring(cnc, force_sensor, well="A1", target_z=-10.0, period_ms=50, filename=None, feedrate=1000):
#     """
#     Test force monitoring at a specific well location.
    
#     Args:
#         cnc: CNCController object
#         force_sensor: ForceSensor object
#         well: Well identifier (e.g., "A1")
#         target_z: Target Z position in mm
#         period_ms: Sampling period in milliseconds
#         filename: Output filename (auto-generated if None)
#         feedrate: Movement feedrate
        
#     Returns:
#         List of measurements [(timestamp, z_pos, force), ...]
#     """
#     print(f"🧪 Testing force monitoring at well {well}")
#     print(f"📊 Moving to well {well} and monitoring force at Z={target_z:.3f}mm")
    
#     # Generate filename if not provided
#     if filename is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         filename = f"results/force_measurements/well_{well}_{timestamp}.csv"
    
#     # Ensure results directory exists
#     os.makedirs(os.path.dirname(filename), exist_ok=True)
    
#     # Move to well at safety height first
#     print(f"📍 Moving to well {well}...")
#     col, row = well[0], well[1:]
#     cnc.move_to_well(col, row, z=0)
    
#     # Get current position
#     current_pos = cnc.get_current_position()
#     if not current_pos:
#         print("❌ Could not get current position")
#         return []
    
#     print(f"✅ Arrived at well {well}: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
    
#     # Move to target Z with force monitoring
#     print(f"📊 Moving to Z={target_z:.3f} with force monitoring...")
#     measurements = cnc.move_to_z_with_force_monitoring(
#         target_z, force_sensor, period_ms, filename=None, feedrate=feedrate
#     )
    
#     # Get test timestamp
#     test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
#     # Save data with metadata
#     with open(filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Test_Time', test_timestamp])
#         writer.writerow(['Well', well])
#         writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
#         writer.writerow(['Sampling_Period(ms)', f"{period_ms}"])
#         writer.writerow(['Well_X(mm)', f"{current_pos[0]:.3f}"])
#         writer.writerow(['Well_Y(mm)', f"{current_pos[1]:.3f}"])
#         writer.writerow([])  # Empty row for separation
#         writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)'])
        
#         for timestamp, z_pos, force_val in measurements:
#             writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}"])
    
#     print(f"💾 Saved {len(measurements)} measurements to {filename}")
#     print(f"✅ Test completed at well {well}")
    
#     return measurements


# def test_well_loop_with_force_limit(cnc, force_sensor, wells=None, target_z=-15.0, force_limit=45.0, period_ms=20, feedrate=1000):
#     """
#     Loop through wells and stop when force limit is exceeded.
    
#     Args:
#         cnc: CNCController object
#         force_sensor: ForceSensor object
#         wells: List of wells to test (e.g., ["A1", "A2", "B1", "B2"])
#         target_z: Target Z position for each well
#         force_limit: Force limit in N (absolute value)
#         period_ms: Sampling period in milliseconds
#         feedrate: Movement feedrate
        
#     Returns:
#         List of result dictionaries with test outcomes
#     """
#     if wells is None:
#         wells = ["A1"]  # Default wells
    
#     print(f"🔄 Starting well loop test with force limit {force_limit}N")
#     print(f"📋 Wells to test: {wells}")
#     print(f"🎯 Target Z: {target_z:.3f}mm")
    
#     run_count = get_and_increment_run_count()
#     run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
#     os.makedirs(run_folder, exist_ok=True)
#     summary_filename = os.path.join(run_folder, f"well_summary_{run_count:03d}_{run_date}.csv")
    
#     results = []
    
#     for well in wells:
#         print(f"\n{'='*50}")
#         print(f"🧪 Testing well {well}")
#         print(f"{'='*50}")
        
#         try:
#             # Move to well at safety height
#             col, row = well[0], well[1:]
#             print(f"📍 Moving to well {well}...")
#             cnc.move_to_well(col, row, z=0)
            
#             # Get current position
#             current_pos = cnc.get_current_position()
#             if not current_pos:
#                 print(f"❌ Could not get position for well {well}")
#                 continue
            
#             print(f"✅ Arrived at well {well}: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
            
#             # Start force monitoring and move to target Z
#             print(f"📊 Moving to Z={target_z:.3f} with force monitoring...")
            
#             # Initialize data collection
#             measurements = []
#             start_time = time.time()
            
#             # Check force sensor connection
#             if not force_sensor.is_connected():
#                 print("❌ Force sensor not connected")
#                 continue
            
#             # Start continuous force monitoring
#             if not force_sensor.start_continuous_monitoring(period_ms):
#                 print("❌ Could not start continuous force monitoring")
#                 continue
            
#             try:
#                 # Send movement command
#                 gcode = f"G01 Z{target_z:.3f} F{feedrate}"
#                 print(f"> Sending: {gcode}")
#                 cnc.ser.write((gcode + '\n').encode())
                
#                 # Monitor force and position during downward movement
#                 force_exceeded = False
#                 stop_z = None
#                 z_start = current_pos[2]
#                 sampling_interval = period_ms / 1000.0
#                 last_sample_time = start_time
                
#                 stuck_z = None
#                 stuck_time = None
#                 stuck_threshold = 1.0  # seconds
                
#                 while True:
#                     current_time = time.time()
                    
#                     # Only sample at the specified period
#                     if current_time - last_sample_time >= sampling_interval:
#                         # Get force reading
#                         force = force_sensor.get_continuous_reading()
                        
#                         # Get position
#                         pos = cnc.get_current_position()
#                         if pos:
#                             z_current = pos[2]
#                         else:
#                             z_current = z_start
                        
#                         # Stuck Z detection
#                         if stuck_z is None or abs(z_current - stuck_z) > 1e-4:
#                             stuck_z = z_current
#                             stuck_time = current_time
#                         elif stuck_time is not None and (current_time - stuck_time > stuck_threshold):
#                             print(f"⚠️ Z position stuck at {z_current:.3f} for >{stuck_threshold}s. Moving up to Z=0.")
#                             break
                        
#                         # Record measurement
#                         timestamp = current_time - start_time
#                         measurements.append((timestamp, z_current, force))
                        
#                         # Check force limit (both positive and negative)
#                         if abs(force) > force_limit:
#                             print(f"⚠️ Force limit exceeded: {force:.3f}N > {force_limit}N at Z={z_current:.3f}mm")
#                             force_exceeded = True
#                             stop_z = z_current
                            
#                             # Emergency stop the movement
#                             print(f"🛑 Emergency stop - sending feed hold")
#                             cnc.ser.write(b'!')
#                             time.sleep(0.2)
                            
#                             # Check machine status
#                             cnc.ser.write(b'?\n')
#                             time.sleep(0.1)
#                             status = cnc.ser.readline().decode('utf-8').strip()
#                             print(f"📡 Machine status: {status}")
                            
#                             # If machine is in hold state, unlock it
#                             if "Hold" in status:
#                                 print(f"🔓 Unlocking machine...")
#                                 cnc.ser.write(b'$X\n')
#                                 time.sleep(0.3)
                                
#                                 # Verify unlock worked
#                                 cnc.ser.write(b'?\n')
#                                 time.sleep(0.1)
#                                 status_after_unlock = cnc.ser.readline().decode('utf-8').strip()
#                                 print(f"📡 Status after unlock: {status_after_unlock}")
                                
#                                 # If still in hold, try cycle start
#                                 if "Hold" in status_after_unlock:
#                                     print(f"🔄 Trying cycle start...")
#                                     cnc.ser.write(b'~\n')
#                                     time.sleep(0.2)
                            
#                             break
                        
#                         # Check if target reached
#                         if abs(z_current - target_z) < 0.1:
#                             print(f"🎯 Target Z={target_z:.3f} reached without exceeding force limit")
#                             break
                        
#                         last_sample_time = current_time
                    
#                     # Small sleep to prevent busy waiting
#                     time.sleep(0.001)
                
#                 # Now move back to Z=0 with force monitoring
#                 print(f"📈 Moving back to Z=0 with force monitoring...")
                
#                 # Ensure machine is not in hold state before moving
#                 cnc.ser.write(b'?\n')
#                 time.sleep(0.1)
#                 status = cnc.ser.readline().decode('utf-8').strip()
#                 print(f"📡 Status before return movement: {status}")
                
#                 if "Hold" in status:
#                     print(f"🔓 Unlocking machine before return movement...")
#                     cnc.ser.write(b'$X\n')
#                     time.sleep(0.3)
                    
#                     # Verify unlock worked
#                     cnc.ser.write(b'?\n')
#                     time.sleep(0.1)
#                     status_after_unlock = cnc.ser.readline().decode('utf-8').strip()
#                     print(f"📡 Status after unlock: {status_after_unlock}")
                    
#                     # If still in hold, try cycle start
#                     if "Hold" in status_after_unlock:
#                         print(f"🔄 Trying cycle start before return...")
#                         cnc.ser.write(b'~\n')
#                         time.sleep(0.3)
                
#                 # Send return movement command
#                 gcode_up = f"G01 Z0.0 F{feedrate}"
#                 print(f"> Sending: {gcode_up}")
#                 cnc.ser.write((gcode_up + '\n').encode())
                
#                 # Monitor force during upward movement
#                 while True:
#                     current_time = time.time()
                    
#                     # Only sample at the specified period
#                     if current_time - last_sample_time >= sampling_interval:
#                         # Get force reading
#                         force = force_sensor.get_continuous_reading()
                        
#                         # Get position
#                         pos = cnc.get_current_position()
#                         if pos:
#                             z_current = pos[2]
#                         else:
#                             z_current = z_start
                        
#                         # Record measurement
#                         timestamp = current_time - start_time
#                         measurements.append((timestamp, z_current, force))
                        
#                         # Check if back to Z=0
#                         if abs(z_current) < 0.1:
#                             print(f"🎯 Returned to Z=0 successfully")
#                             break
                        
#                         last_sample_time = current_time
                    
#                     # Small sleep to prevent busy waiting
#                     time.sleep(0.001)
            
#             finally:
#                 # Stop continuous monitoring
#                 force_sensor.stop_continuous_monitoring()
            
#             # Save well data
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             well_filename = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")
#             test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
#             with open(well_filename, 'w', newline='') as f:
#                 writer = csv.writer(f)
#                 writer.writerow(['Test_Time', test_timestamp])
#                 writer.writerow(['Well', well])
#                 writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
#                 writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
#                 writer.writerow(['Force_Exceeded', str(force_exceeded)])
#                 writer.writerow(['Stop_Z(mm)', f"{stop_z:.3f}" if stop_z else "N/A"])
#                 writer.writerow(['Sampling_Period(ms)', f"{period_ms}"])
#                 writer.writerow(['Well_X(mm)', f"{current_pos[0]:.3f}"])
#                 writer.writerow(['Well_Y(mm)', f"{current_pos[1]:.3f}"])
#                 writer.writerow([])
#                 writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)', 'Movement'])
                
#                 # Determine movement direction for each measurement
#                 for i, (timestamp, z_pos, force_val) in enumerate(measurements):
#                     if i == 0:
#                         movement = "Down"
#                     elif z_pos > measurements[i-1][1]:  # Z is increasing
#                         movement = "Up"
#                     else:
#                         movement = "Down"
                    
#                     writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}", movement])
            
#             # Record result
#             max_force = max(abs(m[2]) for m in measurements) if measurements else 0
#             results.append({
#                 'well': well,
#                 'force_exceeded': force_exceeded,
#                 'stop_z': stop_z,
#                 'max_force': max_force,
#                 'samples': len(measurements),
#                 'filename': well_filename
#             })
            
#             print(f"💾 Saved {len(measurements)} measurements to {well_filename}")
#             print(f"📊 Max force: {max_force:.3f}N")
            
#             # If force limit exceeded, note it but continue to next well
#             if force_exceeded:
#                 print(f"⚠️ Force limit exceeded at well {well}. Target Z adjusted to {stop_z:.3f}mm.")
#             else:
#                 print(f"✅ Well {well} completed without force limit exceeded.")
                
#         except Exception as e:
#             print(f"❌ Error testing well {well}: {e}")
#             results.append({
#                 'well': well,
#                 'force_exceeded': False,
#                 'stop_z': None,
#                 'max_force': 0,
#                 'samples': 0,
#                 'filename': f"ERROR: {e}"
#             })
    
#     # Save summary
#     with open(summary_filename, 'w', newline='') as f:
#         writer = csv.writer(f)
#         writer.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
#         writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
#         writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
#         writer.writerow([])
#         writer.writerow(['Well', 'Force_Exceeded', 'Stop_Z(mm)', 'Max_Force(N)', 'Samples', 'Filename'])
        
#         for result in results:
#             writer.writerow([
#                 result['well'],
#                 result['force_exceeded'],
#                 f"{result['stop_z']:.3f}" if result['stop_z'] else "N/A",
#                 f"{result['max_force']:.3f}",
#                 result['samples'],
#                 result['filename']
#             ])
    
#     print(f"\n{'='*50}")
#     print(f"📋 Well Loop Summary")
#     print(f"{'='*50}")
#     print(f"💾 Summary saved to: {summary_filename}")
#     print(f"📊 Wells tested: {len(results)}")
    
#     # Count force limit events
#     force_limit_count = sum(1 for r in results if r['force_exceeded'])
#     print(f"⚠️ Force limit exceeded at {force_limit_count} wells")
    
#     return results


# def run_force_monitoring_test(wells_to_test, target_z=-15.0, force_limit=45.0, period_ms=20, feedrate=1000):
#     """
#     Main function to run force monitoring test on specified wells.
    
#     Args:
#         wells_to_test: List of wells to test (e.g., ["A1", "A2", "B1", "B2"])
#         target_z: Target Z position for each well (default: -15.0 mm)
#         force_limit: Force limit in N (default: 45.0 N)
#         period_ms: Sampling period in milliseconds (default: 20 ms)
#         feedrate: Movement feedrate (default: 1000)
        
#     Returns:
#         List of result dictionaries with test outcomes
#     """
#     print("🧪 Starting Force Monitoring Test")
#     print("=" * 50)
    
#     try:
#         # Initialize components
#         print("🔧 Initializing components...")
#         force_sensor = ForceSensor()
#         cnc = CNCController()
        
#         if not force_sensor.is_connected():
#             print("❌ Force sensor not connected. Please check USB connection.")
#             return []
        
#         print("✅ Components initialized successfully")
        
#         # Home the machine
#         print("🏠 Homing machine...")
#         cnc.home()
        
#         # Run the well loop test
#         results = test_well_loop_with_force_limit(
#             cnc, force_sensor,
#             wells=wells_to_test,
#             target_z=target_z,
#             force_limit=force_limit,
#             period_ms=period_ms,
#             feedrate=feedrate
#         )
        
#         # Return to home position
#         print("🏠 Returning to home position...")
#         cnc.home()
        
#         print("\n🎉 Force monitoring test completed successfully!")
#         return results
        
#     except Exception as e:
#         print(f"❌ Error during testing: {e}")
#         return []
    
#     finally:
#         # Cleanup
#         print("\n🧹 Cleaning up...")
#         if 'force_sensor' in locals():
#             force_sensor.cleanup()
#         if 'cnc' in locals():
#             cnc.close()
#         print("✅ Cleanup complete")


# if __name__ == "__main__":
#     # Example usage
#     wells_to_test = ["A6", "B6", "C6", "C5", "B5", "A5"]
#     results = run_force_monitoring_test(
#         wells_to_test=wells_to_test,
#         target_z=-15.0,
#         force_limit=45.0,
#         period_ms=10,
#         feedrate=200
#     )
    
#     print(f"\n📊 Test completed with {len(results)} wells tested") 