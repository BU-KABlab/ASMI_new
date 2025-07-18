#!/usr/bin/env python3
"""
Test script for force monitoring functionality
Run this to test force vs Z position monitoring
"""

from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
import time
import csv
import os
from datetime import datetime

def get_and_increment_run_count(run_count_file='src/run_count.txt'):
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

def test_well_force_monitoring(cnc, force_sensor, well="A1", target_z=-10.0, period_ms=50, filename=None, feedrate=1000):
    """
    Test force monitoring at a specific well location.
    """
    print(f"🧪 Testing force monitoring at well {well}")
    print(f"📊 Moving to well {well} and monitoring force at Z={target_z:.3f}mm")
    
    # Generate filename if not provided (default is results/force_measurements/well_A1_timestamp.csv)
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/force_measurements/well_{well}_{timestamp}.csv"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Move to well A1 at safety height first
    print(f"📍 Moving to well {well}...")
    cnc.move_to_well("A", "1", z=0)  # Move to A1 at safety height
    
    # Get current position
    current_pos = cnc.get_current_position()
    if not current_pos:
        print("❌ Could not get current position")
        return []
    
    print(f"✅ Arrived at well {well}: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
    
    # Move to target Z with force monitoring
    print(f"📊 Moving to Z={target_z:.3f} with force monitoring...")
    measurements = cnc.move_to_z_with_force_monitoring(
        target_z, force_sensor, period_ms, filename=None, feedrate=feedrate # Don't save yet
    )
    
    # Get test timestamp
    test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save data with metadata
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_Time', test_timestamp])
        writer.writerow(['Well', well])
        writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
        writer.writerow(['Sampling_Period(ms)', f"{period_ms}"])
        writer.writerow(['Well_X(mm)', f"{current_pos[0]:.3f}"])
        writer.writerow(['Well_Y(mm)', f"{current_pos[1]:.3f}"])
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)'])
        
        for timestamp, z_pos, force_val in measurements:
            writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}"])
    
    print(f"💾 Saved {len(measurements)} measurements to {filename}")
    print(f"✅ Test completed at well {well}")
    
    return measurements


def test_well_loop_with_force_limit(cnc, force_sensor, wells=None, target_z=-15.0, force_limit=45.0, period_ms=20, feedrate=1000):
    """
    Loop through wells and stop when force limit is exceeded.
    
    Args:
        cnc: CNCController object
        force_sensor: ForceSensor object
        wells: List of wells to test (e.g., ["A1", "A2", "B1", "B2"])
        target_z: Target Z position for each well
        force_limit: Force limit in N (absolute value)
        period_ms: Sampling period in milliseconds
        feedrate: Movement feedrate
    """
    if wells is None:
        wells = ["A1"]  # Default wells
    
    print(f"🔄 Starting well loop test with force limit {force_limit}N")
    print(f"📋 Wells to test: {wells}")
    print(f"🎯 Target Z: {target_z:.3f}mm")
    
    run_count = get_and_increment_run_count()
    run_date = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_folder = f"results/measurements/run_{run_count:03d}_{run_date}"
    os.makedirs(run_folder, exist_ok=True)
    summary_filename = os.path.join(run_folder, f"well_summary_{run_count:03d}_{run_date}.csv")
    
    results = []
    
    for well in wells:
        print(f"\n{'='*50}")
        print(f"🧪 Testing well {well}")
        print(f"{'='*50}")
        
        try:
            # Move to well at safety height
            col, row = well[0], well[1:]
            print(f"📍 Moving to well {well}...")
            cnc.move_to_well(col, row, z=0)
            
            # Get current position
            current_pos = cnc.get_current_position()
            if not current_pos:
                print(f"❌ Could not get position for well {well}")
                continue
            
            print(f"✅ Arrived at well {well}: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
            
            # Start force monitoring and move to target Z
            print(f"📊 Moving to Z={target_z:.3f} with force monitoring...")
            
            # Initialize data collection
            measurements = []
            start_time = time.time()
            
            # Check force sensor connection
            if not force_sensor.is_connected():
                print("❌ Force sensor not connected")
                continue
            
            # Start continuous force monitoring
            if not force_sensor.start_continuous_monitoring(period_ms):
                print("❌ Could not start continuous force monitoring")
                continue
            
            try:
                # Send movement command
                gcode = f"G01 Z{target_z:.3f} F{feedrate}"
                print(f"> Sending: {gcode}")
                cnc.ser.write((gcode + '\n').encode())
                
                # Monitor force and position during downward movement with native timing
                force_exceeded = False
                stop_z = None
                z_start = current_pos[2]
                sampling_interval = period_ms / 1000.0  # Convert to seconds
                last_sample_time = start_time
                
                stuck_z = None
                stuck_time = None
                stuck_threshold = 1.0  # seconds
                
                while True:
                    current_time = time.time()
                    
                    # Only sample at the specified period
                    if current_time - last_sample_time >= sampling_interval:
                        # Get force reading
                        force = force_sensor.get_continuous_reading()
                        
                        # Get position
                        pos = cnc.get_current_position()
                        if pos:
                            z_current = pos[2]
                        else:
                            z_current = z_start
                        
                        # Stuck Z detection
                        if stuck_z is None or abs(z_current - stuck_z) > 1e-4:
                            stuck_z = z_current
                            stuck_time = current_time
                        elif stuck_time is not None and (current_time - stuck_time > stuck_threshold):
                            print(f"⚠️ Z position stuck at {z_current:.3f} for >{stuck_threshold}s. Moving up to Z=0.")
                            break  # Exit loop to trigger return to Z=0
                        
                        # Record measurement
                        timestamp = current_time - start_time
                        measurements.append((timestamp, z_current, force))
                        
                        # Check force limit (both positive and negative)
                        if abs(force) > force_limit:
                            print(f"⚠️ Force limit exceeded: {force:.3f}N > {force_limit}N at Z={z_current:.3f}mm")
                            force_exceeded = True
                            stop_z = z_current
                            
                            # Emergency stop the movement - IMPROVED SEQUENCE
                            print(f"🛑 Emergency stop - sending feed hold")
                            cnc.ser.write(b'!')  # Feed hold
                            time.sleep(0.2)  # Wait longer for hold to complete
                            
                            # Check machine status
                            cnc.ser.write(b'?\n')
                            time.sleep(0.1)
                            status = cnc.ser.readline().decode('utf-8').strip()
                            print(f"📡 Machine status: {status}")
                            
                            # If machine is in hold state, unlock it
                            if "Hold" in status:
                                print(f"🔓 Unlocking machine...")
                                cnc.ser.write(b'$X\n')  # Unlock
                                time.sleep(0.3)  # Wait longer for unlock to complete
                                
                                # Verify unlock worked
                                cnc.ser.write(b'?\n')
                                time.sleep(0.1)
                                status_after_unlock = cnc.ser.readline().decode('utf-8').strip()
                                print(f"📡 Status after unlock: {status_after_unlock}")
                                
                                # If still in hold, try cycle start
                                if "Hold" in status_after_unlock:
                                    print(f"🔄 Trying cycle start...")
                                    cnc.ser.write(b'~\n')
                                    time.sleep(0.2)
                            
                            break
                        
                        # Check if target reached
                        if abs(z_current - target_z) < 0.1:
                            print(f"🎯 Target Z={target_z:.3f} reached without exceeding force limit")
                            break
                        
                        last_sample_time = current_time
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
                
                # Now move back to Z=0 with force monitoring - IMPROVED
                print(f"📈 Moving back to Z=0 with force monitoring...")
                
                # Ensure machine is not in hold state before moving
                cnc.ser.write(b'?\n')
                time.sleep(0.1)
                status = cnc.ser.readline().decode('utf-8').strip()
                print(f"📡 Status before return movement: {status}")
                
                if "Hold" in status:
                    print(f"🔓 Unlocking machine before return movement...")
                    cnc.ser.write(b'$X\n')
                    time.sleep(0.3)
                    
                    # Verify unlock worked
                    cnc.ser.write(b'?\n')
                    time.sleep(0.1)
                    status_after_unlock = cnc.ser.readline().decode('utf-8').strip()
                    print(f"📡 Status after unlock: {status_after_unlock}")
                    
                    # If still in hold, try cycle start
                    if "Hold" in status_after_unlock:
                        print(f"🔄 Trying cycle start before return...")
                        cnc.ser.write(b'~\n')
                        time.sleep(0.3)
                
                # Send return movement command
                gcode_up = f"G01 Z0.0 F{feedrate}"
                print(f"> Sending: {gcode_up}")
                cnc.ser.write((gcode_up + '\n').encode())
                
                # Monitor force during upward movement with native timing
                while True:
                    current_time = time.time()
                    
                    # Only sample at the specified period
                    if current_time - last_sample_time >= sampling_interval:
                        # Get force reading
                        force = force_sensor.get_continuous_reading()
                        
                        # Get position
                        pos = cnc.get_current_position()
                        if pos:
                            z_current = pos[2]
                        else:
                            z_current = z_start
                        
                        # Record measurement
                        timestamp = current_time - start_time
                        measurements.append((timestamp, z_current, force))
                        
                        # Check if back to Z=0
                        if abs(z_current) < 0.1:
                            print(f"🎯 Returned to Z=0 successfully")
                            break
                        
                        last_sample_time = current_time
                    
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
            
            finally:
                # Stop continuous monitoring
                force_sensor.stop_continuous_monitoring()
            
            # Save well data
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            well_filename = os.path.join(run_folder, f"well_{well}_{timestamp}.csv")
            test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            with open(well_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Test_Time', test_timestamp])
                writer.writerow(['Well', well])
                writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
                writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
                writer.writerow(['Force_Exceeded', str(force_exceeded)])
                writer.writerow(['Stop_Z(mm)', f"{stop_z:.3f}" if stop_z else "N/A"])
                writer.writerow(['Sampling_Period(ms)', f"{period_ms}"])
                writer.writerow(['Well_X(mm)', f"{current_pos[0]:.3f}"])
                writer.writerow(['Well_Y(mm)', f"{current_pos[1]:.3f}"])
                writer.writerow([])
                writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)', 'Movement'])
                
                # Determine movement direction for each measurement
                for i, (timestamp, z_pos, force_val) in enumerate(measurements):
                    if i == 0:
                        movement = "Down"
                    elif z_pos > measurements[i-1][1]:  # Z is increasing
                        movement = "Up"
                    else:
                        movement = "Down"
                    
                    writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}", movement])
            
            # Record result
            max_force = max(abs(m[2]) for m in measurements) if measurements else 0
            results.append({
                'well': well,
                'force_exceeded': force_exceeded,
                'stop_z': stop_z,
                'max_force': max_force,
                'samples': len(measurements),
                'filename': well_filename
            })
            
            print(f"💾 Saved {len(measurements)} measurements to {well_filename}")
            print(f"📊 Max force: {max_force:.3f}N")
            
            # If force limit exceeded, note it but continue to next well
            if force_exceeded:
                print(f"⚠️ Force limit exceeded at well {well}. Target Z adjusted to {stop_z:.3f}mm.")
            else:
                print(f"✅ Well {well} completed without force limit exceeded.")
                
        except Exception as e:
            print(f"❌ Error testing well {well}: {e}")
            results.append({
                'well': well,
                'force_exceeded': False,
                'stop_z': None,
                'max_force': 0,
                'samples': 0,
                'filename': f"ERROR: {e}"
            })
    
    # Save summary
    with open(summary_filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_Time', datetime.now().strftime("%Y-%m-%d %H:%M:%S")])
        writer.writerow(['Force_Limit(N)', f"{force_limit:.1f}"])
        writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
        writer.writerow([])
        writer.writerow(['Well', 'Force_Exceeded', 'Stop_Z(mm)', 'Max_Force(N)', 'Samples', 'Filename'])
        
        for result in results:
            writer.writerow([
                result['well'],
                result['force_exceeded'],
                f"{result['stop_z']:.3f}" if result['stop_z'] else "N/A",
                f"{result['max_force']:.3f}",
                result['samples'],
                result['filename']
            ])
    
    print(f"\n{'='*50}")
    print(f"📋 Well Loop Summary")
    print(f"{'='*50}")
    print(f"💾 Summary saved to: {summary_filename}")
    print(f"📊 Wells tested: {len(results)}")
    
    # Count force limit events
    force_limit_count = sum(1 for r in results if r['force_exceeded'])
    print(f"⚠️ Force limit exceeded at {force_limit_count} wells")
    
    return results


def test_force_cycle(cnc, force_sensor, target_z, period_ms=50, filename=None, verbose=True):
    """
    Simple function to test force monitoring during a complete down-up cycle.
    Keeps the CNCController class simple by handling the cycle logic here.
    """
    freq = 1000.0 / period_ms
    print(f"📊 Testing force cycle: Z=0 → Z={target_z:.3f} → Z=0 every {period_ms}ms ({freq:.1f} Hz)")
    
    # Generate filename if not provided
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/force_measurements/force_cycle_{timestamp}.csv"
    
    # Ensure results directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Get starting position
    start_pos = cnc.get_current_position()
    if not start_pos:
        print("❌ Could not get starting position")
        return []
    
    z_start = start_pos[2]
    print(f"📍 Starting from Z={z_start:.3f}")
    
    # Initialize data collection
    measurements = []
    start_time = time.time()
    
    # Check force sensor connection
    if not force_sensor.is_connected():
        print("❌ Force sensor not connected")
        return []
    
    # Phase 1: Move down with monitoring
    print(f"📉 Phase 1: Moving down to Z={target_z:.3f}")
    down_measurements = cnc.move_to_z_with_force_monitoring(
        target_z, force_sensor, period_ms, filename=None  # Don't save yet
    )
    measurements.extend(down_measurements)
    
    # Phase 2: Move back up with monitoring
    print(f"📈 Phase 2: Moving back up to Z={z_start:.3f}")
    up_measurements = cnc.move_to_z_with_force_monitoring(
        z_start, force_sensor, period_ms, filename=None  # Don't save yet
    )
    measurements.extend(up_measurements)
    
    # Get test timestamp
    test_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save complete cycle data
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Test_Time', test_timestamp])
        writer.writerow(['Target_Z(mm)', f"{target_z:.3f}"])
        writer.writerow(['Sampling_Period(ms)', f"{period_ms}"])
        writer.writerow([])  # Empty row for separation
        writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)', 'Phase'])
        
        # Track which phase we're in
        current_phase = "Moving_Down"
        down_measurements_count = len(down_measurements)
        
        for i, (timestamp, z_pos, force_val) in enumerate(measurements):
            # Determine phase based on measurement index and Z position
            if i < down_measurements_count:
                # First part: down movement
                if z_pos <= target_z + 0.1:  # At or near target
                    current_phase = "At_Target"
                elif z_pos >= z_start - 0.1:  # At or near original
                    current_phase = "At_Original"
                else:  # Moving down
                    current_phase = "Moving_Down"
            else:
                # Second part: up movement
                if z_pos >= z_start - 0.1:  # At or near original
                    current_phase = "At_Original"
                elif z_pos <= target_z + 0.1:  # At or near target
                    current_phase = "At_Target"
                else:  # Moving up
                    current_phase = "Moving_Up"
            
            writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}", current_phase])
    
    print(f"💾 Saved {len(measurements)} measurements to {filename}")
    print(f"✅ Complete cycle finished. Final Z={z_start:.3f}")
    
    return measurements


def test_force_monitoring(period_ms=50):
    """Test the force monitoring functionality"""
    print("🧪 Testing Force Monitoring Functionality")
    print("=" * 50)
    
    try:
        # Initialize components
        print("🔧 Initializing components...")
        force_sensor = ForceSensor()
        cnc = CNCController()
        
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected. Please check USB connection.")
            return False
        
        print("✅ Components initialized successfully")
        
    
        # Test: Well A1 force monitoring
        print("\n🔄 Test 4: Well A1 force monitoring")
        print("⚠️  Make sure there's nothing in the way!")
        
        # Ask for confirmation
        response = input("Continue with well A1 test? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Test cancelled")
            return False
        
        # Run well A1 force monitoring
        measurements = test_well_force_monitoring(
            cnc, force_sensor, well="A1", target_z=-10.0,
            period_ms=period_ms,
            filename="results/force_measurements/test_well_A1.csv"
        )
        
        if measurements:
            print(f"✅ Test completed! Recorded {len(measurements)} measurements")
            print("📁 Data saved to: results/force_measurements/test_force_cycle.csv")
            
            # Show summary
            forces = [m[2] for m in measurements]
            z_positions = [m[1] for m in measurements]
            print(f"📊 Force range: {min(forces):.3f} to {max(forces):.3f} N")
            print(f"📊 Z range: {min(z_positions):.3f} to {max(z_positions):.3f} mm")
        else:
            print("❌ No measurements recorded")
            return False
        
        print("\n🎉 All tests completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        return False
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'force_sensor' in locals():
            force_sensor.cleanup()
        if 'cnc' in locals():
            cnc.close()
        print("✅ Cleanup complete")


if __name__ == "__main__":
    force_sensor = ForceSensor()
    cnc = CNCController()
    cnc.home()
    
    # Test well loop with force limit
    wells_to_test = ["A6", "B6", "C6", "C5", "B5", "A5"]
    test_well_loop_with_force_limit(
        cnc, force_sensor, 
        wells=wells_to_test,
        target_z=-15.0, 
        force_limit=45.0, 
        period_ms=10, 
        feedrate=200
    )
    cnc.home()