"""
CNC Controller Module for ASMI

Handles CNC machine control, positioning, and G-code operations.

Author: [Your Name/Institution]
Date: 2024
License: MIT
"""

import serial
import time
import csv
import threading
import os
from datetime import datetime
from typing import List, Tuple, Optional

# === CNC CONFIGURATION ===
BAUD_RATE = 115200
GRBL_PORT = '/dev/cu.usbserial-130'

# === WELL PLATE GEOMETRY ===
A1_X = 99.0
A1_Y = 48.5
Z_INITIAL = 0 # safety height
WELL_SPACING = 9.0
ROWS = [str(i) for i in range(1, 13)]
COLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
FEEDRATE = 1000

class CNCController:
    def __init__(self, port=GRBL_PORT, baudrate=BAUD_RATE):
        self.port = port
        self.baudrate = baudrate
        self.ser = serial.Serial(self.port, self.baudrate)
        self.ser.write(b"\r\n\r\n")
        self.ser.reset_input_buffer()
        self.send_gcode("$X")  # Unlock
        print("🔧 CNC Controller initialized")
        # self.sync_position()


    def close(self):
        if self.ser.is_open:
            try:
                # Home the machine for safety
                self.home(zero_after=True)
                # Save current position before closing
                self.save_position()
            except Exception as e:
                print(f"⚠️ Error during homing: {e}")
            finally:
                self.ser.close()
                print("🔒 CNC serial port closed.")
        else:
            print("⚠️ CNC serial port already closed.")


    def wait_for_idle(self, timeout=10.0):
        """Wait for CNC to become idle with timeout"""
        idle_counter = 0
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                self.ser.reset_input_buffer()
                self.ser.write(b'?\n')
                status = self.ser.readline().decode('utf-8', errors='ignore').strip()
                
                if "Idle" in status:
                    idle_counter += 1
                    if idle_counter >= 3:  # Reduced from 10 to 3 for faster response
                        print("✅ CNC is idle")
                        return True
                elif "Alarm" in status or "Error" in status:
                    print(f"⚠️ CNC in error state: {status}")
                    return False
                elif "Run" in status:
                    idle_counter = 0  # Reset counter if still running
                    
                time.sleep(0.1)
                
            except Exception as e:
                print(f"⚠️ Error checking CNC status: {e}")
                time.sleep(0.1)
        
        print(f"⚠️ Timeout waiting for CNC to become idle after {timeout}s")
        return False


    def send_gcode(self, command, wait_for_response=True):
        if not self.ser or not self.ser.is_open:
            raise serial.SerialException("❌ Serial port is not open. Cannot send G-code.")
        print(f"> Sending: {command}")
        self.ser.write((command + '\n').encode())
        
        if not wait_for_response:
            return True
            
        if command.startswith("G92") or command.startswith("M") or command.startswith("$"):
            # Wait for immediate response for these commands
            while True:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"< Response: {response}")
                    break
        else:
            # For movement commands, wait for idle
            if not self.wait_for_idle():
                print("⚠️ CNC did not become idle - movement may have failed")
                return False
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            if response:
                print(f"< Response: {response}")
        return True


    def move_to_well(self, col: str, row: str, z: float = Z_INITIAL):
        # Always check current Z position and move to safety height first
        current_pos = self.get_current_position()
        if current_pos:
            current_z = current_pos[2]
            if current_z != Z_INITIAL:
                self.move_to_z(Z_INITIAL)
                time.sleep(1)
        
        col = col.upper()
        if col not in COLS or row not in ROWS:
            raise ValueError(f"Invalid well: {col}{row}")
        x_target = A1_X + COLS.index(col) * WELL_SPACING
        y_target = A1_Y + (int(row) - 1) * WELL_SPACING
        
        # Move to X,Y at safety height first, then to target Z
        print(f"📍 Moving to well {col}{row}: X={x_target:.3f}, Y={y_target:.3f}")
        gcode_xy = f"G01 X{x_target:.3f} Y{y_target:.3f} Z{Z_INITIAL:.3f} F{FEEDRATE}"
        self.send_gcode(gcode_xy)
        
        # Then move to target Z if different from safety height
        if z != Z_INITIAL:
            self.move_to_z(z)
        
        # Save position after movement
        self.save_position()


    def move_to_z(self, z: float, feedrate: float = FEEDRATE, wait_for_idle: bool = True):
        "Move to a designated absolute Z position."
        gcode = f"G01 Z{z:.3f} F{feedrate}"
        
        if wait_for_idle:
            success = self.send_gcode(gcode, wait_for_response=True)
            if success:
                self.wait_for_idle()
                print(f"✅ Moved to Z={z:.3f}")
            # Save position after movement
            if success:
                self.save_position()
        else:
            # Fast version - no waiting at all
            self.send_gcode(gcode, wait_for_response=False)
            print(f"🔄 Moving to Z={z:.3f} (no wait)")


    def home(self, zero_after: bool = True, timeout: float = 30.0):
        """Home the CNC machine with timeout"""
        print("🏠 Homing CNC...")
        
        try:
            # Send homing command
            print("> Sending: $H")
            self.ser.write(b'$H\n')
            
            # Wait for homing to complete with timeout
            start_time = time.time()
            while time.time() - start_time < timeout:
                try:
                    self.ser.reset_input_buffer()
                    self.ser.write(b'?\n')
                    status = self.ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if "Idle" in status:
                        print("✅ Homing completed")
                        break
                    elif "Alarm" in status:
                        print(f"⚠️ CNC in alarm state during homing: {status}")
                        return False
                    elif "Run" in status:
                        print("🔄 Homing in progress...")
                        
                    time.sleep(0.5)  # Check every 500ms
                    
                except Exception as e:
                    print(f"⚠️ Error during homing: {e}")
                    time.sleep(0.5)
            
            if time.time() - start_time >= timeout:
                print(f"⚠️ Homing timeout after {timeout}s")
                return False
            
            # Zero the machine if requested
            if zero_after:
                print("> Sending: G92 X0 Y0 Z0")
                self.ser.write(b'G92 X0 Y0 Z0\n')
                time.sleep(0.5)  # Brief wait for zero command
                print("✅ CNC homed and zeroed.")
            else:
                print("✅ CNC homed.")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during homing: {e}")
            return False


    def reset_grbl(self):
        """Reset GRBL if it's in an error state"""
        print("🔄 Resetting GRBL...")
        self.ser.write(b'$X\n')  # Unlock
        time.sleep(1)
        self.ser.write(b'$H\n')  # Home
        time.sleep(2)
        print("✅ GRBL reset complete")
    
    def emergency_stop(self):
        """Emergency stop - send stop command without waiting"""
        print("🛑 Emergency stop!")
        try:
            self.ser.write(b'\x18\n')  # Ctrl+X (soft reset)
            print("✅ Emergency stop sent")
        except Exception as e:
            print(f"❌ Error sending emergency stop: {e}")
    
    def unlock(self):
        """Unlock the CNC without homing"""
        print("🔓 Unlocking CNC...")
        try:
            self.ser.write(b'$X\n')
            time.sleep(0.5)
            print("✅ CNC unlocked")
            return True
        except Exception as e:
            print(f"❌ Error unlocking CNC: {e}")
            return False


    def get_current_position(self):
        "Get the current machine position from GRBL."
        start_time = time.time()
        self.ser.reset_input_buffer()
        self.ser.write(b'?\n')
        # time.sleep(0.1)
        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"📡 Raw position response: {response}")
        end_time = time.time()
        print(f"Time taken to get current position: {end_time - start_time} seconds")
        # Check for reset message
        if "Reset to continue" in response:
            print("⚠️ GRBL needs reset. Attempting to reset...")
            self.ser.write(b'$X\n')  # Unlock
            time.sleep(1)
            self.ser.write(b'?\n')  # Try again
            time.sleep(0.1)
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"📡 After reset response: {response}")
        
        if "MPos:" in response:
            try:
                mpos_section = response.split("MPos:")[1].split("|")[0]
                x, y, z = map(float, mpos_section.split(","))
                print(f"✅ Machine position: X={x}, Y={y}, Z={z}")
                end_time = time.time()
                print(f"Time taken to get current position: {end_time - start_time} seconds")
                return [x, y, z]
            
            except Exception as e:
                print(f"⚠️ Failed to parse position: {e}")
                return None
        else:
            print("❌ No position data in GRBL response.")
            return None


    def save_position(self):
        """Save current machine position to file"""
        current_pos = self.get_current_position()
        if current_pos:
            x, y, z = current_pos
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to src folder with timestamp format
            with open("src/last_position.csv", "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'X(mm)', 'Y(mm)', 'Z(mm)'])
                writer.writerow([timestamp, f"{x:.3f}", f"{y:.3f}", f"{z:.3f}"])
            
            print(f"💾 Position saved: X={x:.3f}, Y={y:.3f}, Z={z:.3f}")
            return True
        return False


    def sync_position(self):
        """Sync software position with actual machine position"""
        current_pos = self.get_current_position()
        if not current_pos:
            print("⚠️ Could not get current position - homing machine...")
            self.home()
            return True
        
        x, y, z = current_pos
        
        try:
            with open("src/last_position.csv", "r") as f:
                reader = csv.reader(f)
                next(reader)  # Skip header row
                row = next(reader)  # Get data row
                if len(row) >= 4:  # New format with timestamp
                    saved_x, saved_y, saved_z = float(row[1]), float(row[2]), float(row[3])
                else:  # Old format without timestamp
                    saved_x, saved_y, saved_z = float(row[0]), float(row[1]), float(row[2])
            
            # If saved position is (0,0,0), machine was properly homed last time
            if saved_x == 0 and saved_y == 0 and saved_z == 0:
                return True
            
            # If current position differs from saved position, home the machine
            if x != saved_x or y != saved_y or z != saved_z:
                self.home()
                return True
            else:
                return True
                        
        except FileNotFoundError:
            print("📂 No saved position found - homing machine...")
            self.home()
            return True


    def move_to_z_with_force_monitoring(self, z_target: float, force_sensor, 
                                       period_ms: int = 50, 
                                       feedrate: float = FEEDRATE,
                                       filename: Optional[str] = None) -> List[Tuple[float, float, float]]:
        """
        Move to Z position while monitoring force using the sensor's continuous reading capabilities.
        
        Args:
            z_target: Target Z position
            force_sensor: ForceSensor object
            period_ms: Sampling period in milliseconds (default: 50ms = 20 Hz)
            filename: CSV file to save the data
            
        Returns:
            List of tuples: (timestamp, z_position, force_reading)
        """
        print(f"📊 Moving to Z={z_target:.3f} with force monitoring every {period_ms}ms")
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/force_measurements/force_vs_z_{timestamp}.csv"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Get current position
        current_pos = self.get_current_position()
        if not current_pos:
            print("❌ Could not get current position")
            return []
        
        z_start = current_pos[2]
        print(f"📍 Starting from Z={z_start:.3f}")
        
        # Initialize data collection
        measurements = []
        start_time = time.time()
        
        # Check force sensor connection
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return []
        
        # Start continuous force monitoring
        if not force_sensor.start_continuous_monitoring(period_ms):
            print("❌ Could not start continuous force monitoring")
            return []
        
        try:
            # Send movement command (using the basic move_to_z logic)
            gcode = f"G01 Z{z_target:.3f} F{feedrate}"
            print(f"> Sending: {gcode}")
            self.ser.write((gcode + '\n').encode())
            
            # Monitor force and position during movement
            target_reached = False
            post_target_samples = 0
            max_post_target_samples = int(50 / period_ms)  # Continue for 50ms after reaching target
            z_current = z_start
            
            while not target_reached or post_target_samples < max_post_target_samples:
                current_time = time.time()
                
                # Get force reading using sensor's continuous reading
                force = force_sensor.get_continuous_reading()
                
                # Get position (query every sample for maximum resolution)
                pos = self.get_current_position()
                if pos:
                    z_current = pos[2]
                
                # Record measurement
                timestamp = current_time - start_time
                measurements.append((timestamp, z_current, force))
                
                # Print progress every 20 samples
                if len(measurements) % 20 == 0:
                    print(f"📊 Sample {len(measurements)}, t={timestamp:.2f}s, Z={z_current:.3f}mm, F={force:.3f}N")
                
                # Check if target is reached
                if abs(float(z_current) - z_target) < 0.1:
                    if not target_reached:
                        target_reached = True
                        print(f"🎯 Target Z={z_target:.3f} reached! Continuing monitoring...")
                    post_target_samples += 1
                else:
                    target_reached = False
                    post_target_samples = 0
                
                # Small sleep to prevent busy waiting
                time.sleep(0.001)  # 1ms sleep
            
        finally:
            # Stop continuous monitoring
            force_sensor.stop_continuous_monitoring()
        
        # Save data to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)'])
            for timestamp, z_pos, force_val in measurements:
                writer.writerow([f"{timestamp:.6f}", f"{z_pos:.3f}", f"{force_val:.6f}"])
        
        print(f"💾 Saved {len(measurements)} measurements to {filename}")
        print(f"✅ Movement complete. Final Z={z_target:.3f}")
        
        # Show sampling statistics
        if measurements:
            total_time = measurements[-1][0] - measurements[0][0]
            actual_freq = len(measurements) / total_time if total_time > 0 else 0
            print(f"📊 Actual sampling frequency: {actual_freq:.1f} Hz (period: {period_ms}ms)")
        
        # Save final position
        self.save_position()
        
        return measurements


    def monitor_force_at_z(self, z_position: float, force_sensor, 
                          duration: float = 2.0, period_ms: int = 50,
                          filename: Optional[str] = None) -> List[Tuple[float, float, float]]:
        """
        Monitor force at a specific Z position for a given duration using continuous reading.
        
        Args:
            z_position: Z position to monitor at
            force_sensor: ForceSensor object
            duration: Duration to monitor in seconds (default: 2.0)
            period_ms: Sampling period in milliseconds (default: 50ms = 20 Hz)
            filename: Optional custom filename
            
        Returns:
            List of tuples: (timestamp, z_position, force_reading)
        """
        print(f"📊 Monitoring force at Z={z_position:.3f} for {duration}s every {period_ms}ms")
        
        # Generate filename with timestamp if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/force_measurements/force_at_z{z_position:.1f}_{timestamp}.csv"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Move to target Z position first (using basic move_to_z)
        print(f"📍 Moving to Z={z_position:.3f}")
        self.move_to_z(z_position)
        
        # Initialize data collection
        measurements = []
        start_time = time.time()
        
        # Check force sensor connection
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return []
        
        # Start continuous force monitoring
        if not force_sensor.start_continuous_monitoring(period_ms):
            print("❌ Could not start continuous force monitoring")
            return []
        
        try:
            print(f"📊 Starting force monitoring for {duration} seconds...")
            
            # Monitor for specified duration
            while time.time() - start_time < duration:
                current_time = time.time()
                
                # Get force reading using sensor's continuous reading
                force = force_sensor.get_continuous_reading()
                
                # Get current position
                pos = self.get_current_position()
                z_current = pos[2] if pos else z_position
                
                # Record measurement
                timestamp = current_time - start_time
                measurements.append((timestamp, z_current, force))
                
                # Print every 10th sample
                if len(measurements) % 10 == 0:
                    print(f"📊 t={timestamp:.2f}s, Z={z_current:.3f}mm, F={force:.3f}N")
                
                time.sleep(0.001)  # Small delay to prevent excessive CPU usage
        
        finally:
            # Stop continuous monitoring
            force_sensor.stop_continuous_monitoring()
        
        # Save data to CSV
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Timestamp(s)', 'Z_Position(mm)', 'Force(N)'])
            for timestamp, z_pos, force_val in measurements:
                writer.writerow([f"{timestamp:.3f}", f"{z_pos:.3f}", f"{force_val:.3f}"])
        
        print(f"💾 Saved {len(measurements)} measurements to {filename}")
        
        return measurements


    def save_well_measurement(self, well: str, force_reading: float, z_position: float, 
                            filename: Optional[str] = None) -> str:
        """
        Save a measurement taken at a specific well.
        
        Args:
            well: Well identifier (e.g., 'A1', 'B2')
            force_reading: Force measurement in N
            z_position: Z position when measurement was taken
            filename: Optional custom filename
            
        Returns:
            Path to the saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"results/measurements/well_{well}_{timestamp}.csv"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Well', 'Timestamp', 'Force(N)', 'Z_Position(mm)'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([well, timestamp, f"{force_reading:.3f}", f"{z_position:.3f}"])
        
        print(f"💾 Well measurement saved to {filename}")
        return filename


    def move_to_pickup_position(self, y_position: float = 140.0, feedrate: float = FEEDRATE):
        """Move to a Y position that is convenient for the robot arm to pick up the well."""
        # Always check current Z position and move to safety height first
        current_pos = self.get_current_position()
        if not current_pos:
            print("❌ Could not get current position")
            return
        z_current = current_pos[2]
        if z_current != Z_INITIAL:
            self.move_to_z(Z_INITIAL)
            
        print(f"📍 Moving to pickup position: Y={y_position:.3f}")
        gcode = f"G01 Y{y_position:.3f} F{feedrate}"
        self.send_gcode(gcode)
        self.save_position()