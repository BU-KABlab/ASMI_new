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
A1_X = 98.5
A1_Y = 48
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


    def wait_for_idle(self):
        idle_counter = 0
        while True:
            self.ser.reset_input_buffer()
            self.ser.write(b'?\n')
            status = self.ser.readline().decode('utf-8').strip()
            if "Idle" in status:
                idle_counter += 1
            if idle_counter > 10:
                break
            time.sleep(0.1)


    def send_gcode(self, command):
        if not self.ser or not self.ser.is_open:
            raise serial.SerialException("❌ Serial port is not open. Cannot send G-code.")
        print(f"> Sending: {command}")
        self.ser.write((command + '\n').encode())
        if command.startswith("G92") or command.startswith("M") or command.startswith("$"):
            while True:
                response = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if response:
                    print(f"< Response: {response}")
                    break
        else:
            self.wait_for_idle()
            response = self.ser.readline().decode('utf-8', errors='ignore').strip()
            print(f"< Response: {response}")


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


    def move_to_z(self, z: float, feedrate: float = FEEDRATE):
        "Move to a designated absolute Z position."
        gcode = f"G01 Z{z:.3f} F{feedrate}"
        self.send_gcode(gcode)
        self.wait_for_idle()
        print(f"✅ Moved to Z={z:.3f}")
        
        # Save position after movement
        self.save_position()


    def home(self, zero_after: bool = True):
        print("🏠 Homing CNC...")
        
        self.send_gcode("$H")
        if zero_after:
            self.send_gcode("G92 X0 Y0 Z0")  # Zero after homing
            print("✅ CNC homed and zeroed.")
        else:
            print("✅ CNC homed.")


    def reset_grbl(self):
        """Reset GRBL if it's in an error state"""
        print("🔄 Resetting GRBL...")
        self.ser.write(b'$X\n')  # Unlock
        time.sleep(1)
        self.ser.write(b'$H\n')  # Home
        time.sleep(2)
        print("✅ GRBL reset complete")


    def get_current_position(self):
        "Get the current machine position from GRBL."
        self.ser.reset_input_buffer()
        self.ser.write(b'?\n')
        time.sleep(0.1)
        response = self.ser.readline().decode('utf-8', errors='ignore').strip()
        print(f"📡 Raw position response: {response}")
        
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
                return x, y, z
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
            # Save to both root directory (for compatibility) and results folder
            with open("last_position.csv", "w", newline="") as f:
                csv.writer(f).writerow([x, y, z])
            
            # Also save to results folder with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_filename = f"results/calibration/position_{timestamp}.csv"
            os.makedirs(os.path.dirname(results_filename), exist_ok=True)
            with open(results_filename, "w", newline="") as f:
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
            with open("last_position.csv", "r") as f:
                saved_x, saved_y, saved_z = map(float, csv.reader(f).__next__())
            
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
                if abs(z_current - z_target) < 0.1:
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
            filename = f"results/well_measurements/well_{well}_{timestamp}.csv"
        
        # Ensure results directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Well', 'Timestamp', 'Force(N)', 'Z_Position(mm)'])
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([well, timestamp, f"{force_reading:.3f}", f"{z_position:.3f}"])
        
        print(f"💾 Well measurement saved to {filename}")
        return filename
