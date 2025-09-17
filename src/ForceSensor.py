"""
Force Sensor Module for ASMI

Interfaces with GoDirect force sensors for real-time force measurements.

Author: Hongrui Zhang
Date: 09/2025
License: MIT
"""

import time
import statistics
from godirect import GoDirect
from typing import Optional, Tuple
from .version import __version__

# === FORCE SENSOR CONFIGURATION ===
THRESHOLD = -100  # Minimum force reading to consider valid

class ForceSensor:
    def __init__(self, port=None):
        print("🔧 Initializing force sensor...")
        self.port = port
        self.godirect: Optional[GoDirect] = GoDirect(use_ble=False, use_usb=True)
        self.device = None
        self.sensor = None
        self.connected = False
        self.connect()


    def connect(self) -> bool:
        """Initialize connection with the GoDirect force sensor"""
        try:
            print("🔍 Searching for GoDirect devices...")
            if not self.godirect:
                print("❌ GoDirect object is None")
                return False
                
            device = self.godirect.get_device(threshold=THRESHOLD)
            if not device:
                print("❌ No GoDirect devices found. Check USB connection and power.")
                return False
            
            self.device = device
            print(f"✅ Found device: {self.device}")
            
            if not self.device.open(auto_start=False):
                print("❌ Failed to open device connection")
                return False
            
            # Enable sensors
            try:
                print("🔧 Enabling sensors...")
                self.device.enable_sensors([1])
            except Exception as e:
                print(f"⚠️ Error enabling sensors: {e}")
            
            sensors = self.device.get_enabled_sensors()
            if not sensors:
                print("❌ No sensors enabled. Try closing other programs or replugging sensor.")
                return False
            
            self.sensor = sensors[0]
            print(f"✅ Connected to sensor: {self.sensor.sensor_description}")
            self.connected = True
            return True
            
        except Exception as e:
            print(f"❌ Error connecting to force sensor: {e}")
            return False


    def is_connected(self) -> bool:
        """Check if sensor is connected and working"""
        return bool(self.connected and self.device and self.sensor)


    def get_force_reading(self) -> float:
        """Take a single force measurement.
        Return: force reading"""
        if not self.is_connected():
            print("⚠️ Sensor not connected")
            return 0.0
            
        if not self.device or not self.sensor:
            return 0.0
            
        start_time = time.time()
        self.device.start()
        value = 0.0
        if self.device.read():
            value = self.sensor.values[0]
            self.sensor.clear() 
        self.device.stop()
        end_time = time.time()
        print(f"📊 Force reading: {value:.3f} N, Time taken: {end_time - start_time} seconds")
        return value
    
    
    def get_baseline_force(self, samples: int = 10) -> Tuple[float, float]:
        """Get baseline force reading with standard deviation.
        Return: tuple of average force reading, standard deviation"""
        measurements = []
        for i in range(samples):
            measurements.append(self.get_force_reading())
        avg, std = statistics.mean(measurements), statistics.stdev(measurements)
        print(f"📊 Baseline: {avg:.3f} ± {std:.3f} N")
        return avg, std


    def get_continuous_force_reading(self, duration_seconds=10):
        """Get continuous force reading for specified duration"""
        print(f"🔄 Starting continuous force reading for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early")
        
        start_time = time.time()
        readings = []
        
        try:
            while time.time() - start_time < duration_seconds:
                force = self.get_force_reading()
                readings.append(force)
        except KeyboardInterrupt:
            print("\n🔴 Ctrl+C pressed, stopping continuous force reading")
        
        print(f"📊 Completed {len(readings)} readings in {time.time() - start_time:.1f} seconds")
        return readings


    def cleanup(self):
        """Clean up device connection"""
        if self.device:
            self.device.close()
            print("🔒 Force sensor connection closed")
        self.device = None
        self.godirect = None
        
