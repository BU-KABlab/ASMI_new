import time
import statistics
from godirect import GoDirect
from typing import Optional, Tuple

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
            
        self.device.start()
        value = 0.0
        if self.device.read():
            value = self.sensor.values[0]
            self.sensor.clear()
        self.device.stop()
        print(f"📊 Force reading: {value:.3f} N")
        return value


    def get_baseline_force(self, samples: int = 5) -> Tuple[float, float]:
        """Get baseline force reading with standard deviation.
        Return: tuple of average force reading, standard deviation"""
        if not self.is_connected():
            print("⚠️ Sensor not connected")
            return 0.0, 0.0
            
        if not self.device or not self.sensor:
            return 0.0, 0.0
            
        print(f"📈 Taking {samples} baseline measurements...")
        measurements = []
        self.device.start()
        for i in range(samples):
            if self.device.read():
                measurements.append(self.sensor.values[0])
                self.sensor.clear()
            time.sleep(0.05)
        self.device.stop()
        
        if not measurements:
            print("❌ No measurements collected")
            return 0.0, 0.0
        
        avg, std = statistics.mean(measurements), statistics.stdev(measurements)
        print(f"📊 Baseline: {avg:.3f} ± {std:.3f} N")
        return avg, std


    def start_continuous_monitoring(self, period_ms=50):
        """Start continuous monitoring mode for high-frequency sampling
        
        Args:
            period_ms: Sampling period in milliseconds (default: 50ms = 20 Hz)
                      Lower values = higher frequency (e.g., 1ms = 1000 Hz)
        """
        if not self.is_connected():
            print("⚠️ Sensor not connected")
            return False
        
        if not self.device:
            print("❌ Device not available")
            return False
        
        try:
            # Calculate frequency for logging
            freq = 1000.0 / period_ms
            print(f"✅ Starting continuous monitoring at {freq:.1f} Hz (period: {period_ms}ms)")
            self.device.start(period=period_ms)
            return True
        except Exception as e:
            print(f"❌ Error starting continuous monitoring: {e}")
            return False


    def stop_continuous_monitoring(self):
        """Stop continuous monitoring mode"""
        if self.device:
            try:
                self.device.stop()
                print("✅ Continuous monitoring stopped")
            except Exception as e:
                print(f"⚠️ Error stopping continuous monitoring: {e}")


    def get_continuous_reading(self) -> float:
        """Get a single reading during continuous monitoring (faster than get_force_reading)"""
        if not self.is_connected() or not self.device or not self.sensor:
            return 0.0
        
        try:
            if self.device.read():
                value = self.sensor.values[0]
                self.sensor.clear()
                return value
            else:
                return 0.0
        except Exception as e:
            print(f"⚠️ Error reading force: {e}")
            return 0.0

    def cleanup(self):
        """Clean up device connection"""
        if self.device:
            self.device.close()
            print("🔒 Force sensor connection closed")
        self.device = None
        self.godirect = None