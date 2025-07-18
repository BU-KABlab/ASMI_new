#!/usr/bin/env python3
"""
Test script to compare timing between different force reading methods.
This will show the real performance differences including setup overhead.
"""

import time
from src.ForceSensor import ForceSensor

def test_timing_comparison(num_readings=10):
    """Compare timing between get_force_reading, get_baseline_force, and continuous monitoring"""
    
    print("🔧 Initializing force sensor...")
    sensor = ForceSensor()
    
    if not sensor.is_connected():
        print("❌ Sensor not connected. Exiting.")
        return
    
    print("\n" + "="*60)
    print("TIMING COMPARISON TEST")
    print("="*60)
    
    # Test 1: Single get_force_reading
    print("\n1️⃣ Testing get_force_reading() (single reading):")
    start_total = time.time()
    value = sensor.get_force_reading()
    end_total = time.time()
    print(f"📊 Total time for get_force_reading(): {end_total - start_total:.3f} seconds")
    
    # Test 2: get_baseline_force with 5 samples
    print("\n2️⃣ Testing get_baseline_force() (5 samples):")
    start_total = time.time()
    avg, std = sensor.get_baseline_force(samples=num_readings)
    end_total = time.time()
    print(f"📊 Total time for get_baseline_force(5): {end_total - start_total:.3f} seconds")
    print(f"📊 Result: {avg:.3f} ± {std:.3f} N")
    
    # Test 3: Continuous monitoring (including setup overhead)
    print("\n3️⃣ Testing continuous monitoring (including setup):")
    sensor.get_continuous_force_reading()
    
    
    sensor.cleanup()

if __name__ == "__main__":
    test_timing_comparison() 