#!/usr/bin/env python3
"""
Test script for dynamic indentation functionality
Demonstrates the new dynamic indentation control that stops based on:
1. Indents for 1mm after contact detection
2. Reaches force limit
3. Reaches z_target
"""

from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
from src.force_monitoring import dynamic_indentation_measurement
import time

def test_dynamic_indentation():
    """Test the dynamic indentation measurement"""
    print("🧪 Testing Dynamic Indentation Measurement")
    print("=" * 50)
    
    try:
        # Initialize CNC and force sensor
        print("🔧 Initializing CNC controller...")
        cnc = CNCController()
        
        print("🔧 Initializing force sensor...")
        force_sensor = ForceSensor()
        
        if not force_sensor.is_connected():
            print("❌ Force sensor not connected")
            return False
        
        # Test parameters
        well = "A1"
        z_target = -12.0  # Target Z position
        step_size = 0.1   # Step size for movement
        force_limit = 45.0  # Force limit in N
        max_indentation_depth = 1.0  # Max indentation depth after contact
        
        print(f"🎯 Test Parameters:")
        print(f"   Well: {well}")
        print(f"   Target Z: {z_target:.1f}mm")
        print(f"   Step size: {step_size:.1f}mm")
        print(f"   Force limit: {force_limit:.1f}N")
        print(f"   Max indentation depth: {max_indentation_depth:.1f}mm")
        print()
        
        # Move to well
        print(f"📍 Moving to well {well}...")
        col, row = well[0], well[1:]
        cnc.move_to_well(col, row)
        
        # Run dynamic indentation measurement
        print("🔄 Starting dynamic indentation measurement...")
        success = dynamic_indentation_measurement(
            cnc=cnc,
            force_sensor=force_sensor,
            well=well,
            z_target=z_target,
            step_size=step_size,
            force_limit=force_limit,
            max_indentation_depth=max_indentation_depth
        )
        
        if success:
            print("✅ Dynamic indentation test completed successfully!")
        else:
            print("❌ Dynamic indentation test failed!")
        
        return success
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        return False
    finally:
        try:
            cnc.close()
            force_sensor.cleanup()
        except:
            pass

if __name__ == "__main__":
    test_dynamic_indentation() 