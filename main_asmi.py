#!/usr/bin/env python3
"""
ASMI - Automated Sample Measurement Interface
Main entry point for the ASMI system
"""

from src.CNCController import CNCController
from src.ForceSensor import ForceSensor
import sys

def print_help():
    """Print available commands"""
    print("\n🔧 Available Commands:")
    print("  cnc.home()                    - Home the CNC machine")
    print("  cnc.move_to_well('A', '1')    - Move to specific well (e.g., A1)")
    print("  cnc.move_to_z(10)             - Move to specific Z height")
    print("  cnc.move_to_z(-5)             - Move down 5mm (negative Z)")
    print("  cnc.move_to_z_with_force_monitoring(-5, force) - Move down with force monitoring")
    print("  cnc.move_to_z_with_force_monitoring(-5, force, 500) - High-frequency monitoring (500 Hz)")
    print("  cnc.monitor_force_at_z(0, force) - Monitor force at top position (Z=0)")
    print("  cnc.get_current_position()    - Get current machine position")
    print("  cnc.save_well_measurement('A1', 1.5, 5.0) - Save well measurement")
    print("  force.get_force_reading()     - Take a single force measurement")
    print("  force.get_baseline_force()    - Get baseline force with statistics")
    print("\n🧪 Test Commands:")
    print("  test_force()                  - Quick force sensor test")
    print("  test_position()               - Quick position test")
    print("  test_force_monitoring()       - Test force monitoring (small movement)")
    print("  help                          - Show this help message")
    print("  quit or exit                  - Exit the program")
    print("  clear                         - Clear the terminal")
    print()

def main():
    """Main function to run the ASMI system with interactive interface"""
    print("🚀 Starting ASMI - Automated Sample Measurement Interface")
    print("📝 Type commands interactively. Type 'help' for available commands.")
    
    # Initialize components
    try:
        force_sensor = ForceSensor()
        cnc = CNCController()
        
        # Create local variables for easy access
        force = force_sensor
        
        # Define test functions
        def test_force():
            """Quick force sensor test"""
            print("🧪 Testing force sensor...")
            reading = force.get_force_reading()
            baseline_avg, baseline_std = force.get_baseline_force(samples=3)
            print(f"✅ Single reading: {reading:.3f} N")
            print(f"✅ Baseline: {baseline_avg:.3f} ± {baseline_std:.3f} N")
        
        def test_position():
            """Quick position test"""
            print("🧪 Testing position...")
            pos = cnc.get_current_position()
            if pos:
                print(f"✅ Current position: X={pos[0]:.3f}, Y={pos[1]:.3f}, Z={pos[2]:.3f}")
            else:
                print("❌ Could not get position")
        
        def test_force_monitoring():
            """Test force monitoring with small movement"""
            print("🧪 Testing force monitoring...")
            current_pos = cnc.get_current_position()
            if not current_pos:
                print("❌ Could not get current position")
                return
            
            current_z = current_pos[2]
            target_z = current_z - 1.0  # Move 1mm down (negative Z)
            
            print(f"📍 Moving from Z={current_z:.3f} (top) to Z={target_z:.3f} (down)")
            print("⚠️  Make sure there's nothing in the way!")
            
            response = input("Continue? (y/n): ").strip().lower()
            if response != 'y':
                print("❌ Test cancelled")
                return
            
            measurements = cnc.move_to_z_with_force_monitoring(
                target_z, force, period_ms=200
            )
            
            if measurements:
                print(f"✅ Test completed! {len(measurements)} measurements recorded")
            else:
                print("❌ No measurements recorded")
        
        print("\n✅ System ready! Type your commands:")
        print("Example: cnc.home()")
        print("Example: cnc.move_to_well('A', '1')")
        print("Example: cnc.move_to_z(-5)  # Move down 5mm")
        print("Example: force.get_force_reading()")
        print("Example: test_force_monitoring()")
        print()
        
        while True:
            try:
                # Get user input
                command = input("ASMI> ").strip()
                
                # Handle special commands
                if command.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif command.lower() == 'help':
                    print_help()
                    continue
                elif command.lower() == 'clear':
                    import os
                    os.system('clear' if os.name == 'posix' else 'cls')
                    continue
                elif not command:
                    continue
                
                # Execute the command
                print(f"🔧 Executing: {command}")
                result = eval(command)
                
                # Print result if it's not None
                if result is not None:
                    print(f"📊 Result: {result}")
                    
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error executing command: {e}")
                print("💡 Type 'help' for available commands")
    
    except Exception as e:
        print(f"❌ Error initializing system: {e}")
    
    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if 'force_sensor' in locals():
            force_sensor.cleanup()
        if 'cnc' in locals():
            cnc.close()
        print("✅ Cleanup complete")

if __name__ == "__main__":
    cnc = CNCController()
    cnc.home()  # Home the machine first
    cnc.send_gcode("G01 Y100 F1000")  # Add feedrate parameter
