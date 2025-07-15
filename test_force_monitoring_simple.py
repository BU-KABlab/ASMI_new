#!/usr/bin/env python3
"""
Simplified test script for force monitoring functionality
Just specify wells_to_test and run!
"""

from src.force_monitoring import run_force_monitoring_test

def run_test(wells_to_test, target_z, force_limit, period_ms, feedrate):
    print("🧪 Force Monitoring Test")
    print("=" * 30)
    print(f"📋 Wells to test: {wells_to_test}")
    print(f"🎯 Target Z: {target_z} mm")
    print(f"⚠️ Force limit: {force_limit} N")
    print(f"📊 Sampling period: {period_ms} ms")
    print(f"🚀 Feedrate: {feedrate}")
    print()
    
    # Run the test
    results = run_force_monitoring_test(
        wells_to_test=wells_to_test,
        target_z=target_z,
        force_limit=force_limit,
        period_ms=period_ms,
        feedrate=feedrate
    )
    
    # Print final summary
    if results:
        print(f"\n📊 Final Summary:")
        print(f"✅ Wells tested: {len(results)}")
        force_limit_count = sum(1 for r in results if r['force_exceeded'])
        print(f"⚠️ Force limit exceeded at {force_limit_count} wells")
        
        if force_limit_count > 0:
            print(f"\n⚠️ Wells with force limit exceeded:")
            for result in results:
                if result['force_exceeded']:
                    print(f"   - {result['well']}: stopped at Z={result['stop_z']:.3f}mm, max force={result['max_force']:.3f}N")
    else:
        print("❌ No results returned from test") 
        

if __name__ == "__main__":
    
    # Optional: Customize test parameters
    target_z = -14.0      # Target Z position in mm
    force_limit = 45.0    # Force limit in N
    period_ms = 10        # Sampling period in milliseconds
    feedrate = 200        # Movement feedrate
    
    # Define the wells you want to test
    wells_to_test = ["A1"]
    
    run_test(wells_to_test, target_z, force_limit, period_ms, feedrate)
    