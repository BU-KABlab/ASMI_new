#!/usr/bin/env python3
"""
Advanced test script for force monitoring functionality
Supports command-line arguments for flexible testing
"""

import argparse
import sys
from src.force_monitoring import run_force_monitoring_test

def parse_wells(wells_string):
    """Parse wells string into list (e.g., 'A1,A2,B1,B2' -> ['A1', 'A2', 'B1', 'B2'])"""
    if not wells_string:
        return []
    return [well.strip() for well in wells_string.split(',') if well.strip()]

def main():
    parser = argparse.ArgumentParser(
        description='Force monitoring test for ASMI system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test specific wells
  python test_force_monitoring_advanced.py --wells A1,A2,B1,B2
  
  # Test with custom parameters
  python test_force_monitoring_advanced.py --wells A6,B6,C6 --target-z -20 --force-limit 50
  
  # Test with high precision
  python test_force_monitoring_advanced.py --wells A5,B5,C5 --period-ms 5 --feedrate 100
  
  # Test all wells in a pattern
  python test_force_monitoring_advanced.py --wells A1,A2,A3,A4,A5,A6,B1,B2,B3,B4,B5,B6,C1,C2,C3,C4,C5,C6
        """
    )
    
    parser.add_argument(
        '--wells', 
        type=str, 
        default='A6,B6,C6,C5,B5,A5',
        help='Comma-separated list of wells to test (default: A6,B6,C6,C5,B5,A5)'
    )
    
    parser.add_argument(
        '--target-z', 
        type=float, 
        default=-15.0,
        help='Target Z position in mm (default: -15.0)'
    )
    
    parser.add_argument(
        '--force-limit', 
        type=float, 
        default=45.0,
        help='Force limit in N (default: 45.0)'
    )
    
    parser.add_argument(
        '--period-ms', 
        type=int, 
        default=10,
        help='Sampling period in milliseconds (default: 10)'
    )
    
    parser.add_argument(
        '--feedrate', 
        type=int, 
        default=200,
        help='Movement feedrate (default: 200)'
    )
    
    parser.add_argument(
        '--no-confirm', 
        action='store_true',
        help='Skip confirmation prompt'
    )
    
    args = parser.parse_args()
    
    # Parse wells
    wells_to_test = parse_wells(args.wells)
    if not wells_to_test:
        print("❌ No valid wells specified")
        sys.exit(1)
    
    # Display test configuration
    print("🧪 Force Monitoring Test (Advanced)")
    print("=" * 40)
    print(f"📋 Wells to test: {wells_to_test}")
    print(f"🎯 Target Z: {args.target_z} mm")
    print(f"⚠️ Force limit: {args.force_limit} N")
    print(f"📊 Sampling period: {args.period_ms} ms")
    print(f"🚀 Feedrate: {args.feedrate}")
    print()
    
    # Ask for confirmation unless --no-confirm is used
    if not args.no_confirm:
        response = input("Continue with test? (y/n): ").strip().lower()
        if response != 'y':
            print("❌ Test cancelled")
            sys.exit(0)
    
    # Run the test
    print("🚀 Starting test...")
    results = run_force_monitoring_test(
        wells_to_test=wells_to_test,
        target_z=args.target_z,
        force_limit=args.force_limit,
        period_ms=args.period_ms,
        feedrate=args.feedrate
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
        
        # Show successful wells
        successful_wells = [r['well'] for r in results if not r['force_exceeded']]
        if successful_wells:
            print(f"\n✅ Wells completed successfully:")
            for well in successful_wells:
                print(f"   - {well}")
    else:
        print("❌ No results returned from test")
        sys.exit(1)

if __name__ == "__main__":
    main() 