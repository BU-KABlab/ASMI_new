#!/usr/bin/env python3
"""
Test script: move to wells A1, B1, H12, etc.

safe_z (Z_INITIAL) is higher than well_top_z (e.g. -50 vs -73 mm).
Flow: at safe_z move XY to well -> move DOWN to well_top_z -> wait 1s -> move UP to safe_z.

Usage:
    python move_to_well_test.py
    python move_to_well_test.py --wells A1,B2,H12
    python move_to_well_test.py --well-top-z -73.0 --no-home
"""

import argparse
import time

from src.CNCController import CNCController, Z_INITIAL


def parse_well(well: str) -> tuple[str, str]:
    """Parse 'A1', 'B2', 'H12' -> (col, row)."""
    well = well.strip().upper()
    if len(well) < 2:
        raise ValueError(f"Invalid well: {well}")
    col = well[0]
    row = well[1:]
    return col, row


def main():
    parser = argparse.ArgumentParser(description="Test move to wells")
    parser.add_argument(
        "--wells",
        type=str,
        default="A1,H12",
        help="Comma-separated wells to test (default: A1,B1,H12)",
    )
    parser.add_argument(
        "--well-top-z",
        type=float,
        default=-73.0,
        help="Z position at well top (default: -73.0 mm)",
    )
    parser.add_argument(
        "--no-home",
        action="store_true",
        help="Skip homing at start",
    )
    args = parser.parse_args()

    wells = [w.strip() for w in args.wells.split(",") if w.strip()]
    if not wells:
        print("❌ No wells specified")
        return 1

    print(f"📋 Wells to test: {wells}")
    print(f"📐 safe_z (Z_INITIAL): {Z_INITIAL} mm  (higher, above plate)")
    print(f"📐 well_top_z: {args.well_top_z} mm  (lower, at well surface)")

    cnc = CNCController()
    try:
        if not args.no_home:
            print("\n🏠 Homing...")
            if not cnc.home(zero_after=True):
                print("⚠️ Homing failed, proceeding with caution")
            time.sleep(0.5)

        for well in wells:
            try:
                col, row = parse_well(well)
            except ValueError as e:
                print(f"⚠️ Skip invalid well '{well}': {e}")
                continue

            print(f"\n📍 Well {well} (col={col}, row={row})")
            print("   At safe_z: move XY to well -> move DOWN to well_top_z -> wait 1s -> move UP to safe_z")
            cnc.move_to_well(col, row, z=args.well_top_z)
            print("   Waiting 1 s...")
            time.sleep(1)
            print("   Returning to safe_z...")
            cnc.move_to_safe_z()
            print(f"   ✅ Done with {well}")

        print("\n✅ All wells tested")
        return 0
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user")
        return 1
    finally:
        cnc.close()


if __name__ == "__main__":
    exit(main())
