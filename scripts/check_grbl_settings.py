#!/usr/bin/env python3
"""
Check what GRBL settings the controller actually reports when connected.
Run this on the same machine/port as main_asmi.py to diagnose mismatches.

Usage: python scripts/check_grbl_settings.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml

with open(ROOT / "configs/gantry/asmi_gantry.yaml") as f:
    cfg = yaml.safe_load(f)

port = cfg.get("serial_port", "/dev/ttyUSB0")
print(f"Connecting to {port}...")

from gantry.gantry import Gantry

gantry = Gantry(config=cfg)
try:
    gantry._mill.connect_to_mill(port=port)
    live = gantry._mill.grbl_settings()
    gantry._mill.disconnect()
except Exception as e:
    print(f"Connection failed: {e}")
    sys.exit(1)

# Show critical settings
critical = {
    "$100": "steps_per_mm_x",
    "$101": "steps_per_mm_y",
    "$102": "steps_per_mm_z",
    "$130": "max_travel_x",
    "$131": "max_travel_y",
    "$132": "max_travel_z",
}

expected = cfg.get("grbl_settings", {})
field_to_grbl = {
    "steps_per_mm_x": "$100",
    "steps_per_mm_y": "$101",
    "steps_per_mm_z": "$102",
    "max_travel_x": "$130",
    "max_travel_y": "$131",
    "max_travel_z": "$132",
}

print("\n--- Controller (live) vs YAML (expected) ---")
for grbl, name in critical.items():
    live_val = live.get(grbl, "?")
    exp_val = expected.get(name, "?")
    match = "✓" if str(live_val) == str(exp_val) else "✗ MISMATCH"
    print(f"  {name}: live={live_val}  expected={exp_val}  {match}")
