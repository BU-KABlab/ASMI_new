#!/usr/bin/env python3
"""
ASMI runner — uses PANDA_CORE protocol engine for measurement and SQL persistence.

Usage:
    python3 main_asmi.py              # real hardware, configs from configs/
    python3 main_asmi.py --mock       # offline dry run with mock instruments
    python3 main_asmi.py --no-home    # skip homing (use if already homed via UGS)
    python3 main_asmi.py --skip-force-sensor  # real gantry, mock force sensor (for GoDirect issues)

Configuration lives in YAML files under configs/:
    gantry/asmi_gantry.yaml          — CNC serial port, working volume, GRBL
    deck/asmi_deck.yaml              — well plate geometry and calibration
    board/asmi_board.yaml            — ASMI instrument with indentation params
    protocol/asmi_indentation.yaml   — scan protocol (method: indentation)
    analysis.yaml                    — wells, analysis, workflow

Troubleshooting:
    • GRBL Alarm: Unlock ($X) and home ($H) clear it. The runner does both;
      if alarm persists, home manually via UGS first, then use --no-home.
    • struct.error "unpack requires a buffer of 18 bytes": GoDirect firmware/
      protocol mismatch. Use --skip-force-sensor to run with mock force data,
      or try: pip install -U godirect; different USB port; verify sensor model.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import yaml

from gantry.gantry import Gantry
from protocol_engine.setup import setup_protocol
from data.data_store import DataStore

_PROJECT_ROOT = Path(__file__).resolve().parent

_CONFIGS = _PROJECT_ROOT / 'configs'
_GANTRY_YAML = _CONFIGS / 'gantry' / 'asmi_gantry.yaml'
_DECK_YAML = _CONFIGS / 'deck' / 'asmi_deck.yaml'
_BOARD_YAML = _CONFIGS / 'board' / 'asmi_board.yaml'
_PROTOCOL_YAML = _CONFIGS / 'protocol' / 'asmi_indentation.yaml'
_EXPERIMENT_YAML = _CONFIGS / 'analysis.yaml'


def run(
    mock: bool = False,
    skip_home: bool = False,
    skip_force_sensor: bool = False,
) -> None:
    """Run the ASMI indentation protocol with SQL persistence."""
    with open(_EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)

    db_path = cfg.get('paths', {}).get('database', 'data/asmi_data.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    gantry_cfg = cfg.get('gantry', {})

    # Build gantry
    if mock:
        gantry = Gantry(offline=True)
    else:
        with open(_GANTRY_YAML) as f:
            gantry_config = yaml.safe_load(f)
        gantry = Gantry(config=gantry_config)
        gantry.connect()
        gantry.set_serial_timeout(10)
        gantry.unlock()
        gantry.configure_speeds(
            homing_feed=gantry_cfg.get('homing_feed'),
            homing_seek=gantry_cfg.get('homing_seek'),
            max_rate=gantry_cfg.get('max_rate'),
            acceleration=gantry_cfg.get('acceleration'),
        )
        time.sleep(gantry_cfg.get('unlock_delay', 1.0))
        if not skip_home and cfg.get('workflow', {}).get('home_before_measure', True):
            gantry.home()
            gantry.zero_coordinates()
        gantry.set_safe_z(gantry_cfg.get('safe_z', -50.0))
        gantry.set_serial_timeout(0.05)

    # Load protocol (mock instruments when --mock or --skip-force-sensor)
    protocol, context = setup_protocol(
        str(_GANTRY_YAML), str(_DECK_YAML), str(_BOARD_YAML),
        str(_PROTOCOL_YAML), gantry=gantry, mock_mode=mock or skip_force_sensor,
    )

    # Filter wells if specified in experiment config
    wells_to_test = cfg.get('wells', {}).get('wells_to_test')
    if wells_to_test:
        plate = context.deck['plate']
        keep = {str(w).upper() for w in wells_to_test}
        to_remove = [w for w in plate.wells if w not in keep]
        for w in to_remove:
            del plate.wells[w]
        print(f"Measuring {len(plate.wells)} wells: {sorted(plate.wells.keys())}")

    # DataStore
    store = DataStore(db_path=db_path)
    context.data_store = store
    context.campaign_id = store.create_campaign(
        description='ASMI indentation run',
        deck_config=str(_DECK_YAML),
        board_config=str(_BOARD_YAML),
        gantry_config=str(_GANTRY_YAML),
        protocol_config=str(_PROTOCOL_YAML),
    )
    print(f"Campaign {context.campaign_id} created in {db_path}")

    # Connect instruments and run
    if skip_force_sensor:
        print("Using mock force sensor (--skip-force-sensor).")
    print("Connecting force sensor...")
    context.board.connect_instruments()
    print("Starting measurement.")
    try:
        results = protocol.run(context)
        scan_results = results[0] if results else {}
        print(f"\n{len(scan_results)} wells measured")
        for well_id, result in scan_results.items():
            pts = result.get('data_points', 0)
            exceeded = result.get('force_exceeded', False)
            print(f"  {well_id}: {pts} points, force_exceeded={exceeded}")
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        context.board.disconnect_instruments()
        if not mock:
            try:
                gantry.set_serial_timeout(5)
                gantry.home()
            except Exception:
                pass
            gantry.disconnect()
        store.close()
        print(f"Data persisted to {db_path}")


if __name__ == "__main__":
    run(
        mock="--mock" in sys.argv,
        skip_home="--no-home" in sys.argv,
        skip_force_sensor="--skip-force-sensor" in sys.argv,
    )
