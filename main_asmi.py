#!/usr/bin/env python3
"""
ASMI runner — uses CubOS protocol engine for measurement and SQL persistence.

Usage:
    python3 main_asmi.py              # real hardware, configs from configs/
    python3 main_asmi.py --mock       # offline dry run with mock instruments
    python3 main_asmi.py --no-home    # skip homing (use if already homed via UGS)
    python3 main_asmi.py --skip-force-sensor  # real gantry, mock force sensor (for GoDirect issues)

Configuration lives in YAML files under configs/:
    gantry/<gantry>.yaml             — serial port, working volume, GRBL,
                                       and `instruments:` (mounted asmi)
    deck/asmi_deck.yaml              — well plate geometry and calibration
    protocol/<protocol>.yaml         — protocol command sequence
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
_GANTRY_YAML = _CONFIGS / 'gantry' / 'new_asmi_gantry_calibration.yaml'
_DECK_YAML = _CONFIGS / 'deck' / 'asmi_deck.yaml'
_PROTOCOL_YAML = _CONFIGS / 'protocol' / 'asmi_indentation_test.yaml'
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
        time.sleep(1.0)
        gantry.set_serial_timeout(0.05)

    # Load protocol (mock instruments when --mock or --skip-force-sensor).
    # Instruments live under the gantry YAML's `instruments:` block, so no
    # separate board YAML is needed.
    protocol, context = setup_protocol(
        str(_GANTRY_YAML), str(_DECK_YAML), str(_PROTOCOL_YAML),
        gantry=gantry, mock_mode=mock or skip_force_sensor,
    )

    # NB: wells to be measured are owned by the protocol YAML (e.g. `measure
    # position: plate.E5`, or `scan` which iterates the full plate).
    # main_asmi.py used to filter plate.wells from analysis.yaml's
    # `wells_to_test` here — that was a separation-of-concerns violation
    # (analysis-time config dictating protocol behavior) and broke `measure`
    # protocols whose hard-coded position got stripped out before the protocol
    # ran. Removed; analysis.yaml is now strictly post-processing config.

    # DataStore — only persist real measurements. Mocked runs (--mock or
    # --skip-force-sensor) produce synthetic zero-force data that would
    # otherwise pollute the analysis DB.
    store: DataStore | None = None
    if not (mock or skip_force_sensor):
        store = DataStore(db_path=db_path)
        context.data_store = store
        context.campaign_id = store.create_campaign(
            description='ASMI indentation run',
            deck_config=str(_DECK_YAML),
            board_config=str(_GANTRY_YAML),
            gantry_config=str(_GANTRY_YAML),
            protocol_config=str(_PROTOCOL_YAML),
        )
        print(f"Campaign {context.campaign_id} created in {db_path}")
    else:
        reason = "--mock" if mock else "--skip-force-sensor"
        print(f"DataStore disabled ({reason}); no campaign row will be written.")

    # Connect instruments and run
    if skip_force_sensor:
        print("Using mock force sensor (--skip-force-sensor).")
    print("Connecting force sensor...")
    context.board.connect_instruments()
    print("Starting measurement.")
    try:
        results = protocol.run(context)
        print(f"\nProtocol complete: {len(results)} step(s) executed.")
        for i, result in enumerate(results):
            if not isinstance(result, dict):
                continue
            # `measure` returns the bare indentation result; `scan` returns
            # {well_id: result}. Detect which by looking for the canonical
            # measurement keys.
            if 'data_points' in result:
                pts = result.get('data_points', 0)
                exceeded = result.get('force_exceeded', False)
                print(f"  step {i} (measure): {pts} points, force_exceeded={exceeded}")
            else:
                for well_id, well_result in result.items():
                    if not isinstance(well_result, dict):
                        continue
                    pts = well_result.get('data_points', 0)
                    exceeded = well_result.get('force_exceeded', False)
                    print(f"  step {i} {well_id}: {pts} points, force_exceeded={exceeded}")
    except KeyboardInterrupt:
        print("\nAborted by user.")
    finally:
        context.board.disconnect_instruments()
        if not mock:
            gantry.disconnect()
        if store is not None:
            store.close()
            print(f"Data persisted to {db_path}")


if __name__ == "__main__":
    run(
        mock="--mock" in sys.argv,
        skip_home="--no-home" in sys.argv,
        skip_force_sensor="--skip-force-sensor" in sys.argv,
    )
