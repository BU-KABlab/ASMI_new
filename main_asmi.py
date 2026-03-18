#!/usr/bin/env python3
"""
ASMI runner — uses PANDA_CORE protocol engine for measurement and SQL persistence.

Usage:
    python3 main_asmi.py              # real hardware, configs from configs/
    python3 main_asmi.py --mock       # offline dry run with mock instruments
    python3 main_asmi.py --no-home    # skip homing (use if already homed via UGS)

Configuration lives in YAML files under configs/:
    gantry/asmi_gantry.yaml          — CNC serial port, working volume, GRBL
    deck/asmi_deck.yaml              — well plate geometry and calibration
    board/asmi_board.yaml            — ASMI instrument with indentation params
    protocol/asmi_indentation.yaml   — scan protocol (method: indentation)
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

# ── Config paths ──────────────────────────────────────────────────────────
_CONFIGS = _PROJECT_ROOT / 'configs' 
_GANTRY_YAML = _CONFIGS / 'gantry' / 'asmi_gantry.yaml' 
_DECK_YAML = _CONFIGS / 'deck' / 'asmi_deck.yaml' 
_BOARD_YAML = _CONFIGS / 'board' / 'asmi_board.yaml'
_MOCK_BOARD_YAML = _CONFIGS / 'board' / 'mock_asmi_board.yaml' 
_PROTOCOL_YAML = _CONFIGS / 'protocol' / 'asmi_indentation.yaml' 
_EXPERIMENT_YAML = _CONFIGS / 'experiment.yaml' 


def run(mock: bool = False, skip_home: bool = False) -> None:
    """Run the ASMI indentation protocol with SQL persistence.

    Args:
        mock: If True, use Gantry(offline=True) + mock instruments.
    """
    os.makedirs(_DB_PATH.parent, exist_ok=True)

    # Build gantry
    if mock:
        gantry = Gantry(offline=True)
    else:
        with open(_GANTRY_YAML) as f:
            gantry_config = yaml.safe_load(f)
        gantry = Gantry(config=gantry_config)
        gantry.connect()
        # Increase serial timeout for homing (GRBL may be slow to respond when moving)
        gantry._mill.ser_mill.timeout = 10
        gantry.unlock()
        # Apply speed overrides from config (homing + movement)
        gantry_cfg = cfg.get("gantry", {})
        mill = gantry._mill
        if gantry_cfg.get("homing_feed") is not None:
            mill.set_grbl_setting("24", str(gantry_cfg["homing_feed"]))
        if gantry_cfg.get("homing_seek") is not None:
            mill.set_grbl_setting("25", str(gantry_cfg["homing_seek"]))
        if gantry_cfg.get("max_rate") is not None:
            for axis, code in [("X", "110"), ("Y", "111"), ("Z", "112")]:
                mill.set_grbl_setting(code, str(gantry_cfg["max_rate"]))
        if gantry_cfg.get("acceleration") is not None:
            for axis, code in [("X", "120"), ("Y", "121"), ("Z", "122")]:
                mill.set_grbl_setting(code, str(gantry_cfg["acceleration"]))
        # Wait after unlock — if machine was at limit switch (Pn:X), give it time to clear
        unlock_delay = cfg.get("gantry", {}).get("unlock_delay", 1.0)
        time.sleep(unlock_delay)
        do_home = not skip_home and cfg.get('workflow', {}).get('home_before_measure', True)
        if do_home:
            gantry.home()
            gantry._mill.execute_command("G92 X0 Y0 Z0")
        # Use safe_z as clearance height for XY moves (move to Z first, then XY to well)
        safe_z = cfg.get('gantry', {}).get('safe_z', -50.0)
        gantry._mill.max_z_height = float(safe_z)
        # Short serial timeout for fast status polling during moves (0.1s per read)
        gantry._mill.ser_mill.timeout = 0.1

    # Load protocol (mock_mode swaps asmi -> mock_asmi in board loader)
    protocol, context = setup_protocol(
        str(_GANTRY_YAML), str(_DECK_YAML), str(_BOARD_YAML),
        str(_PROTOCOL_YAML), gantry=gantry, mock_mode=mock,
    )

    # Filter wells if wells_to_test is specified in experiment config
    wells_to_test = cfg.get("wells", {}).get("wells_to_test")
    if wells_to_test:
        plate = context.deck["plate"]
        keep = {str(w).upper() for w in wells_to_test}
        to_remove = [w for w in plate.wells if w not in keep]
        for w in to_remove:
            del plate.wells[w]
        print(f"Measuring {len(plate.wells)} wells: {sorted(plate.wells.keys())}")

    # Filter wells if wells_to_test is specified in experiment config
    wells_to_test = cfg.get("wells", {}).get("wells_to_test")
    if wells_to_test:
        plate = context.deck["plate"]
        keep = {str(w).upper() for w in wells_to_test}
        to_remove = [w for w in plate.wells if w not in keep]
        for w in to_remove:
            del plate.wells[w]
        print(f"Measuring {len(plate.wells)} wells: {sorted(plate.wells.keys())}")

    # DataStore
    store = DataStore(db_path=str(_DB_PATH))
    context.data_store = store
    context.campaign_id = store.create_campaign(
        description='ASMI indentation run',
        deck_config=str(_DECK_YAML),
        board_config=str(_BOARD_YAML),
        gantry_config=str(_GANTRY_YAML),
        protocol_config=str(_PROTOCOL_YAML),
    )
    print(f"Campaign {context.campaign_id} created in {_DB_PATH}")

    # Connect instruments and run (GoDirect USB scan can take ~5–15 s)
    print("Connecting force sensor...")
    context.board.connect_instruments()
    print("Starting measurement.")
    aborted = False
    try:
        results = protocol.run(context)
        scan_results = results[0] if results else {}
        print(f"\n{len(scan_results)} wells measured")
        for well_id, result in scan_results.items():
            pts = result.get('data_points', 0)
            exceeded = result.get('force_exceeded', False)
            print(f"  {well_id}: {pts} points, force_exceeded={exceeded}")

        # Auto-analyze if workflow.generate_heatmap is True
        if scan_results and cfg.get('workflow', {}).get('generate_heatmap', False):
            sys.path.insert(0, str(_PROJECT_ROOT))
            from scripts.analyze_from_db import analyze_campaign
            print("\n--- Analysis ---")
            analyze_campaign(
                campaign_id=context.campaign_id,
                cfg=cfg,
                db_path=db_path,
                generate_plot=True,
            )
    except KeyboardInterrupt:
        aborted = True
        print("\nAborted by user.")
    finally:
        context.board.disconnect_instruments()
        if not mock:
            try:
                # Homing needs longer serial timeout (GRBL responds slowly when moving)
                gantry._mill.ser_mill.timeout = 5
                gantry.home()
            except Exception:
                pass
            gantry.disconnect()
        store.close()
        print(f"Data persisted to {_DB_PATH}")


if __name__ == "__main__":
    run(mock="--mock" in sys.argv, skip_home="--no-home" in sys.argv)
