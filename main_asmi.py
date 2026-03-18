#!/usr/bin/env python3
"""
ASMI runner — uses PANDA_CORE protocol engine for measurement and SQL persistence.

Usage:
    python3 main_asmi.py          # real hardware
    python3 main_asmi.py --mock   # offline dry run with mock instruments

Configuration lives in YAML files under configs/:
    gantry/asmi_gantry.yaml          — CNC serial port, working volume, GRBL
    deck/asmi_deck.yaml              — well plate geometry and calibration
    board/asmi_board.yaml            — ASMI instrument with indentation params
    protocol/asmi_indentation.yaml   — scan protocol (method: indentation)
"""

from __future__ import annotations

import os
import sys
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
_DB_PATH = _PROJECT_ROOT / 'data' / 'asmi_data.db'


def run(mock: bool = False) -> None:
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
        gantry.unlock()
        gantry.home()

    # Load protocol (mock_mode swaps asmi -> mock_asmi in board loader)
    protocol, context = setup_protocol(
        str(_GANTRY_YAML), str(_DECK_YAML), str(_BOARD_YAML),
        str(_PROTOCOL_YAML), gantry=gantry, mock_mode=mock,
    )

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

    # Connect instruments and run
    context.board.connect_instruments()
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
                gantry.home()
            except Exception:
                pass
            gantry.disconnect()
        store.close()
        print(f"Data persisted to {_DB_PATH}")


if __name__ == "__main__":
    run(mock="--mock" in sys.argv)
