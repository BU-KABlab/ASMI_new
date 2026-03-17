#!/usr/bin/env python3
"""
ASMI runner — uses PANDA_CORE protocol engine for measurement and SQL persistence.

Usage:
    python3 main_asmi.py          # real hardware, configs from configs/
    python3 main_asmi.py --mock   # offline dry run with mock instruments

All configuration lives in YAML files under configs/:
    gantry/asmi_gantry.yaml   — CNC serial port, working volume, GRBL settings
    deck/asmi_deck.yaml       — well plate geometry and calibration
    board/asmi_board.yaml     — ASMI instrument with indentation parameters
    protocol/asmi_indentation.yaml — scan protocol (method: indentation)
    experiment.yaml           — analysis params, paths, workflow flags
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import yaml

# ── PANDA_CORE path setup ────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent
_PANDA_CORE_ROOT = _PROJECT_ROOT / '..' / 'PANDA_CORE'
_PANDA_CORE_SRC = _PANDA_CORE_ROOT / 'src'
if _PANDA_CORE_SRC.is_dir() and str(_PANDA_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(_PANDA_CORE_SRC))
if _PANDA_CORE_ROOT.is_dir() and str(_PANDA_CORE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PANDA_CORE_ROOT))

from gantry.gantry import Gantry
from gantry.offline import OfflineGantry
from protocol_engine.setup import setup_protocol
from data.data_store import DataStore

# ── Config paths ──────────────────────────────────────────────────────────
_CONFIGS = _PROJECT_ROOT / 'configs'
_GANTRY_YAML = _CONFIGS / 'gantry' / 'asmi_gantry.yaml'
_DECK_YAML = _CONFIGS / 'deck' / 'asmi_deck.yaml'
_BOARD_YAML = _CONFIGS / 'board' / 'asmi_board.yaml'
_MOCK_BOARD_YAML = _CONFIGS / 'board' / 'mock_asmi_board.yaml'
_PROTOCOL_YAML = _CONFIGS / 'protocol' / 'asmi_indentation.yaml'
_EXPERIMENT_YAML = _CONFIGS / 'experiment.yaml'


def run(mock: bool = False) -> None:
    """Run the ASMI indentation protocol with SQL persistence.

    Args:
        mock: If True, use OfflineGantry + MockASMI (no hardware).
    """
    with open(_EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)

    paths = cfg.get('paths', {})
    db_path = paths.get('database', 'data/asmi_data.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    board_yaml = _MOCK_BOARD_YAML if mock else _BOARD_YAML

    # Build gantry
    if mock:
        gantry = OfflineGantry()
    else:
        with open(_GANTRY_YAML) as f:
            gantry_config = yaml.safe_load(f)
        gantry = Gantry(config=gantry_config)
        gantry.connect()
        gantry.unlock()
        if cfg.get('workflow', {}).get('home_before_measure', True):
            gantry.home()

    # Load protocol
    protocol, context = setup_protocol(
        str(_GANTRY_YAML), str(_DECK_YAML), str(board_yaml),
        str(_PROTOCOL_YAML), gantry=gantry,
    )

    # DataStore
    store = DataStore(db_path=db_path)
    context.data_store = store
    context.campaign_id = store.create_campaign(
        description=cfg.get('description', 'ASMI indentation run'),
        deck_config=str(_DECK_YAML),
        board_config=str(board_yaml),
        gantry_config=str(_GANTRY_YAML),
        protocol_config=str(_PROTOCOL_YAML),
    )
    print(f"Campaign {context.campaign_id} created in {db_path}")

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
        print(f"Data persisted to {db_path}")


if __name__ == "__main__":
    run(mock="--mock" in sys.argv)
