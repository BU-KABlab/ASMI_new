#!/usr/bin/env python3
"""
Backfill existing measurement/plot runs into the database.

Usage:
  cd ASMI_new && python scripts/backfill_runs_to_db.py
  cd ASMI_new && python scripts/backfill_runs_to_db.py run_732_20251030_122001 run_738_20260228_083050
"""

import os
import sys

# Run from ASMI_new
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.getcwd()))

from sql_interface.store import store_run_to_db

MEASUREMENTS_BASE = "results/measurements"


def main():
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        if not os.path.isdir(MEASUREMENTS_BASE):
            print(f"❌ {MEASUREMENTS_BASE} not found")
            return
        folders = sorted(
            d for d in os.listdir(MEASUREMENTS_BASE)
            if os.path.isdir(os.path.join(MEASUREMENTS_BASE, d)) and d.startswith("run_")
        )
    for run_folder_name in folders:
        try:
            rid = store_run_to_db(
                run_folder_name=run_folder_name,
                store_csv_content=True,
                store_plot_blobs=False,
            )
            if rid:
                print(f"✅ {run_folder_name} -> run_id={rid}")
            else:
                print(f"⚠️ {run_folder_name} skipped (no data)")
        except Exception as e:
            print(f"❌ {run_folder_name}: {e}")


if __name__ == "__main__":
    main()
