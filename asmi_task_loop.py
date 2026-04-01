"""
ASMI automation queue worker — uses the PANDA_CORE entry (main_asmi.run).

Polls the lab SQL DB for pending rows in ``asmi``, runs one well per task via
``main_asmi.run``, analyzes the new campaign with ``scripts.analyze_from_db``,
writes modulus / contact back to the automation DB, and enqueues arm return.

Requires: sql_interface DB config, hardware reachable like ``python main_asmi.py``.

Environment:
  ASMI_SKIP_FORCE_SENSOR=1  — same as ``main_asmi.py --skip-force-sensor``
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Run from ASMI_new so imports resolve
_ROOT = Path(__file__).resolve().parent
os.chdir(_ROOT)
sys.path.insert(0, str(_ROOT))

import yaml
from sqlalchemy import select, update, insert, desc

from sql_interface.db import engine
from sql_interface.models import asmi as asmi_table, arm_tasks
from main_asmi import run as run_asmi
from scripts.analyze_from_db import analyze_campaign

_EXPERIMENT_YAML = _ROOT / "configs" / "experiment.yaml"


def get_pending_asmi_tasks(conn):
    """Return pending tasks only if no task is currently running."""
    running_check = select(asmi_table).where(
        asmi_table.c.time_started != None,
        asmi_table.c.time_completed == None,
    )
    if conn.execute(running_check).fetchone():
        print("ASMI already running a task. Waiting...")
        return []

    stmt = select(asmi_table).where(asmi_table.c.time_started == None)
    return conn.execute(stmt).mappings().fetchall()


def run_asmi_task(conn, task):
    job_id = task["job_id"]
    substrate_id = task["substrate_id"]
    well_id = task["well_id"]

    # Interlock: most recent completed arm task for this substrate must end at "asmi"
    location_check = (
        select(arm_tasks)
        .where(arm_tasks.c.substrate_id == substrate_id, arm_tasks.c.time_completed != None, arm_tasks.c.manual == False)
        .order_by(desc(arm_tasks.c.time_completed))
        .limit(1)
    )
    last_move = conn.execute(location_check).mappings().fetchone()
    if not last_move or last_move["to_location"] != "asmi":
        print(
            f"ERROR: Interlock failed: substrate {substrate_id} last destination is "
            f"{last_move['to_location'] if last_move else None}, not asmi. Skipping task {job_id}."
        )
        return

    # Claim the task
    with engine.begin() as claim_conn:
        claim_conn.execute(
            update(asmi_table)
            .where(asmi_table.c.job_id == job_id)
            .values(time_started=datetime.now())
        )
    print(f"\nStarting ASMI Task {job_id} | Substrate: {substrate_id} | Well: {well_id}")

    skip_force = os.environ.get("ASMI_SKIP_FORCE_SENSOR", "").strip() in ("1", "true", "yes")

    with open(_EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)
    db_path = cfg.get("paths", {}).get("database", "data/asmi_data.db")
    db_path = _ROOT / db_path if not os.path.isabs(db_path) else Path(db_path)

    summary = None
    try:
        summary = run_asmi(
            mock=False,
            skip_home=False,
            skip_force_sensor=skip_force,
            wells_to_test=[well_id.upper()],
            return_summary=True,
        )
    except Exception as e:
        print(f"ERROR during ASMI measurement for task {job_id}: {e}")
        summary = None

    success = bool(
        summary
        and summary.get("scan_results")
        and any(
            (r or {}).get("data_points", 0) > 0
            for r in summary["scan_results"].values()
        )
    )

    elastic_modulus = None
    contact_z = None
    if success and summary:
        try:
            analysis_results = analyze_campaign(
                int(summary["campaign_id"]),
                cfg,
                db_path,
                generate_plot=cfg.get("workflow", {}).get("generate_heatmap", False),
                well_filter=well_id.upper(),
            )
            if analysis_results:
                r0 = analysis_results[0]
                elastic_modulus = getattr(r0, "elastic_modulus", None)
                contact_z = getattr(r0, "contact_z", None)
        except Exception as e:
            print(f"WARNING: Analysis after measurement failed: {e}")

    # Write result back to automation DB (force column = E as before)
    outcome = "success" if success else "failed"
    with engine.begin() as complete_conn:
        complete_conn.execute(
            update(asmi_table)
            .where(asmi_table.c.job_id == job_id)
            .values(
                time_completed=datetime.now(),
                status=outcome,
                force=elastic_modulus,
                **{"z-position": contact_z},
            )
        )
    print(f"{'✅' if success else '❌'} Task {job_id} {outcome} | E={elastic_modulus} | z={contact_z}")

    # Measurement lives in PANDA SQLite (paths.database). Legacy ``store_run_to_db`` expects
    # ``results/measurements/run_*`` folders; if you still use that layout, call it here.

    # If all wells for this substrate are done, home via a one-off run is already done;
    # enqueue return arm task
    with engine.begin() as check_conn:
        remaining = check_conn.execute(
            select(asmi_table).where(
                asmi_table.c.substrate_id == substrate_id,
                asmi_table.c.time_completed == None,
            )
        ).fetchall()

    if not remaining:
        print(f"All ASMI tasks complete for substrate {substrate_id}. Enqueuing return arm task.")
        next_from = task.get("next_arm_from")
        next_to = task.get("next_arm_to")
        if next_from and next_to:
            with engine.begin() as insert_conn:
                insert_conn.execute(
                    insert(arm_tasks).values(
                        substrate_id=substrate_id,
                        from_location=next_from,
                        to_location=next_to,
                        manual=False,
                    )
                )
            print(f"Enqueued arm_task: {substrate_id} {next_from} -> {next_to}")
        else:
            print(f"No next_arm routing set for task {job_id}, skipping arm enqueue.")


def poll_asmi_tasks():
    print("Starting ASMI task polling loop (PANDA_CORE main_asmi.run)...")
    print("  Set ASMI_SKIP_FORCE_SENSOR=1 to use mock force sensor if needed.")

    while True:
        try:
            with engine.connect() as conn:
                tasks = get_pending_asmi_tasks(conn)
                if not tasks:
                    print("No available ASMI tasks.")
                for task in tasks:
                    run_asmi_task(conn, task)
        except Exception as e:
            print(f"Polling error: {e}")
        time.sleep(10)


if __name__ == "__main__":
    poll_asmi_tasks()
