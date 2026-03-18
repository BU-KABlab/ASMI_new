import sys
import os

# Run from ASMI_new so relative imports (src/, results/) resolve correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import time
from datetime import datetime
from sqlalchemy import select, update, insert, desc
from sql_interface.db import engine
from sql_interface.models import asmi as asmi_table, arm_tasks
from sql_interface.store import store_run_to_db

from main_asmi import run_measure_analyze_plot, ensure_run_folder
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor


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


def run_asmi_task(conn, task, cnc: CNCController, force_sensor: ForceSensor):
    job_id = task["job_id"]
    substrate_id = task["substrate_id"]
    well_id = task["well_id"]

    # Interlock: most recent completed arm task for this substrate must end at "asmi"
    location_check = (
        select(arm_tasks)
        .where(arm_tasks.c.substrate_id == substrate_id, arm_tasks.c.time_completed != None)
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

    # Home CNC before each measurement
    try:
        cnc.unlock()
        cnc.home(zero_after=True)
    except Exception as e:
        print(f"WARNING: Pre-measurement home failed: {e}")

    # Run measurement and analysis
    run_folder = ensure_run_folder()
    try:
        per_well_results, run_folder_name = run_measure_analyze_plot(
            cnc=cnc,
            force_sensor=force_sensor,
            well=well_id.upper(),
            contact_method="retrospective",
            measure_with_return=False,
            z_target=-90.0,
            step_size=0.01,
            force_limit=5.0,
            well_top_z=-73.0,
            well_bottom_z=-84.2,
            run_folder=run_folder,
            fit_method="hertzian",
            apply_system_correction=True,
            max_depth=0.5, 
            min_depth=0.24, 
            poisson_ratio=0.5,
            apply_force_correction=True, # Apply geometry correction (F/(c*d^b)) before Hertzian fit
            iterative_d0_refinement=True,
            k_system_override=64.27,
            remeasure_if_first_force_above=0.05, # if the absolute value of the first measurement force is greater than 0.05 N, re-measure the well with a well_top_z_remargin_offset of 0.3 mm
            well_top_z_remargin_offset=0.3, # offset to raise the well_top_z when re-measuring (e.g., -72.8 -> -72.5)
        )
        success = bool(per_well_results)
    except Exception as e:
        print(f"ERROR during ASMI measurement for task {job_id}: {e}")
        per_well_results = None
        success = False

    # Extract primary result values
    elastic_modulus = None
    contact_z = None
    if success and per_well_results:
        r = per_well_results[0]
        elastic_modulus = getattr(r, "elastic_modulus", None)
        # Try common attribute names for contact z depth
        contact_z = (
            getattr(r, "contact_z", None)
            or getattr(r, "d0", None)
            or getattr(r, "contact_point", None)
        )

    # Write result back to DB
    outcome = "success" if success else "failed"
    with engine.begin() as complete_conn:
        complete_conn.execute(
            update(asmi_table)
            .where(asmi_table.c.job_id == job_id)
            .values(
                time_completed=datetime.now(),
                status=outcome,
                force=elastic_modulus,          # elastic modulus (Pa or kPa per fit)
                **{"z-position": contact_z},    # contact depth (mm)
            )
        )
    print(f"{'✅' if success else '❌'} Task {job_id} {outcome} | E={elastic_modulus} | z={contact_z}")

    # Store measurement and plot artifacts to DB
    try:
        run_folder_name = os.path.basename(run_folder)
        stored_run_id = store_run_to_db(
            run_folder_name=run_folder_name,
            substrate_id=substrate_id,
            asmi_job_id=job_id,
            store_csv_content=True,
            store_plot_blobs=False,
        )
        if stored_run_id:
            print(f"📦 Stored run {run_folder_name} to DB (run_id={stored_run_id})")
    except Exception as e:
        print(f"⚠️ Failed to store run to DB: {e}")

    # If all wells for this substrate are done, home CNC and enqueue return arm task
    with engine.begin() as check_conn:
        remaining = check_conn.execute(
            select(asmi_table).where(
                asmi_table.c.substrate_id == substrate_id,
                asmi_table.c.time_completed == None,
            )
        ).fetchall()

    if not remaining:
        print(f"All ASMI tasks complete for substrate {substrate_id}. Homing CNC and enqueuing return arm task.")
        try:
            cnc.home(zero_after=True)
        except Exception as e:
            print(f"WARNING: Post-run home failed: {e}")
        with engine.begin() as insert_conn:
            insert_conn.execute(
                insert(arm_tasks).values(
                    substrate_id=substrate_id,
                    from_location="asmi",
                    to_location="sharc-uv",
                    manual=False,
                )
            )
        print(f"Enqueued arm_task: {substrate_id} asmi -> sharc-uv")


def poll_asmi_tasks():
    print("Starting ASMI task polling loop...")

    try:
        cnc = CNCController()
        cnc.unlock()
        cnc.home(zero_after=True)
    except Exception as e:
        print(f"ERROR: Could not initialize CNC: {e}")
        return

    try:
        force_sensor = ForceSensor()
    except Exception as e:
        print(f"ERROR: Could not initialize ForceSensor: {e}")
        return

    while True:
        with engine.connect() as conn:
            tasks = get_pending_asmi_tasks(conn)
            if not tasks:
                print("No available ASMI tasks.")
            for task in tasks:
                run_asmi_task(conn, task, cnc, force_sensor)
        time.sleep(10)


if __name__ == "__main__":
    poll_asmi_tasks()
