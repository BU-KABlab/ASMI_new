"""
Store ASMI measurement and plot artifacts into the database.

Scans results/measurements/<run_folder>/ and results/plots/<run_folder>/
and inserts structured records into asmi_runs, asmi_measurement_files,
asmi_plot_files, and asmi_well_summary.
"""

import os
import csv
import re
from datetime import datetime
from sqlalchemy import select, insert, delete

from .db import engine
from .models import asmi_runs, asmi_measurement_files, asmi_plot_files, asmi_well_summary


MEASUREMENTS_BASE = "results/measurements"
PLOTS_BASE = "results/plots"


def _parse_run_folder(run_folder_name: str) -> tuple[int | None, datetime | None]:
    """Extract run_count and datetime from run_XXX_YYYYMMDD_HHMMSS."""
    m = re.match(r"run_(\d+)_(\d{8})_(\d{6})", run_folder_name)
    if not m:
        return None, None
    run_count = int(m.group(1))
    try:
        dt = datetime.strptime(f"{m.group(2)}_{m.group(3)}", "%Y%m%d_%H%M%S")
    except ValueError:
        dt = None
    return run_count, dt


def _parse_well_from_filename(filename: str) -> str | None:
    """Extract well ID from well_<WELL>_*.csv or <WELL>_*.png."""
    m = re.match(r"well_([A-H]\d{1,2})", filename, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.match(r"([A-H]\d{1,2})_", filename)
    if m:
        return m.group(1).upper()
    return None


def _infer_plot_type(filename: str) -> str:
    """Infer plot_type from filename."""
    name = filename.lower()
    if "raw_force" in name or "raw_data" in name:
        return "raw_force"
    if "contact_detection" in name:
        return "contact_detection"
    if "analysis" in name:
        return "analysis"
    if "summary.txt" in name:
        return "summary_txt"
    if "summary.csv" in name:
        return "summary_csv"
    if "heatmap" in name:
        return "heatmap"
    return "other"


def store_run_to_db(
    run_folder_name: str,
    substrate_id: str | None = None,
    asmi_job_id: int | None = None,
    store_csv_content: bool = True,
    store_plot_blobs: bool = False,
    base_dir: str | None = None,
) -> int | None:
    """
    Scan a run folder and store all measurement CSVs and plot files into the DB.

    Args:
        run_folder_name: e.g. "run_738_20260228_083050"
        substrate_id: Optional substrate ID (from asmi task)
        asmi_job_id: Optional asmi job_id (from asmi task)
        store_csv_content: If True, store full CSV text in asmi_measurement_files
        store_plot_blobs: If True, store PNG/binary content; else only file_path
        base_dir: Project root (default: cwd)

    Returns:
        run_id if successful, None otherwise.
    """
    base = base_dir or os.getcwd()
    meas_path = os.path.join(base, MEASUREMENTS_BASE, run_folder_name)
    plots_path = os.path.join(base, PLOTS_BASE, run_folder_name)

    if not os.path.isdir(meas_path) and not os.path.isdir(plots_path):
        return None

    run_count, created_dt = _parse_run_folder(run_folder_name)
    created_at = created_dt or datetime.now()

    with engine.begin() as conn:
        # Insert or get run
        existing = conn.execute(
            select(asmi_runs).where(asmi_runs.c.run_folder_name == run_folder_name)
        ).fetchone()
        if existing:
            run_id = existing[0]
            # Clear existing files so we can re-insert (idempotent refresh)
            conn.execute(delete(asmi_measurement_files).where(asmi_measurement_files.c.run_id == run_id))
            conn.execute(delete(asmi_plot_files).where(asmi_plot_files.c.run_id == run_id))
            conn.execute(delete(asmi_well_summary).where(asmi_well_summary.c.run_id == run_id))
        else:
            result = conn.execute(
                insert(asmi_runs).values(
                    run_folder_name=run_folder_name,
                    run_count=run_count,
                    created_at=created_at,
                    substrate_id=substrate_id,
                    asmi_job_id=asmi_job_id,
                )
            )
            run_id = result.inserted_primary_key[0]

        # Measurement CSVs
        if os.path.isdir(meas_path):
            for fname in sorted(os.listdir(meas_path)):
                if not fname.endswith(".csv") or not fname.startswith("well_"):
                    continue
                well_id = _parse_well_from_filename(fname)
                if not well_id:
                    continue
                file_path = os.path.join(MEASUREMENTS_BASE, run_folder_name, fname)
                full_path = os.path.join(base, file_path)
                csv_content = None
                elastic_modulus = None
                contact_z = None
                test_time = None
                if store_csv_content and os.path.isfile(full_path):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            csv_content = f.read()
                        # Parse metadata for summary
                        with open(full_path, "r", encoding="utf-8") as f:
                            reader = csv.reader(f)
                            for row in reader:
                                if len(row) >= 2 and row[0] == "Test_Time":
                                    test_time = row[1] if len(row) > 1 else None
                                    break
                    except Exception:
                        pass

                conn.execute(
                    insert(asmi_measurement_files).values(
                        run_id=run_id,
                        well_id=well_id,
                        asmi_job_id=asmi_job_id,
                        file_path=file_path,
                        csv_content=csv_content,
                        elastic_modulus=elastic_modulus,
                        contact_z=contact_z,
                        created_at=created_at,
                    )
                )

        # Plot files
        if os.path.isdir(plots_path):
            for fname in sorted(os.listdir(plots_path)):
                full_path = os.path.join(plots_path, fname)
                if not os.path.isfile(full_path):
                    continue
                well_id = _parse_well_from_filename(fname)
                plot_type = _infer_plot_type(fname)
                file_path = os.path.join(PLOTS_BASE, run_folder_name, fname)
                file_content = None
                if store_plot_blobs:
                    try:
                        with open(full_path, "rb") as f:
                            file_content = f.read()
                    except Exception:
                        pass
                elif plot_type in ("summary_txt", "summary_csv"):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            file_content = f.read().encode("utf-8")
                    except Exception:
                        pass

                conn.execute(
                    insert(asmi_plot_files).values(
                        run_id=run_id,
                        well_id=well_id,
                        asmi_job_id=asmi_job_id,
                        plot_type=plot_type,
                        file_path=file_path,
                        file_content=file_content,
                        created_at=created_at,
                    )
                )

        # Parse summary.csv into asmi_well_summary for easy SQL querying
        summary_path = os.path.join(plots_path, "summary.csv")
        if os.path.isfile(summary_path):
            try:
                with open(summary_path, "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        well = row.get("Well", "").strip().upper()
                        if not well:
                            continue
                        elastic_modulus = None
                        elastic_modulus_original = None
                        std = None
                        r2 = None
                        for k, v in row.items():
                            if not v or k == "Well":
                                continue
                            try:
                                val = float(v)
                            except (ValueError, TypeError):
                                continue
                            k_lower = k.lower().replace(" ", "").replace("_", "")
                            if "elasticmodulus" in k_lower and "original" not in k_lower:
                                elastic_modulus = val
                            elif "elasticmodulus" in k_lower and "original" in k_lower:
                                elastic_modulus_original = val
                            elif "std" in k_lower or "uncertainty" in k_lower:
                                std = val
                            elif "r2" in k_lower:
                                r2 = val
                        conn.execute(
                            insert(asmi_well_summary).values(
                                run_id=run_id,
                                well_id=well,
                                elastic_modulus=elastic_modulus,
                                elastic_modulus_original=elastic_modulus_original,
                                std=std,
                                r2=r2,
                                created_at=created_at,
                            )
                        )
            except Exception:
                pass

    return run_id
