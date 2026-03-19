import datetime
from enum import Enum as PyEnum
from sqlalchemy import (
    MetaData, Table, Column,
    String, Integer, Boolean, Float,
    DateTime, Enum, Text, ForeignKey
)
from sqlalchemy.dialects.mysql import LONGTEXT, LONGBLOB

metadata = MetaData()

class ActionType(PyEnum):
    added = "added"
    removed = "removed"

heatershaker = Table(
    "heatershaker", metadata,
    Column("job_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255)),
    Column("use_heater_shaker", Boolean, nullable=False),
    Column("temperature", Float),
    Column("heating_time", Integer),
    Column("shaking_rpm", Integer),
    Column("shaking_time", Integer),
    Column("status", String(255)),
    Column("time_started", DateTime),
    Column("time_completed", DateTime),
)

sharc = Table(
    "sharc", metadata,
    Column("job_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255), nullable=False),
    Column("well_id", String(255), nullable=False),
    Column("uv_intensity", Float),
    Column("exposure_time", Integer),
    Column("z", Float, default=-16.7, nullable=False),
    Column("status", String(255)),
    Column("time_started", DateTime),
    Column("time_completed", DateTime),
)

arm_tasks = Table(
    "arm_tasks", metadata,
    Column("job_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255), nullable=False),
    Column("from_location", String(255), nullable=False),
    Column("to_location", String(255), nullable=False),
    Column("status", String(255)),
    Column("time_started", DateTime),
    Column("time_completed", DateTime),
    Column("manual", Boolean, default=False, nullable=False, quote=True),
)

asmi = Table(
    "asmi", metadata,
    Column("job_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255), nullable=False),
    Column("well_id", String(255), nullable=False),
    Column("force", Float, nullable=True),
    Column("z-position", Float, nullable=True),
    Column("status", String(255)),
    Column("time_started", DateTime),
    Column("time_completed", DateTime),
)

storage_start_log = Table(
    "storage_start", metadata,
    Column("log_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255), nullable=False),
    Column("action", Enum(ActionType), nullable=False),
    Column("stack_position", Integer, nullable=False),
    Column("manual", Boolean, default=False, nullable=False, quote=True),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow, nullable=False)
)

storage_end_log = Table(
    "storage_end", metadata,
    Column("log_id", Integer, primary_key=True, autoincrement=True),
    Column("substrate_id", String(255), nullable=False),
    Column("action", Enum(ActionType), nullable=False),
    Column("stack_position", Integer, nullable=False),
    Column("manual", Boolean, default=False, nullable=False, quote=True),
    Column("timestamp", DateTime, default=datetime.datetime.utcnow, nullable=False)
)

# --- ASMI measurement and plot storage (links results/measurements/ and results/plots/ to DB) ---

asmi_runs = Table(
    "asmi_runs", metadata,
    Column("run_id", Integer, primary_key=True, autoincrement=True),
    Column("run_folder_name", String(255), nullable=False, unique=True),
    Column("run_count", Integer, nullable=True),
    Column("created_at", DateTime, nullable=False),
    Column("substrate_id", String(255), nullable=True),
    Column("asmi_job_id", Integer, ForeignKey("asmi.job_id"), nullable=True),
)

asmi_measurement_files = Table(
    "asmi_measurement_files", metadata,
    Column("measurement_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("asmi_runs.run_id"), nullable=False),
    Column("well_id", String(16), nullable=False),
    Column("asmi_job_id", Integer, ForeignKey("asmi.job_id"), nullable=True),
    Column("file_path", String(512), nullable=False),
    Column("csv_content", LONGTEXT, nullable=True),
    Column("elastic_modulus", Float, nullable=True),
    Column("contact_z", Float, nullable=True),
    Column("created_at", DateTime, nullable=True),
)

asmi_plot_files = Table(
    "asmi_plot_files", metadata,
    Column("plot_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("asmi_runs.run_id"), nullable=False),
    Column("well_id", String(16), nullable=True),
    Column("asmi_job_id", Integer, ForeignKey("asmi.job_id"), nullable=True),
    Column("plot_type", String(64), nullable=False),
    Column("file_path", String(512), nullable=False),
    Column("file_content", LONGBLOB, nullable=True),
    Column("created_at", DateTime, nullable=True),
)

asmi_well_summary = Table(
    "asmi_well_summary", metadata,
    Column("summary_id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", Integer, ForeignKey("asmi_runs.run_id"), nullable=False),
    Column("well_id", String(16), nullable=False),
    Column("elastic_modulus", Float, nullable=True),
    Column("elastic_modulus_original", Float, nullable=True),
    Column("std", Float, nullable=True),
    Column("r2", Float, nullable=True),
    Column("contact_z", Float, nullable=True),
    Column("created_at", DateTime, nullable=True),
)
