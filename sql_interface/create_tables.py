#!/usr/bin/env python3
"""Create ASMI storage tables (asmi_runs, asmi_measurement_files, asmi_plot_files, asmi_well_summary)."""

from .db import engine
from . import models  # ensure new tables are registered in metadata

if __name__ == "__main__":
    models.metadata.create_all(engine)
    print("✅ Tables created (or already exist).")
