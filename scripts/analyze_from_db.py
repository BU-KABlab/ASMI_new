#!/usr/bin/env python3
"""
Analyze ASMI indentation data from asmi_data.db.

Reads campaigns/experiments/asmi_measurements from the SQLite database,
runs Hertzian or linear fitting via IndentationAnalyzer, and optionally
generates plots.

Usage:
  python scripts/analyze_from_db.py
  python scripts/analyze_from_db.py --campaign 10
  python scripts/analyze_from_db.py --campaign 10 --well A2 --plot
"""

import argparse
import csv
import struct
import sys
from pathlib import Path

# Run from ASMI_new
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml
from src.analysis import IndentationAnalyzer
from src.plot import ASMIPlotter

# Cubos deck loader — used to derive `well_bottom_z` from the plate's
# `well_depth_mm` instead of carrying it as a manual analysis.yaml knob.
from deck.labware.well_plate import WellPlate
from deck.loader import load_deck_from_yaml

DB_PATH = ROOT / "data" / "asmi_data.db"
EXPERIMENT_YAML = ROOT / "configs" / "analysis.yaml"
DECK_YAML = ROOT / "configs" / "deck" / "asmi_deck.yaml"


def unpack_blob(blob: bytes) -> tuple[float, ...]:
    """Unpack little-endian doubles from BLOB."""
    if not blob:
        return ()
    n = len(blob) // 8
    return struct.unpack(f"<{n}d", blob)


def resolve_well_bottom_z(deck_path: Path) -> float | None:
    """Return the inside well-floor Z (deck-frame) for the first well plate.

    Computed as `a1.z - well_depth_mm`. Returns None if the deck has no
    well plate, or the plate doesn't declare `well_depth_mm` — caller
    falls back to the IndentationAnalyzer default in that case.
    """
    deck = load_deck_from_yaml(deck_path)
    for key in deck:
        labware = deck[key]
        if isinstance(labware, WellPlate) and labware.well_depth_mm is not None:
            return labware.get_well_center("A1").z - labware.well_depth_mm
    return None


def build_analyzer_data(
    z_positions: tuple[float, ...],
    raw_forces: tuple[float, ...],
    corrected_forces: tuple[float, ...],
    baseline_avg: float,
    baseline_std: float,
) -> list:
    """Build self.data format expected by IndentationAnalyzer."""
    rows = [
        ["Baseline_Force(N)", str(baseline_avg)],
        ["Baseline_Std(N)", str(baseline_std)],
    ]
    for i, (z, f_raw, f_corr) in enumerate(zip(z_positions, raw_forces, corrected_forces)):
        rows.append([str(i), str(z), str(f_raw), str(f_corr)])
    return rows


def analyze_campaign(
    campaign_id: int,
    cfg: dict,
    db_path: str | Path,
    generate_plot: bool = True,
    well_filter: str | None = None,
    deck_path: Path = DECK_YAML,
) -> list:
    """Analyze a campaign's measurements. Returns list of AnalysisResult."""
    import sqlite3

    analysis_cfg = cfg.get("analysis", {})

    # Compute well-floor Z from the plate definition (`a1.z - well_depth_mm`).
    # Falls back to None when the deck plate doesn't declare `well_depth_mm`,
    # in which case IndentationAnalyzer's parameter default is used.
    well_bottom_z = resolve_well_bottom_z(deck_path)
    if well_bottom_z is None:
        print(
            "Warning: deck plate doesn't declare `well_depth_mm`; "
            "falling back to IndentationAnalyzer default for well_bottom_z."
        )

    conn = sqlite3.connect(db_path)
    exp_q = "SELECT e.id, e.well_id FROM experiments e WHERE e.campaign_id = ?"
    exp_params = [campaign_id]
    if well_filter:
        exp_q += " AND e.well_id = ?"
        exp_params.append(well_filter.upper())
    exp_q += " ORDER BY e.well_id"
    exps = conn.execute(exp_q, exp_params).fetchall()

    results = []
    analyzer = None
    for exp_id, well_id in exps:
        rows = conn.execute(
            """
            SELECT z_positions, raw_forces, corrected_forces,
                   baseline_avg, baseline_std
            FROM asmi_measurements
            WHERE experiment_id = ?
            ORDER BY id DESC LIMIT 1
            """,
            (exp_id,),
        ).fetchall()

        if not rows:
            print(f"  {well_id}: no measurement")
            continue

        z_blob, raw_blob, corr_blob, b_avg, b_std = rows[0]
        z_arr = unpack_blob(z_blob)
        raw_arr = unpack_blob(raw_blob)
        corr_arr = unpack_blob(corr_blob)

        if len(z_arr) < 10:
            print(f"  {well_id}: too few points ({len(z_arr)})")
            continue

        # Build analyzer data and run analysis
        analyzer = IndentationAnalyzer(data_dir=str(ROOT))
        analyzer.data = build_analyzer_data(z_arr, raw_arr, corr_arr, b_avg, b_std)
        # Override system stiffness from config (for apply_system_correction)
        k_system = cfg.get("system", {}).get("k_system")
        if k_system is not None:
            analyzer.K_SYSTEM = float(k_system)

        result = analyzer.analyze_well(
            well=well_id,
            poisson_ratio=analysis_cfg.get("poisson_ratio", 0.5),
            contact_method=analysis_cfg.get("contact_method", "retrospective"),
            fit_method=analysis_cfg.get("fit_method", "hertzian"),
            apply_system_correction=analysis_cfg.get("apply_system_correction", False),
            apply_force_correction=analysis_cfg.get("apply_force_correction", False),
            iterative_d0_refinement=analysis_cfg.get("iterative_d0_refinement", False),
            retrospective_threshold=analysis_cfg.get("retrospective_threshold"),
            max_depth=analysis_cfg.get("max_depth", 0.5),
            min_depth=analysis_cfg.get("min_depth", 0.24),
            **({"well_bottom_z": well_bottom_z} if well_bottom_z is not None else {}),
            use_legacy_height=analysis_cfg.get("use_legacy_height", False),
            legacy_height_step_mm=analysis_cfg.get("legacy_height_step_mm", 0.02),
        )

        if result:
            results.append(result)
            E_mpa = result.elastic_modulus / 1e6
            print(f"  {well_id}: E = {E_mpa:.2f} MPa, R² = {result.fit_quality:.3f}")
        else:
            print(f"  {well_id}: analysis failed")

    conn.close()

    if generate_plot and results and analyzer:
        plots_base = cfg.get("paths", {}).get("plots_base", "results/plots")
        run_folder = f"campaign_{campaign_id}"
        run_folder_plots = ROOT / plots_base / run_folder
        run_folder_plots.mkdir(parents=True, exist_ok=True)

        plotter = ASMIPlotter()
        plotter._analyzer = analyzer
        for r in results:
            plotter.plot_results(r, run_folder=run_folder)
        print(f"\n📊 Plots saved to {plots_base}/{run_folder}/")

        # Generate heatmap if workflow.generate_heatmap and we have results
        if cfg.get("workflow", {}).get("generate_heatmap", False) and results:
            summary_csv = run_folder_plots / "summary.csv"
            with open(summary_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["Well", "ElasticModulus", "R2", "Std", "ElasticModulus_Original"],
                    extrasaction="ignore",
                )
                writer.writeheader()
                for r in results:
                    row = {
                        "Well": r.well,
                        "ElasticModulus": r.elastic_modulus,
                        "R2": r.fit_quality,
                        "Std": r.uncertainty,
                    }
                    if getattr(r, "original_elastic_modulus", None) is not None:
                        row["ElasticModulus_Original"] = r.original_elastic_modulus
                    writer.writerow(row)
            heatmap_path = run_folder_plots / "heatmap.png"
            plotter.plot_well_heatmap(str(summary_csv), save_path=str(heatmap_path))
            print(f"📊 Heatmap saved to {heatmap_path}")

    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--campaign", type=int, help="Campaign ID (default: latest)")
    ap.add_argument("--well", type=str, help="Filter by well (e.g. A2)")
    ap.add_argument("--plot", action="store_true", help="Generate analysis plots")
    args = ap.parse_args()

    with open(EXPERIMENT_YAML) as f:
        cfg = yaml.safe_load(f)

    import sqlite3
    conn = sqlite3.connect(DB_PATH)
    q = "SELECT id, description FROM campaigns"
    params = []
    if args.campaign:
        q += " WHERE id = ?"
        params.append(args.campaign)
    q += " ORDER BY id DESC LIMIT 1"
    row = conn.execute(q, params).fetchone()
    conn.close()
    if not row:
        print("No campaigns found.")
        return
    cid, desc = row
    print(f"\n=== Campaign {cid}: {desc} ===\n")

    analyze_campaign(
        campaign_id=cid,
        cfg=cfg,
        db_path=DB_PATH,
        generate_plot=args.plot,
        well_filter=args.well,
    )


if __name__ == "__main__":
    main()
