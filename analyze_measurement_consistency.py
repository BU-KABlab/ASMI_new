#!/usr/bin/env python3
"""
Analyze consistency of Elastic Modulus across multiple measurement runs.

Reads *_summary.txt from each run folder under results/plots/, extracts Elastic Modulus
for each well, and computes consistency statistics (mean, std, CV%) across runs.
Skips wells that did not fit successfully (no Elastic Modulus in summary).

Usage:
    python analyze_measurement_consistency.py run_747_... run_748_... run_749_...
    # Or edit FOLDER_NAMES below and run without args
"""

import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # use Agg backend for matplotlib to avoid X11 window system

from src.plot import ASMIPlotter

# Base path for plot folders (relative to script directory)
PLOTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")

# Edit this list or pass folder names as command-line arguments
FOLDER_NAMES = [
    "run_853_20260227_235201",
    "run_854_20260228_010655",
    "run_855_20260228_022140",
    "run_856_20260228_033521",
    "run_857_20260228_044905",
    "run_858_20260228_060302",
    "run_859_20260228_071718",
    "run_860_20260228_083050",
    "run_861_20260228_094437"
]

# Regex patterns to extract the three Elastic Modulus values from summary
# Support both hyphen (-) and en-dash (–) in "0–max" and "min–max"
RE_SYSTEM_CORRECTED = re.compile(r"Elastic Modulus \(system corrected\):\s*([\d.]+)\s*Pa", re.IGNORECASE)
RE_0_MAX = re.compile(r"Elastic Modulus \(0[-–]max_depth, no force correction\):\s*([\d.]+)\s*Pa", re.IGNORECASE)
RE_MIN_MAX_FC = re.compile(r"Elastic Modulus \(min[-–]max_depth, with force correction\):\s*([\d.]+)\s*Pa", re.IGNORECASE)
# Fallback for legacy format: "Elastic Modulus: 899074 Pa"
RE_LEGACY = re.compile(r"Elastic Modulus:\s*([\d.]+)\s*Pa", re.IGNORECASE)


def parse_summary_file(filepath: str) -> dict[str, float] | None:
    """
    Parse all three Elastic Modulus values from a well summary file.
    Returns {"system_corrected": E, "0_max": E, "min_max_fc": E} or None if no valid data.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        out = {}
        for key, pat in [
            ("system_corrected", RE_SYSTEM_CORRECTED),
            ("0_max", RE_0_MAX),
            ("min_max_fc", RE_MIN_MAX_FC),
        ]:
            m = pat.search(content)
            if m:
                out[key] = float(m.group(1))
        if not out:
            m = RE_LEGACY.search(content)
            if m:
                out["system_corrected"] = float(m.group(1))
        return out if out else None
    except (OSError, ValueError):
        pass
    return None


def well_sort_key(well: str) -> tuple:
    """Sort wells as A1, A2, ... A12, B1, ... H12."""
    if not well or len(well) < 2:
        return (999, 999)
    row = well[0].upper()
    try:
        col = int(well[1:].replace("_down", "").replace("_up", ""))
    except ValueError:
        col = 0
    return (ord(row) - ord("A"), col)


E_TYPES = ("system_corrected", "0_max", "min_max_fc")
E_TYPE_LABELS = {
    "system_corrected": "System Corrected",
    "0_max": "0–max (no FC)",
    "min_max_fc": "min–max (with FC)",
}


def collect_elastic_moduli(folder_names: list[str]) -> dict[str, dict[str, list[tuple[str, float]]]]:
    """
    Collect Elastic Modulus for each well and each E type across all run folders.
    Returns: {well: {E_type: [(run_folder, E_Pa), ...]}}
    """
    data: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for folder in folder_names:
        folder_path = os.path.join(PLOTS_BASE, folder)
        if not os.path.isdir(folder_path):
            print(f"⚠️ Folder not found: {folder_path}")
            continue
        for fname in os.listdir(folder_path):
            if not fname.endswith("_summary.txt"):
                continue
            well = fname.replace("_summary.txt", "")
            filepath = os.path.join(folder_path, fname)
            parsed = parse_summary_file(filepath)
            if parsed is not None:
                for e_type, E in parsed.items():
                    data[well][e_type].append((folder, E))
    return dict(data)


def analyze_consistency_single(values: list[float]) -> dict | None:
    """Compute consistency stats for a list of values. Returns None if < 2 values."""
    if len(values) < 2:
        return None
    n = len(values)
    mean_e = sum(values) / n
    variance = sum((x - mean_e) ** 2 for x in values) / (n - 1) if n > 1 else 0
    std_e = variance ** 0.5
    cv_pct = (std_e / mean_e * 100) if mean_e > 0 else 0
    return {
        "n": n,
        "mean_Pa": mean_e,
        "std_Pa": std_e,
        "cv_pct": cv_pct,
        "min_Pa": min(values),
        "max_Pa": max(values),
    }


def analyze_consistency(data: dict[str, dict[str, list[tuple[str, float]]]]) -> dict[str, list[dict]]:
    """
    Compute consistency stats for each well and each E type.
    Returns: {E_type: [{"well", "n", "mean_Pa", "std_Pa", "cv_pct", "min_Pa", "max_Pa"}, ...]}
    """
    result: dict[str, list[dict]] = {et: [] for et in E_TYPES}
    for well in sorted(data.keys(), key=well_sort_key):
        for e_type in E_TYPES:
            values = [e for _, e in data[well].get(e_type, [])]
            stats = analyze_consistency_single(values)
            if stats is not None:
                result[e_type].append({"well": well, **stats})
    return result


def main():
    if len(sys.argv) > 1:
        folder_names = [f.strip() for f in sys.argv[1:] if f.strip()]
    else:
        folder_names = [f for f in FOLDER_NAMES if f]
    if not folder_names:
        print("Usage: python analyze_measurement_consistency.py <folder1> <folder2> ...")
        print("  Or edit FOLDER_NAMES in the script.")
        print(f"\nExample folders in {PLOTS_BASE}:")
        if os.path.isdir(PLOTS_BASE):
            for d in sorted(os.listdir(PLOTS_BASE))[:15]:
                if d.startswith("run_") and os.path.isdir(os.path.join(PLOTS_BASE, d)):
                    print(f"  {d}")
        return 1
    print(f"📂 Analyzing {len(folder_names)} run folders...")
    data = collect_elastic_moduli(folder_names)
    wells_with_data = sum(1 for v in data.values() if any(v.values()))
    total_well_runs = sum(len(lst) for v in data.values() for lst in v.values())
    print(f"   Found {total_well_runs} well-run pairs across {wells_with_data} wells")
    rows_by_type = analyze_consistency(data)
    if not any(rows_by_type.values()):
        print("❌ No wells with ≥2 runs. Need more data for consistency analysis.")
        return 1

    plotter = ASMIPlotter(font_size=8)
    n_runs = len(folder_names)

    for e_type in E_TYPES:
        rows = rows_by_type[e_type]
        if not rows:
            continue
        label = E_TYPE_LABELS[e_type]
        print("\n" + "=" * 90)
        print(f"Elastic Modulus Consistency by Well – {label}")
        print("=" * 90)
        print(f"{'Well':<8} {'n':>4} {'Mean (Pa)':>12} {'Std (Pa)':>12} {'CV (%)':>8} {'Min (Pa)':>12} {'Max (Pa)':>12}")
        print("-" * 90)
        for r in rows:
            print(f"{r['well']:<8} {r['n']:>4} {r['mean_Pa']:>12.0f} {r['std_Pa']:>12.0f} {r['cv_pct']:>7.2f}% {r['min_Pa']:>12.0f} {r['max_Pa']:>12.0f}")
        print("=" * 90)
        all_cv = [r["cv_pct"] for r in rows]
        mean_cv = sum(all_cv) / len(all_cv) if all_cv else 0
        max_cv_well = max(rows, key=lambda x: x["cv_pct"]) if rows else None
        print(f"\n📊 {label}: mean CV = {mean_cv:.2f}%, max CV = {max_cv_well['cv_pct']:.2f}% (well {max_cv_well['well']})")

        # Save CSV
        out_csv = os.path.join(PLOTS_BASE, f"consistency_analysis_{e_type}.csv")
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            f.write("Well,n,Mean_Pa,Std_Pa,CV_pct,Min_Pa,Max_Pa\n")
            for r in rows:
                f.write(f"{r['well']},{r['n']},{r['mean_Pa']:.2f},{r['std_Pa']:.2f},{r['cv_pct']:.2f},{r['min_Pa']:.2f},{r['max_Pa']:.2f}\n")
        print(f"💾 Saved: {out_csv}")

        wells = [r["well"] for r in rows]
        means = [r["mean_Pa"] for r in rows]
        stds = [r["std_Pa"] for r in rows]
        cvs = [r["cv_pct"] for r in rows]
        x = range(len(wells))
        row_colors = plt.cm.tab10([0, 1, 2, 3, 4, 5, 6, 7])
        bar_colors = [row_colors[(ord(w[0].upper()) - ord("A")) % 8] for w in wells]

        # Bar plot
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        fig.suptitle(f"Elastic Modulus Consistency by Well – {label} (n={n_runs} runs)", fontsize=14, fontweight="bold")
        axes[0].bar(x, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=2, error_kw={"elinewidth": 1})
        mean_mean = sum(means) / len(means)
        axes[0].axhline(y=mean_mean, color="gray", linestyle="--", linewidth=1.5, label=f"Mean = {mean_mean:.0f} Pa")
        axes[0].set_ylabel("Mean ± Std (Pa)")
        axes[0].set_title("Mean Elastic Modulus with Error Bars (Std)")
        axes[0].grid(axis="y", alpha=0.3)
        axes[0].legend(loc="upper right", fontsize=8)
        axes[1].bar(x, cvs, color=bar_colors, alpha=0.8)
        axes[1].axhline(y=mean_cv, color="gray", linestyle="--", linewidth=1.5, label=f"Mean CV = {mean_cv:.2f}%")
        axes[1].set_ylabel("CV (%)")
        axes[1].set_xlabel("Well")
        axes[1].set_title("Coefficient of Variation")
        axes[1].grid(axis="y", alpha=0.3)
        axes[1].legend(loc="upper right", fontsize=8)
        plt.xticks(x, wells, rotation=90, fontsize=8)
        plt.tight_layout()
        out_plot = os.path.join(PLOTS_BASE, f"consistency_analysis_{e_type}.png")
        plt.savefig(out_plot, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Saved: {out_plot}")

        # Scatter plot
        fig2, ax2 = plt.subplots(figsize=(14, 6))
        all_x, all_y, all_colors = [], [], []
        well_to_idx = {w: i for i, w in enumerate(wells)}
        for well in wells:
            values = [e for _, e in data[well].get(e_type, [])]
            n_pts = len(values)
            jitter = np.random.uniform(-0.15, 0.15, n_pts) if n_pts > 1 else np.zeros(n_pts)
            xs = np.full(n_pts, well_to_idx[well]) + jitter
            all_x.extend(xs)
            all_y.extend(values)
            all_colors.extend([bar_colors[well_to_idx[well]]] * n_pts)
        ax2.scatter(all_x, all_y, c=all_colors, alpha=0.7, s=25, edgecolors="white", linewidths=0.3)
        ax2.set_xlabel("Well")
        ax2.set_ylabel("Elastic Modulus (Pa)")
        ax2.set_title(f"Elastic Modulus by Well – {label} (n={n_runs} runs)")
        ax2.set_xticks(x)
        ax2.set_xticklabels(wells, rotation=90, fontsize=8)
        ax2.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        out_scatter = os.path.join(PLOTS_BASE, f"consistency_scatter_{e_type}.png")
        plt.savefig(out_scatter, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"📊 Saved: {out_scatter}")

        # Heatmap
        heatmap_csv = os.path.join(PLOTS_BASE, f"consistency_heatmap_src_{e_type}.csv")
        with open(heatmap_csv, "w", newline="", encoding="utf-8") as f:
            f.write("Well,ElasticModulus,Std\n")
            for r in rows:
                f.write(f"{r['well']},{r['mean_Pa']:.2f},{r['std_Pa']:.2f}\n")
        out_heatmap = os.path.join(PLOTS_BASE, f"consistency_heatmap_{e_type}.png")
        plotter.plot_well_heatmap(
            heatmap_csv,
            value_col="ElasticModulus",
            cmap="viridis",
            annotate=True,
            save_path=out_heatmap,
            convert_to_mpa=True,
            title_suffix=f" (Consistency Mean ± Std, n={n_runs} runs)",
        )
        print(f"📊 Saved: {out_heatmap}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
