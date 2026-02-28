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

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg") # use Agg backend for matplotlib to avoid X11 window system

# Base path for plot folders (relative to script directory)
PLOTS_BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "plots")

# Edit this list or pass folder names as command-line arguments
FOLDER_NAMES = [
    "run_747_20260225_183711",
    "run_748_20260225_191006",
    "run_749_20260225_194358",
    "run_750_20260225_201811",
    "run_751_20260225_205238",
    "run_752_20260225_212710",
    "run_753_20260225_220137",
    "run_754_20260225_223611",
    "run_755_20260225_231054",
    "run_756_20260225_234537",
    "run_757_20260226_002020"
]

# Regex to extract Elastic Modulus from summary (handles "Elastic Modulus: 899074 Pa")
ELASTIC_MODULUS_RE = re.compile(r"Elastic Modulus:\s*([\d.]+)\s*Pa", re.IGNORECASE)


def parse_summary_file(filepath: str) -> float | None:
    """Parse Elastic Modulus from a well summary file. Returns None if not found or failed fit."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        m = ELASTIC_MODULUS_RE.search(content) # search for the Elastic Modulus in the summary file
        if m: # if found, return the Elastic Modulus
            return float(m.group(1))
    except (OSError, ValueError): # if not found, return None
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


def collect_elastic_moduli(folder_names: list[str]) -> dict[str, list[tuple[str, float]]]:
    """
    Collect Elastic Modulus for each well across all run folders.
    Returns: {well: [(run_folder, E_Pa), ...]}
    """
    data: dict[str, list[tuple[str, float]]] = defaultdict(list)
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
            E = parse_summary_file(filepath)
            if E is not None:
                data[well].append((folder, E))
    return dict(data)


def analyze_consistency(data: dict[str, list[tuple[str, float]]]) -> list[dict]:
    """Compute consistency stats for each well. Skips wells with < 2 runs."""
    rows = []
    for well in sorted(data.keys(), key=well_sort_key):
        values = [e for _, e in data[well]]
        if len(values) < 2:
            continue  # Need at least 2 runs for consistency
        n = len(values)
        mean_e = sum(values) / n
        variance = sum((x - mean_e) ** 2 for x in values) / (n - 1) if n > 1 else 0
        std_e = variance ** 0.5
        cv_pct = (std_e / mean_e * 100) if mean_e > 0 else 0
        rows.append({
            "well": well,
            "n": n,
            "mean_Pa": mean_e,
            "std_Pa": std_e,
            "cv_pct": cv_pct,
            "min_Pa": min(values),
            "max_Pa": max(values),
        })
    return rows


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
    wells_with_data = sum(1 for v in data.values() if v)
    total_well_runs = sum(len(v) for v in data.values())
    print(f"   Found {total_well_runs} well-run pairs across {wells_with_data} wells")
    rows = analyze_consistency(data)
    if not rows:
        print("❌ No wells with ≥2 runs. Need more data for consistency analysis.")
        return 1
    # Print table
    print("\n" + "=" * 90)
    print("Elastic Modulus Consistency by Well")
    print("=" * 90)
    print(f"{'Well':<8} {'n':>4} {'Mean (Pa)':>12} {'Std (Pa)':>12} {'CV (%)':>8} {'Min (Pa)':>12} {'Max (Pa)':>12}")
    print("-" * 90)
    for r in rows:
        print(f"{r['well']:<8} {r['n']:>4} {r['mean_Pa']:>12.0f} {r['std_Pa']:>12.0f} {r['cv_pct']:>7.2f}% {r['min_Pa']:>12.0f} {r['max_Pa']:>12.0f}")
    print("=" * 90)
    # Summary stats
    all_cv = [r["cv_pct"] for r in rows]
    mean_cv = sum(all_cv) / len(all_cv) if all_cv else 0
    max_cv_well = max(rows, key=lambda x: x["cv_pct"]) if rows else None
    print(f"\n📊 Overall: mean CV = {mean_cv:.2f}%, max CV = {max_cv_well['cv_pct']:.2f}% (well {max_cv_well['well']})")
    # Save CSV
    out_csv = os.path.join(PLOTS_BASE, "consistency_analysis.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        f.write("Well,n,Mean_Pa,Std_Pa,CV_pct,Min_Pa,Max_Pa\n")
        for r in rows:
            f.write(f"{r['well']},{r['n']},{r['mean_Pa']:.2f},{r['std_Pa']:.2f},{r['cv_pct']:.2f},{r['min_Pa']:.2f},{r['max_Pa']:.2f}\n")
    print(f"💾 Saved: {out_csv}")

    # Plot Mean ± Std (error bars) and CV% in 2 subplots
    wells = [r["well"] for r in rows]
    means = [r["mean_Pa"] for r in rows]
    stds = [r["std_Pa"] for r in rows]
    cvs = [r["cv_pct"] for r in rows]
    x = range(len(wells))
    n_runs = len(folder_names)

    # Row colors: A-H each get a distinct color
    row_colors = plt.cm.tab10([0, 1, 2, 3, 4, 5, 6, 7])
    bar_colors = [row_colors[(ord(w[0].upper()) - ord("A")) % 8] for w in wells]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Elastic Modulus Consistency by Well (over {n_runs} measurements)", fontsize=14, fontweight="bold")

    axes[0].bar(x, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=2, error_kw={"elinewidth": 1})
    mean_mean = sum(means) / len(means)
    axes[0].axhline(y=mean_mean, color="gray", linestyle="--", linewidth=1.5, label=f"Mean = {mean_mean:.0f} Pa")
    axes[0].set_ylabel("Mean ± Std (Pa)")
    axes[0].set_title("Mean Elastic Modulus with Error Bars (Std)")
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)

    axes[1].bar(x, cvs, color=bar_colors, alpha=0.8)
    mean_cv = sum(cvs) / len(cvs)
    axes[1].axhline(y=mean_cv, color="gray", linestyle="--", linewidth=1.5, label=f"Mean CV = {mean_cv:.2f}%")
    axes[1].set_ylabel("CV (%)")
    axes[1].set_xlabel("Well")
    axes[1].set_title("Coefficient of Variation")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(loc="upper right", fontsize=8)

    plt.xticks(x, wells, rotation=90, fontsize=8)
    plt.tight_layout()
    out_plot = os.path.join(PLOTS_BASE, "consistency_analysis.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {out_plot}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
