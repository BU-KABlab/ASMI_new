"""
Batch analysis script for ASMI measurements.
Input a folder name to automatically analyze all wells and output E.
Use main() with parameters. Edit the parameters below when running as script.
Set baseline_points to auto-compute baseline from first N points (for data without baseline row).

Integrates convert_measurement_format for converting Z/Raw_Force/Corrected_Force to well/depth/force.
"""

import csv
import sys
import os
import string
import numpy as np
import matplotlib.pyplot as pyplot
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.optimize import curve_fit

from .convert_measurement_format import convert_folder, convert_file, compute_baseline_from_points

COLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
ROWS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]

# 96-well plate layout: A-H x 1-12
PLATE_ROWS = list(string.ascii_uppercase[:8])
PLATE_COLS = list(range(1, 13))


def load_csv_from_folder(folder_name):
    """Load data from CSV file in the given folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir) if script_dir else os.getcwd()

    # Possible paths: folder/folder.csv, data/folder/, results/measurements/folder/
    possible_paths = [
        os.path.join(base_dir, folder_name, folder_name + ".csv"),
        os.path.join(base_dir, folder_name + ".csv"),
        os.path.join(base_dir, "data", folder_name, folder_name + ".csv"),
        os.path.join(base_dir, "data", folder_name + ".csv"),
        os.path.join(base_dir, "results", "measurements", folder_name, folder_name + ".csv"),
        os.path.join(script_dir, folder_name, folder_name + ".csv"),
        os.path.join(script_dir, folder_name + ".csv"),
        os.path.join(os.getcwd(), "results", "measurements", folder_name, folder_name + ".csv"),
        os.path.join(os.getcwd(), folder_name, folder_name + ".csv"),
        os.path.join(os.getcwd(), folder_name + ".csv"),
    ]

    csv_path = None
    for path in possible_paths:
        if os.path.isfile(path):
            csv_path = path
            break

    if csv_path is None:
        # Try to find any .csv in folder (check base_dir, data/, results/measurements/)
        for folder_path in [
            os.path.join(base_dir, folder_name),
            os.path.join(base_dir, "data", folder_name),
            os.path.join(base_dir, "results", "measurements", folder_name),
            os.path.join(script_dir, folder_name),
            os.path.join(script_dir, "data", folder_name),
            os.path.join(os.getcwd(), "results", "measurements", folder_name),
        ]:
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.endswith(".csv") and not f.endswith("_results.csv"):
                        csv_path = os.path.join(folder_path, f)
                        break
                if csv_path is not None:
                    break

    if csv_path is None:
        raise FileNotFoundError(
            f"Could not find CSV file for folder '{folder_name}'. "
            f"Looked in: {script_dir}, {base_dir}, {os.getcwd()}"
        )

    with open(csv_path, "r") as file:
        reader = csv.reader(file)
        data = list(reader)

    cleaned_data = [row for row in data if row]
    return cleaned_data


def collect_run_data(data, well, p_ratio, negative_force_is_contact=True):
    """Collect data for specific run from csv file. Returns (run_array, p_ratio) or (None, None) on failure.
    If negative_force_is_contact=True (default): negative force = compressive/contact, positive = no contact.
    If False: original measure convention (positive = contact after zeroing).
    """
    well_data = []
    no_contact = []
    run_array = []
    forces = []
    for i in range(0, len(data)):
        if data[i][0] == well:
            values = [data[i][1], data[i][2]]
            well_data.append(values)

    if len(well_data) == 0:
        return None, p_ratio

    threshold = -1 * float(well_data[0][0]) + 2 * float(well_data[0][1])
    for l in range(1, len(well_data)):
        f = float(well_data[l][1])
        is_no_contact = (f <= threshold) if not negative_force_is_contact else (f > threshold)
        if is_no_contact:
            no_contact.append(l)
        run_array.append([well_data[l][0], well_data[l][1]])

    if len(no_contact) > 0:
        last_no_contact = int(no_contact[len(no_contact) - 1])
        contact_after = len(run_array) - last_no_contact - 1
        if contact_after <= 10:
            if last_no_contact + 1 >= len(well_data):
                start_val = 1
            else:
                return None, p_ratio
        else:
            start_val = last_no_contact + 1
    else:
        start_val = 1  # use first data row for depth offset (row 0 is baseline: force, stdev only)
    baseline_force = float(well_data[0][0])
    for k in range(0, len(run_array)):
        run_array[k][0] = round(-1 * (float(run_array[k][0]) - float(well_data[start_val][0])), 2)
        raw_force = float(run_array[k][1])
        if negative_force_is_contact and raw_force < 0:
            run_array[k][1] = -raw_force + baseline_force
        else:
            run_array[k][1] = raw_force + baseline_force
        forces.append(run_array[k][1])

    if forces == [] or max(forces) - min(forces) < 0.02:
        return None, p_ratio

    return run_array, p_ratio


def approximate_height(run_array):
    depths = []
    for i in range(0, len(run_array)):
        depths.append(run_array[i][0])
    for j in range(0, len(depths)):
        depths[j] = abs(depths[j])
    zero = min(depths)
    num = depths.index(zero)
    z_pos = (num * 0.02) + 3
    approx_height = 15 - z_pos
    return approx_height


def split(run_array):
    depths = []
    forces = []
    for i in range(0, len(run_array)):
        depths.append(run_array[i][0])
        forces.append(run_array[i][1])
    return depths, forces


def find_d_and_f_in_range(run_array):
    forces = []
    depths = []
    for i in range(0, len(run_array)):
        if run_array[i][0] >= 0.24 and run_array[i][0] <= 0.5:
            forces.append(run_array[i][1])
            depths.append(run_array[i][0])
    return depths, forces


def correct_force(depths, forces, p_ratio, approx_height):
    new_array = []
    for i in range(0, len(depths)):
        if p_ratio < 0.325:
            if approx_height >= 9.5:
                b, c = 0.13, 1.24
            elif approx_height >= 8.5 and approx_height < 9.5:
                b, c = 0.131, 1.24
            elif approx_height >= 7.5 and approx_height < 8.5:
                b, c = 0.133, 1.25
            elif approx_height >= 6.5 and approx_height < 7.5:
                b, c = 0.132, 1.24
            elif approx_height >= 5.5 and approx_height < 6.5:
                b, c = 0.132, 1.24
            elif approx_height >= 4.5 and approx_height < 5.5:
                b, c = 0.139, 1.27
            elif approx_height >= 3.5 and approx_height < 4.5:
                b, c = 0.149, 1.3
            else:
                b, c = 0.162, 1.38
        elif p_ratio >= 0.325 and p_ratio < 0.375:
            if approx_height >= 9.5:
                b, c = 0.132, 1.25
            elif approx_height >= 8.5 and approx_height < 9.5:
                b, c = 0.132, 1.25
            elif approx_height >= 7.5 and approx_height < 8.5:
                b, c = 0.134, 1.25
            elif approx_height >= 6.5 and approx_height < 7.5:
                b, c = 0.136, 1.26
            elif approx_height >= 5.5 and approx_height < 6.5:
                b, c = 0.126, 1.25
            elif approx_height >= 4.5 and approx_height < 5.5:
                b, c = 0.133, 1.27
            elif approx_height >= 3.5 and approx_height < 4.5:
                b, c = 0.144, 1.32
            else:
                b, c = 0.169, 1.42
        elif p_ratio >= 0.375 and p_ratio < 0.425:
            if approx_height >= 9.5:
                b, c = 0.181, 1.33
            elif approx_height >= 8.5 and approx_height < 9.5:
                b, c = 0.182, 1.34
            elif approx_height >= 7.5 and approx_height < 8.5:
                b, c = 0.183, 1.34
            elif approx_height >= 6.5 and approx_height < 7.5:
                b, c = 0.183, 1.34
            elif approx_height >= 5.5 and approx_height < 6.5:
                b, c = 0.194, 1.38
            elif approx_height >= 4.5 and approx_height < 5.5:
                b, c = 0.198, 1.4
            elif approx_height >= 3.5 and approx_height < 4.5:
                b, c = 0.203, 1.44
            else:
                b, c = 0.176, 1.46
        elif p_ratio >= 0.425 and p_ratio < 0.475:
            if approx_height >= 9.5:
                b, c = 0.156, 1.35
            elif approx_height >= 8.5 and approx_height < 9.5:
                b, c = 0.152, 1.34
            elif approx_height >= 7.5 and approx_height < 8.5:
                b, c = 0.156, 1.35
            elif approx_height >= 6.5 and approx_height < 7.5:
                b, c = 0.161, 1.37
            elif approx_height >= 5.5 and approx_height < 6.5:
                b, c = 0.153, 1.37
            elif approx_height >= 4.5 and approx_height < 5.5:
                b, c = 0.166, 1.42
            elif approx_height >= 3.5 and approx_height < 4.5:
                b, c = 0.179, 1.47
            else:
                b, c = 0.205, 1.59
        else:
            if approx_height >= 9.5:
                b, c = 0.203, 1.58
            elif approx_height >= 8.5 and approx_height < 9.5:
                b, c = 0.207, 1.6
            elif approx_height >= 7.5 and approx_height < 8.5:
                b, c = 0.212, 1.62
            elif approx_height >= 6.5 and approx_height < 7.5:
                b, c = 0.217, 1.65
            elif approx_height >= 5.5 and approx_height < 6.5:
                b, c = 0.21, 1.64
            elif approx_height >= 4.5 and approx_height < 5.5:
                b, c = 0.22, 1.68
            elif approx_height >= 3.5 and approx_height < 4.5:
                b, c = 0.17, 1.58
            else:
                b, c = 0.182, 1.64
        val = (forces[i]) / (c * pow(depths[i], b))
        new_array.append(val)
    return new_array


def adjust_depth(run_array, d0):
    for i in range(0, len(run_array)):
        run_array[i][0] = run_array[i][0] - d0
    return run_array


def find_E(A, p_ratio):
    r_sphere = 0.0025
    sphere_p_ratio = 0.28
    sphere_E = 1.8 * pow(10, 11)
    polymer_p_ratio = p_ratio
    actual_A = A * pow(1000, 1.5)
    E_star = (actual_A * 0.75) / pow(r_sphere, 0.5)
    E_inv = (
        1 / (E_star * (1 - pow(polymer_p_ratio, 2)))
        - (1 - pow(sphere_p_ratio, 2)) / (sphere_E * (1 - pow(polymer_p_ratio, 2)))
    )
    E_polymer = 1 / E_inv
    return E_polymer


def adjust_E(E):
    if E < 660000:
        factor = 457 * pow(E, -0.457)
        E = E / factor
    return E


def Hertz_func(depth, A, d0):
    return A * pow(depth - d0, 1.5)


def get_wells_from_data(data):
    """Get unique wells from data (first column)."""
    wells = []
    seen = set()
    for row in data:
        if len(row) >= 1 and row[0] not in seen:
            w = row[0]
            if w[0] in COLS and w.lstrip("ABCDEFGH") in ROWS:
                wells.append(w)
                seen.add(w)
    return wells


def analyze_well(data, well, p_ratio, show_plot=False, save_plot_path=None):
    """
    Analyze a single well and return (E, std_dev, r2) or (None, None, None) on failure.
    """
    run_array, p_ratio = collect_run_data(data, well, p_ratio)
    if run_array is None:
        return None, None, None

    height = approximate_height(run_array)
    depth_in_range, force_in_range = find_d_and_f_in_range(run_array)
    adjusted_forces = correct_force(depth_in_range, force_in_range, p_ratio, height)
    depth_in_range = np.asarray(depth_in_range)
    adjusted_forces = np.asarray(adjusted_forces)

    try:
        parameters, covariance = curve_fit(
            Hertz_func, depth_in_range, adjusted_forces, p0=[2, 0.03], maxfev=2000
        )
    except Exception:
        return None, None, None

    fit_A = float(parameters[0])
    fit_d0 = float(parameters[1])
    count = 0
    continue_to_adjust = abs(fit_d0) >= 0.01
    min_d0 = 100

    while continue_to_adjust:
        count += 1
        old_d0 = fit_d0
        run_array = adjust_depth(run_array, fit_d0)
        height = approximate_height(run_array)
        depth_in_range, force_in_range = find_d_and_f_in_range(run_array)
        adjusted_forces = correct_force(depth_in_range, force_in_range, p_ratio, height)
        depth_in_range = np.asarray(depth_in_range)
        adjusted_forces = np.asarray(adjusted_forces)
        try:
            parameters, covariance = curve_fit(
                Hertz_func, depth_in_range, adjusted_forces, p0=[2, 0.03], maxfev=2000
            )
        except Exception:
            return None, None, None
        fit_A = float(parameters[0])
        fit_d0 = float(parameters[1])
        if abs(fit_d0) < min_d0:
            min_d0 = abs(fit_d0)
        if abs(round(old_d0, 5)) == abs(round(fit_d0, 5)):
            fit_d0 = -0.75 * fit_d0
        elif abs(fit_d0) < 0.01:
            continue_to_adjust = False
            break
        elif count > 100 and count < 200:
            if abs(round(fit_d0, 2)) == round(min_d0, 2):
                break
        elif count >= 200 and count < 300:
            if abs(round(fit_d0, 1)) == round(min_d0, 1):
                break
        elif count == 300:
            break

    E = find_E(fit_A, p_ratio)
    E = adjust_E(E)
    E = round(E)
    err = np.sqrt(np.diag(covariance))
    std_dev = round(find_E(err[0], p_ratio))

    # R² = 1 - SS_res/SS_tot
    y_pred = fit_A * np.power(np.maximum(depth_in_range - fit_d0, 0), 1.5)
    ss_res = np.sum((adjusted_forces - y_pred) ** 2)
    ss_tot = np.sum((adjusted_forces - np.mean(adjusted_forces)) ** 2)
    r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

    if show_plot or save_plot_path:
        pyplot.scatter(depth_in_range, adjusted_forces, label="Data")
        y_var = [fit_A * pow(depth_in_range[i], 1.5) for i in range(len(depth_in_range))]
        pyplot.plot(depth_in_range, y_var, label="Hertz fit")
        pyplot.xlabel("Depth (mm)")
        pyplot.ylabel("Force (N)")
        pyplot.title(f"Force vs. Indentation Depth of Well {well}")
        legend_text = f"E = {E} N/m²\nuncertainty = ±{std_dev} N/m²\nR² = {r2:.3f}\nA = {fit_A:.4f}\nd₀ = {fit_d0:.4f}"
        pyplot.legend(title=legend_text)
        if save_plot_path:
            pyplot.savefig(save_plot_path, dpi=150)
            pyplot.close()
        else:
            pyplot.show()

    return E, std_dev, r2


def run_analysis(folder_name, p_ratio=0.5, show_plot=False, save_plot=False, baseline_points=None, save_heatmap=True):
    """Load data from folder, analyze all wells, output E.
    If baseline_points is set, run convert first to add baseline from first N points per well.
    If save_plot=True, save plots to results/plots/<folder_name>/Well.png (no display).
    If save_heatmap=True (default), save E heatmap to results/plots/<folder_name>/heatmap.png.
    """
    if baseline_points is not None:
        convert_folder(
            folder_name,
            output_path=None,
            default_well=None,
            baseline_points=baseline_points,
        )
    data = load_csv_from_folder(folder_name)
    wells = get_wells_from_data(data)

    if not wells:
        print("No valid wells found in data.")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    plot_dir = os.path.join(base_dir, "results", "plots", folder_name) if (save_plot or save_heatmap) else None
    if plot_dir:
        os.makedirs(plot_dir, exist_ok=True)

    results = []
    for well in wells:
        save_path = os.path.join(plot_dir, f"{well}.png") if (plot_dir and save_plot) else None
        E, std_dev, r2 = analyze_well(
            data, well, p_ratio,
            show_plot=show_plot and not save_plot,
            save_plot_path=save_path,
        )
        if E is not None:
            results.append((well, E, std_dev, r2))
            print(f"Well {well}: E = {E} N/m^2, Uncertainty = {std_dev} N/m^2, R² = {r2:.3f}")
        else:
            results.append((well, "no data", "no data", None))
            print(f"Well {well}: no data (sample too short/soft or insufficient data)")

    if save_heatmap and plot_dir and results:
        heatmap_path = os.path.join(plot_dir, "heatmap.png")
        plot_e_heatmap(results, save_path=heatmap_path)

    return results


def plot_e_heatmap(
    results_or_csv,
    save_path=None,
    cmap="viridis",
    annotate=True,
    convert_to_mpa=True,
    font_size=10,
    title_suffix=None,
):
    """Plot a 96-well plate heatmap of Elastic Modulus (E) values.

    Args:
        results_or_csv: Either:
            - List of (well, E, std_dev, r2) or (well, E, std_dev) tuples from run_analysis()
            - Path to summary CSV with columns: Well, ElasticModulus, optional Std, optional R2
        save_path: Path to save the plot (if None, displays plot)
        cmap: Colormap name (default: 'viridis')
        annotate: Whether to annotate wells with values (default: True)
        convert_to_mpa: Convert Pa to MPa (default: True)
        font_size: Base font size for labels
        title_suffix: Optional suffix for title (e.g., " (System Corrected)")
    """
    well_to_idx = {
        f"{row}{col}": (i, j)
        for i, row in enumerate(PLATE_ROWS)
        for j, col in enumerate(PLATE_COLS)
    }

    heatmap = np.full((8, 12), np.nan)
    stdmap = np.full((8, 12), np.nan)
    r2map = np.full((8, 12), np.nan)

    if isinstance(results_or_csv, str):
        # Load from CSV
        with open(results_or_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                well = row.get("Well", "").strip()
                if well not in well_to_idx:
                    continue
                try:
                    val = float(row.get("ElasticModulus", row.get("E", np.nan)))
                except (ValueError, TypeError):
                    continue
                std_val = row.get("Std", row.get("Uncertainty", np.nan))
                try:
                    std_val = float(std_val) if std_val else np.nan
                except (ValueError, TypeError):
                    std_val = np.nan
                r2_val = row.get("R2", row.get("R²", np.nan))
                try:
                    r2_val = float(r2_val) if r2_val else np.nan
                except (ValueError, TypeError):
                    r2_val = np.nan
                i, j = well_to_idx[well]
                heatmap[i, j] = val / 1e6 if convert_to_mpa else val
                stdmap[i, j] = (std_val / 1e6) if (convert_to_mpa and not np.isnan(std_val)) else std_val
                r2map[i, j] = r2_val
    else:
        # Use results from run_analysis()
        for item in results_or_csv:
            well = item[0]
            E = item[1]
            std_dev = item[2]
            r2 = item[3] if len(item) >= 4 else np.nan
            if well not in well_to_idx:
                continue
            if E == "no data" or E is None:
                continue
            try:
                val = float(E)
            except (ValueError, TypeError):
                continue
            i, j = well_to_idx[well]
            heatmap[i, j] = val / 1e6 if convert_to_mpa else val
            try:
                std_val = float(std_dev) if std_dev != "no data" and std_dev is not None else np.nan
                stdmap[i, j] = (std_val / 1e6) if (convert_to_mpa and not np.isnan(std_val)) else std_val
            except (ValueError, TypeError):
                stdmap[i, j] = np.nan
            try:
                r2map[i, j] = float(r2) if r2 is not None and r2 != "no data" else np.nan
            except (ValueError, TypeError):
                r2map[i, j] = np.nan

    valid = heatmap[~np.isnan(heatmap)]
    if len(valid) == 0:
        print("No valid E values to plot.")
        return

    fig, ax = pyplot.subplots(figsize=(12, 7))
    vmin, vmax = np.nanmin(heatmap), np.nanmax(heatmap)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = pyplot.get_cmap(cmap)

    for i, row_label in enumerate(PLATE_ROWS):
        for j, col_label in enumerate(PLATE_COLS):
            x, y = j, 7 - i
            value = heatmap[i, j]
            color = cmap_obj(norm(value)) if not np.isnan(value) else (0.9, 0.9, 0.9, 1)
            circle = mpatches.Circle((x, y), 0.4, color=color, ec="black", lw=1.0)
            ax.add_patch(circle)
            if annotate and not np.isnan(value):
                rgb = cmap_obj(norm(value))[:3]
                brightness = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
                text_color = "black" if brightness > 0.5 else "white"
                ax.text(x, y + 0.1, f"{value:.2f}", ha="center", va="center", fontsize=font_size + 2, color=text_color, fontweight="bold")
                if not np.isnan(stdmap[i, j]):
                    ax.text(x, y - 0.05, f"±{stdmap[i, j]:.2f}", ha="center", va="center", fontsize=font_size, color=text_color, fontweight="bold")
                if not np.isnan(r2map[i, j]):
                    ax.text(x, y - 0.22, f"R²={r2map[i, j]:.2f}", ha="center", va="center", fontsize=font_size - 2, color=text_color)

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 7.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(12))
    ax.set_xticklabels([str(c) for c in PLATE_COLS])
    ax.set_yticks(range(8))
    ax.set_yticklabels(PLATE_ROWS[::-1])
    ax.tick_params(axis="both", which="major", labelsize=font_size + 4)

    unit = "MPa" if convert_to_mpa else "Pa"
    title = f"96-Well Plate Young's Modulus Heatmap ({unit})"
    if title_suffix:
        title = title + title_suffix
    ax.set_title(title, fontsize=font_size + 10)

    sm = pyplot.cm.ScalarMappable(cmap=cmap_obj, norm=norm)
    sm.set_array([])
    cbar = pyplot.colorbar(sm, ax=ax, pad=0.02)
    cbar.set_label(f"Elastic Modulus ({unit})", fontsize=font_size + 8)
    cbar.ax.tick_params(labelsize=font_size + 6)

    pyplot.tight_layout()
    if save_path:
        pyplot.savefig(save_path, dpi=300)
        print(f"Saved heatmap to {save_path}")
        # Save heatmap data to CSV
        csv_path = save_path.replace(".png", "_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Well", "Row", "Column", "ElasticModulus", "Std", "R2"])
            for i, row_label in enumerate(PLATE_ROWS):
                for j, col_label in enumerate(PLATE_COLS):
                    well = f"{row_label}{col_label}"
                    val = heatmap[i, j]
                    std_val = stdmap[i, j]
                    r2_val = r2map[i, j]
                    writer.writerow([
                        well, row_label, col_label,
                        f"{val:.4f}" if not np.isnan(val) else "",
                        f"{std_val:.4f}" if not np.isnan(std_val) else "",
                        f"{r2_val:.4f}" if not np.isnan(r2_val) else "",
                    ])
        print(f"Saved heatmap data to {csv_path}")
        pyplot.close()
    else:
        pyplot.show()


def main(
    folder,
    p_ratio=0.4,
    show_plot=False,
    save_plot=True,
    baseline_points=10,
    save_heatmap=True,
):
    """
    Analyze ASMI measurement folder and output elastic modulus E.
    Edit parameters when running as script.

    Parameters
    ----------
    folder : str
        Folder name containing CSV data (e.g., "run_774_20260206_133925")
    p_ratio : float, default 0.4
        Poisson's ratio (0.3 to 0.5)
    show_plot : bool, default False
        Whether to show plots (interactive)
    save_plot : bool, default True
        Whether to save plots to results/plots/<folder>/Well.png (no display)
    baseline_points : int or None, default 10
        If set, auto-compute baseline from first N points of each well (for data without baseline row).
        Set to None if data already has baseline row.
    save_heatmap : bool, default True
        Whether to save 96-well E heatmap to results/plots/<folder>/heatmap.png
    """
    if p_ratio < 0.3 or p_ratio > 0.5:
        print("Error: Poisson's ratio must be between 0.3 and 0.5")
        sys.exit(1)
    run_analysis(
        folder_name=folder,
        p_ratio=p_ratio,
        show_plot=show_plot,
        save_plot=save_plot,
        baseline_points=baseline_points,
        save_heatmap=save_heatmap,
    )


if __name__ == "__main__":
    try:
        main(
            folder="run_774_20260206_133925",
            p_ratio=0.5,
            show_plot=False,
            save_plot=True,
            baseline_points=10,
            save_heatmap=True,
        )
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
