"""
Convert measurement data from format:
  Z_Position(mm), Raw_Force(N), Corrected_Force(N)
to format expected by analysis_batch.py:
  well, depth, force

Use main() with parameters. Edit the parameters below when running as script.
"""

import csv
import os
import statistics

COLS = ["A", "B", "C", "D", "E", "F", "G", "H"]
ROWS = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"]


def parse_well_from_filename(filename):
    """Extract well from filename like A2.csv, well_A2.csv, A2_measurements.csv."""
    base = os.path.splitext(os.path.basename(filename))[0]
    for col in COLS:
        for row in ROWS:
            well = col + row
            if well in base.upper() or f"_{well}_" in base.upper() or base.upper().endswith(f"_{well}"):
                return well
    return None


def _print_baselines_from_folder(folder_name, default_well=None, baseline_points=10):
    """Print baseline for each CSV in folder."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    for folder_path in [
        os.path.join(script_dir, folder_name),
        os.path.join(script_dir, "data", folder_name),
        os.path.join(base_dir, "results", "measurements", folder_name),
        os.path.join(base_dir, folder_name),
        folder_name,
    ]:
        if os.path.isdir(folder_path):
            break
    else:
        raise FileNotFoundError(f"Folder not found: {folder_name}")
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv") and not f.endswith("_results.csv")]
    for f in sorted(csv_files):
        inp = os.path.join(folder_path, f)
        rows = _read_and_convert(inp, well=default_well)
        if rows:
            n = min(baseline_points, len(rows))
            mean_f, stdev_f, well = compute_baseline_from_points(rows, n)
            print(f"  {f} (well {well}): baseline={mean_f:.6f} N, stdev={stdev_f:.6f} (first {n} pts)")
        else:
            print(f"  {f}: no valid data")


def compute_baseline_from_points(rows, n):
    """Compute baseline force (mean, stdev) from first N points. rows = [[well, z, force], ...]"""
    if len(rows) < n:
        return None, None, None
    forces = [float(r[2]) for r in rows[:n]]
    stdev = statistics.stdev(forces) if len(forces) > 1 else 0.0
    return statistics.mean(forces), stdev, rows[0][0]


def convert_file(input_path, output_path, well=None, baseline_points=None):
    """
    Convert CSV from Z_Position, Raw_Force, Corrected_Force to well, depth, force.
    Uses Raw_Force (analysis does its own correction).
    If baseline_points=N, compute baseline from first N points and prepend as row 1.
    """
    output_rows = _read_and_convert(input_path, well=well)
    if not output_rows:
        raise ValueError(
            f"No valid data or cannot determine well from '{input_path}'. Use --well A2."
        )

    if baseline_points is not None:
        mean_f, stdev_f, well = compute_baseline_from_points(output_rows, baseline_points)
        if mean_f is None:
            raise ValueError(
                f"Need at least {baseline_points} points to compute baseline, "
                f"but only {len(output_rows)} rows found."
            )
        # Prepend baseline row: well, baseline_force, stdev
        baseline_row = [well, mean_f, stdev_f]
        # Keep only contact data (from point N+1 onwards)
        output_rows = [baseline_row] + output_rows[baseline_points:]
        print(f"  Baseline: force={mean_f:.6f} N, stdev={stdev_f:.6f} (from first {baseline_points} points)")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)
    return len(output_rows)


def convert_folder(folder_name, output_path=None, default_well=None, baseline_points=None):
    """
    Convert all CSV files in folder into one combined file (well, depth, force).
    Each input file = one well; well from filename (e.g. A2.csv) or default_well if single file.
    If baseline_points=N, compute baseline from first N points per file.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(script_dir)
    folder_paths = [
        os.path.join(script_dir, folder_name),
        os.path.join(script_dir, "data", folder_name),
        os.path.join(base_dir, "results", "measurements", folder_name),
        os.path.join(os.getcwd(), "results", "measurements", folder_name),
        os.path.join(base_dir, folder_name),
        folder_name,
    ]
    folder_path = None
    for p in folder_paths:
        if os.path.isdir(p):
            folder_path = p
            break
    if folder_path is None:
        raise FileNotFoundError(f"Folder not found: {folder_name}")

    combined_name = os.path.basename(folder_path.rstrip("/")) + ".csv"
    csv_files = [
        f for f in os.listdir(folder_path)
        if f.endswith(".csv") and not f.endswith("_results.csv") and f != combined_name
    ]

    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {folder_path}")

    combined_rows = []
    for f in sorted(csv_files):
        inp = os.path.join(folder_path, f)
        well = parse_well_from_filename(f) or default_well
        rows = _read_and_convert(inp, well=well)
        if rows:
            if baseline_points is not None:
                mean_f, stdev_f, well = compute_baseline_from_points(rows, baseline_points)
                if mean_f is not None:
                    baseline_row = [well, mean_f, stdev_f]
                    rows = [baseline_row] + rows[baseline_points:]
                    print(f"  {f} -> {len(rows)} rows (well: {well}, baseline={mean_f:.6f} N from first {baseline_points} pts)")
                else:
                    print(f"  {f} -> 0 rows (need {baseline_points}+ points for baseline)")
                    continue
            else:
                print(f"  {f} -> {len(rows)} rows (well: {rows[0][0]})")
            combined_rows.extend(rows)
        else:
            print(f"  {f} -> 0 rows (skipped, well: {well or '?'})")

    if not combined_rows:
        raise ValueError("No valid data to convert. Use --well A2 if filenames don't indicate well.")

    out = output_path or os.path.join(folder_path, os.path.basename(folder_path.rstrip("/")) + ".csv")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(combined_rows)
    print(f"Combined output: {out} ({len(combined_rows)} rows)")
    return len(combined_rows)


def _is_well(s):
    """Check if string looks like well (e.g. A2, B3)."""
    s = str(s).strip().upper()
    return len(s) >= 2 and s[0] in "ABCDEFGH" and s[1:].isdigit()


def _can_float(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError, IndexError):
        return False


def _read_and_convert(input_path, well=None):
    """Read CSV and return list of [well, depth, force] rows.
    Handles:
      - Timestamp, Z_Position(mm), Raw_Force(N), Corrected_Force(N)  [main_asmi format: col1=Z, col2=force]
      - Z_Position(mm), Raw_Force(N), Corrected_Force(N)  [no well column]
      - well, depth, force  [already in target format]
    Uses Raw_Force (analysis does its own correction).
    """
    with open(input_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return []

    start_row = 0
    z_col, force_col = 0, 1
    well_col = None  # if set, each row has well in this column

    # Find header row that contains Z_Position and Raw_Force (handles metadata rows before data)
    header_row_idx = None
    for idx, row in enumerate(rows):
        if not row or len(row) < 2:
            continue
        header_lower = [str(h).lower().strip() for h in row]
        has_z = any("z_position" in h or ("z" in h and "position" in h) for h in header_lower)
        has_force = any("raw" in h and "force" in h for h in header_lower)
        if has_z and has_force:
            header_row_idx = idx
            break

    if header_row_idx is not None:
        start_row = header_row_idx + 1
        header_lower = [str(h).lower().strip() for h in rows[header_row_idx]]
        for i, h in enumerate(header_lower):
            if "z_position" in h or ("z" in h and "position" in h):
                z_col = i
            elif "raw" in h and "force" in h:
                force_col = i
            elif "raw" in h:
                force_col = i
            elif "force" in h and i == 2:
                force_col = i
    else:
        # No header found; infer from first data row
        first_data = None
        for row in rows:
            if len(row) >= 3 and _can_float(row[0]):
                first_data = row
                break
        if first_data is not None:
            if len(first_data) >= 4:
                # [timestamp, Z, raw_force, corrected_force] - main_asmi format
                z_col, force_col = 1, 2
            elif _is_well(rows[0][0]) and len(rows[0]) >= 3:
                well_col, z_col, force_col = 0, 1, 2
            else:
                z_col, force_col = 0, 1
        elif rows[0] and _is_well(rows[0][0]) and len(rows[0]) >= 3:
            well_col, z_col, force_col = 0, 1, 2
        else:
            z_col, force_col = 0, 1

    result = []
    for i in range(start_row, len(rows)):
        row = rows[i]
        if len(row) <= max(z_col, force_col):
            continue
        w = well
        if well_col is not None and len(row) > well_col:
            w = row[well_col]
        if w is None:
            w = parse_well_from_filename(input_path)
        if w is None:
            continue
        try:
            z_val = float(row[z_col])
            force_val = float(row[force_col])
            result.append([w, z_val, force_val])
        except (ValueError, IndexError):
            continue
    return result


def main(
    input_path,
    well=None,
    output_path=None,
    baseline_points=None,
    baseline_only=False,
):
    """
    Convert measurement format. Edit parameters when running as script.

    Parameters
    ----------
    input_path : str
        Input CSV file or folder (e.g., "run_774_20260206_133925")
    well : str, optional
        Well name (e.g., "A2"). Required if single file or filename doesn't indicate well.
    output_path : str, optional
        Output path. For single file: output CSV. For folder: combined output CSV.
    baseline_points : int, optional
        Compute baseline from first N points (no-contact) and prepend as row 1.
    baseline_only : bool, default False
        Only compute and print baseline force, do not convert/write file.
    """
    if os.path.isfile(input_path):
        if baseline_only:
            rows = _read_and_convert(input_path, well=well)
            if not rows:
                print("No valid data.")
            else:
                n = baseline_points or 10
                if len(rows) < n:
                    n = len(rows)
                    print(f"Using first {n} points (fewer than requested)")
                mean_f, stdev_f, w = compute_baseline_from_points(rows, n)
                print(f"Well: {w}")
                print(f"Baseline force: {mean_f:.6f} N")
                print(f"Stdev: {stdev_f:.6f} N")
        else:
            out = output_path or input_path.replace(".csv", "_converted.csv")
            if out == input_path:
                out = input_path.replace(".csv", "_converted.csv")
            n = convert_file(input_path, out, well=well, baseline_points=baseline_points)
            print(f"Converted {n} rows -> {out}")
    elif os.path.isdir(input_path):
        if baseline_only:
            _print_baselines_from_folder(input_path, well, baseline_points or 10)
        else:
            convert_folder(
                input_path, output_path=output_path, default_well=well, baseline_points=baseline_points
            )
    else:
        if baseline_only:
            _print_baselines_from_folder(input_path, well, baseline_points or 10)
        else:
            convert_folder(
                input_path, output_path=output_path, default_well=well, baseline_points=baseline_points
            )


if __name__ == "__main__":
    main(
        input_path="run_774_20260206_133925",
        well=None,
        output_path=None,
        baseline_points=None,
        baseline_only=False,
    )
