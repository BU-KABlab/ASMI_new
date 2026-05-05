# ASMI — Automated Soft Matter Indenter

Automated mechanical characterization of soft materials. ASMI drives a CNC gantry with a GoDirect force sensor through the [CubOS](https://github.com/ursa-laboratories/CubOS) protocol engine, scans wells in a 96-well plate, and persists every measurement to SQLite.

## Installation

```bash
pip install -r requirements.txt
```

Hardware:
- CNC gantry over USB (GRBL, e.g. Genmitsu 3018 PRO)
- GoDirect force sensor (Vernier)

## Running `main_asmi.py`

`main_asmi.py` is config-driven — there are no Python arguments to edit. Set up the YAML in `configs/` (see below) and run:

```bash
python3 main_asmi.py                 # real hardware, configs from configs/
python3 main_asmi.py --mock          # offline dry run; mock gantry + mock force sensor
python3 main_asmi.py --no-home       # skip homing (already homed via UGS)
python3 main_asmi.py --skip-force-sensor  # real gantry, mock force sensor
```

### Flag semantics

| Flag                  | Gantry      | Force sensor | When to use |
|-----------------------|-------------|--------------|-------------|
| _(none)_              | real        | real         | Real measurement run |
| `--mock`              | mock        | mock         | YAML/protocol dry run, CI |
| `--skip-force-sensor` | real        | mock         | GoDirect firmware mismatch (`struct.error`); validating motion against real plate without acquiring force data |
| `--no-home`           | (real only) | unchanged    | Already homed in UGS — skip the homing cycle to avoid re-zeroing |

`--mock` and `--skip-force-sensor` both run instruments offline (`mock_mode=True` is passed to CubOS). `--mock` additionally swaps the gantry for an offline `Gantry(offline=True)`. `--no-home` only suppresses the initial homing call; the protocol's own `home:` steps still execute when not in `--mock`.

### Run flow

1. Load `configs/analysis.yaml` for the run-level settings (database path, wells, etc.).
2. Connect (or mock) the gantry; unlock GRBL.
3. Build the protocol from `configs/gantry/`, `configs/deck/`, `configs/protocol/` via `protocol_engine.setup_protocol`. Instruments are read from the `instruments:` block inside the gantry YAML.
4. Filter the plate to `wells.wells_to_test` from `analysis.yaml`.
5. Create a campaign row in SQLite (`paths.database`).
6. Connect the force sensor (real or mock), run the protocol, persist each well's measurement, disconnect.

### Troubleshooting

- **GRBL alarm on connect.** The runner sends `$X` (unlock) and `$H` (home). If the alarm persists, home manually in UGS and run with `--no-home`.
- **`struct.error: unpack requires a buffer of 18 bytes`.** GoDirect firmware/protocol mismatch. Run with `--skip-force-sensor` to keep the real gantry but mock force data, or `pip install -U godirect`, switch USB ports, and verify the sensor model.

## Configuration (`configs/`)

| File | Purpose |
|------|---------|
| `gantry/<gantry>.yaml` | Serial port, working volume, GRBL settings, homing strategy, **and mounted instruments** under the `instruments:` block |
| `deck/asmi_deck.yaml` | Plate type and per-well calibration anchors (A1, A2 → row/col offsets) |
| `protocol/<protocol>.yaml` | Sequence of CubOS commands (`home`, `scan`, `measure`, `move`, …) |
| `analysis.yaml` | Run-level settings: wells, contact/fit method, depth window, paths |

All three `gantry`/`deck`/`protocol` files are validated by CubOS Pydantic schemas with `extra="forbid"` — unknown keys cause a load error. (CubOS still supports a separate `board/<board>.yaml` for legacy setups, but new ASMI configs put instruments directly inside the gantry YAML.)

### Z-height convention (must match CubOS)

CubOS user-space coordinates are **positive Z = further above the deck**:

- `working_volume.z_max` is the home (top of travel).
- A labware's `z` is its reference height (well rim or sample surface) measured in the same user-space.
- `safe_approach_height` is a **positive offset above** the labware reference, used for XY travel between wells.
- `measurement_height` is the **signed offset** at which the instrument engages: `0` = touch the reference, `<0` = dip below it. Must satisfy `safe_approach_height ≥ measurement_height`.
- Indentation descends: Z decreases from `well_top_z` toward `z_limit`.

A typical mounted-instrument entry inside the gantry YAML looks like:

```yaml
# configs/gantry/<gantry>.yaml — bottom of file
instruments:
  asmi:
    type: asmi
    vendor: vernier            # required by CubOS schema
    offset_x: 0.0
    offset_y: 0.0
    depth: 0.0
    measurement_height: 0.0    # default action Z (protocol can override per-call)
    safe_approach_height: 3.0  # XY-travel clearance above the labware
    force_threshold: -50
    sensor_channels: [1]
```

### Indentation method kwargs

`scan` accepts only `plate`, `instrument`, `method`, `delay_s`, `method_kwargs`. Indentation parameters belong inside `method_kwargs`:

```yaml
# configs/protocol/asmi_indentation.yaml
positions:
  safe_z: [0.0, 0.0, 80.0]

protocol:
  - home:
  - scan:
      plate: plate
      instrument: asmi
      method: indentation
      method_kwargs:
        z_limit: 17.0          # absolute user-space Z to stop at (mm)
        step_size: 0.01        # mm per step
        force_limit: 10.0      # N — stop when |corrected force| exceeds
        baseline_samples: 10
        measure_with_return: false
        # measurement_height: 50.0  # optional override of well_top_z
  - move:
      instrument: asmi
      position: safe_z
      travel_z: 80.0
  - home:
```

`measurement_height` inside `method_kwargs` is interpreted by `ASMI.indentation` as the **absolute** Z to descend to before starting the indent (the "well top"); when omitted it falls back to the instrument's configured `measurement_height` after `approach_and_descend`.

## Heights when running `--skip-force-sensor`

`--skip-force-sensor` instantiates the ASMI driver with `offline=True`, so `measure()` returns synthetic zero-force readings, but the **gantry is real** and uses the configured Z values literally. Verify before running on the bench:

- `working_volume.z_max` ≥ deck `a1.z` + `safe_approach_height` + any `entry_travel_height` your protocol uses.
- `a1.z` is calibrated against the actual plate height with the indenter mounted (jog and read coords).
- `measurement_height` and `safe_approach_height` on the asmi `instruments:` entry produce a non-crashing approach (`a1.z + safe_approach_height` is above the rim, `a1.z + measurement_height` lands on/just into the sample).
- `method_kwargs.z_limit` does not exceed the gantry's reachable Z (it must stay within `working_volume.z_min`, given user-space convention).

The mock force sensor never raises a force-limit stop, so a misconfigured `z_limit` will drive the indenter into the plate. Always sanity-check Z values with `--mock` first, then with `--skip-force-sensor` at low feed rate before re-enabling the real sensor.

## Output

Per run, CubOS persists a campaign + experiment + measurement rows to the SQLite database at `paths.database` (default `data/asmi_data.db`). Inspect with the helpers under `scripts/`:

```bash
python3 scripts/view_asmi_db.py            # browse campaigns / experiments
python3 scripts/analyze_from_db.py         # pull measurements into the analysis pipeline
python3 scripts/check_grbl_settings.py     # dump live GRBL $-settings
```

Plots and CSVs from analysis land in `results/plots/<run_folder>/` and `results/measurements/<run_folder>/`.

## Batch analysis (legacy KABlab pipeline)

`main_asmi_2.py` runs the original KABlab Hertzian fit (baseline-threshold contact, geometry correction, iterative `d0` refinement) over an existing run folder:

```python
from main_asmi_2 import main

main(
    existing_run_folder="run_774_20260206_133925",
    p_ratio=0.5,
    baseline_points=10,
    save_plot=True,
    save_heatmap=True,
)
```

- `main_asmi_2.py` — entry point; parameters edited in the `main()` call.
- `src/analysis_batch_2.py` — original KABlab Hertzian fitting + geometry correction.
- `src/convert_measurement_format.py` — converts `Z / Raw_Force / Corrected_Force` columns to `well / depth / force`.

## Project layout

```
ASMI_new/
├── main_asmi.py                # CubOS-driven measurement entry point
├── main_asmi_2.py              # Legacy batch-analysis entry point
├── configs/
│   ├── analysis.yaml           # Run-level settings (wells, fit, paths)
│   ├── gantry/<gantry>.yaml    # CNC config + mounted `instruments:` block
│   ├── deck/asmi_deck.yaml
│   └── protocol/<protocol>.yaml
├── scripts/                    # DB inspection, GRBL helpers, backfill
├── src/
│   ├── analysis.py             # Analysis pipeline (extrapolation, retrospective, …)
│   ├── analysis_batch_2.py     # Legacy KABlab batch analysis
│   ├── convert_measurement_format.py
│   └── plot.py                 # Visualization
└── data/                       # SQLite database lives here
```

## License

MIT
