# ASMI - Automated Soft Matter Indenter

Automated system for high-throughput mechanical characterization of soft materials using CNC positioning and force sensors. Designed for hydrogel characterization, tissue engineering, and 96-well plate screening.

## Installation

```bash
pip install -r requirements.txt
```

**Hardware Requirements:**
- CNC machine with USB connection
- GoDirect force sensor (Vernier)

## Quick Start

Edit the `main()` function call at the bottom of `main_asmi.py` to configure your measurement parameters.

### Basic Measurement

```python
main(
    do_measure=True,
    wells_to_test=["A1", "A2", "B1"],
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
)
```

### Analyze Existing Data

```python
main(
    do_measure=False,
    existing_run_folder="run_732_20251030_122001",
    wells_to_test=["E5", "E6", "E7"],
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
)
```

## Main Function Parameters

### Essential Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `do_measure` | bool | `True` | `True` to measure, `False` to analyze existing data |
| `wells_to_test` | list[str] | `None` | List of wells (e.g., `["A1", "A2"]`) or `[None]` for current position |
| `contact_method` | str | `"retrospective"` | `"extrapolation"`, `"retrospective"`, or `"simple_threshold"` |
| `fit_method` | str | `"hertzian"` | `"hertzian"` for elastic modulus, `"linear"` for spring constant |
| `apply_system_correction` | bool | `True` | Apply system compliance correction for Hertzian fits |

### Measurement Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `z_target` | float | `-15.0` | Target indentation depth (mm) |
| `step_size` | float | `0.02` | Movement step size (mm) |
| `force_limit` | float | `5.0` | Maximum force limit (N) |
| `well_top_z` | float | `-9.0` | Well top position before indentation (mm), or `None` to use current Z |
| `measure_with_return` | bool | `False` | Enable return measurements (up/down) |
| `retrospective_threshold` | float | `None` | Force threshold for retrospective contact detection (N) |

### Analysis Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `existing_run_folder` | str | `None` | Folder name for existing data (e.g., `"run_732_20251030_122001"`) |
| `existing_measured_with_return` | bool | `True` | Whether existing data has return measurements |
| `generate_heatmap` | bool | `True` | Generate 96-well plate heatmaps after analysis |

### Advanced Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `home_before_measure` | bool | `True` | Home CNC before measurements |
| `move_to_pickup` | bool | `False` | Move to pickup position after measurements |
| `pickup_position` | tuple | `(0.0, 0.0, 0.0)` | XYZ coordinates for pickup position (mm) |
| `lock_xy_single_spot` | bool | `False` | Lock XY position for single spot measurements |
| `lock_xy_position` | tuple | `None` | Specific XY coordinates to lock (mm) |
| `cnc` | CNCController | `None` | Pre-initialized CNC controller (auto-created if `None`) |
| `force_sensor` | ForceSensor | `None` | Pre-initialized force sensor (auto-created if `None`) |

## Usage Examples

### Example 1: Measure Specific Wells

```python
main(
    do_measure=True,
    wells_to_test=["E5", "E6", "E7"],
    contact_method="retrospective",
    retrospective_threshold=1.0,
    fit_method="hertzian",
    apply_system_correction=True,
    z_target=-15.0,
    step_size=0.01,
    force_limit=10.0,
    well_top_z=-9.0,
)
```

### Example 2: Analyze Existing Data with System Correction

```python
main(
    do_measure=False,
    existing_run_folder="run_732_20251030_122001",
    wells_to_test=["E5", "E6", "E7"],
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
    generate_heatmap=True,
)
```

### Example 3: Measure with Return (Up/Down)

```python
main(
    do_measure=True,
    wells_to_test=["A1", "A2"],
    measure_with_return=True,
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
)
```

### Example 4: Measure System Compliance (Linear Fit)

```python
main(
    do_measure=True,
    wells_to_test=[None],  # Current position
    contact_method="retrospective",
    retrospective_threshold=13.0,  # High threshold for system compliance
    fit_method="linear",  # Linear fit for spring constant
    z_target=-90.0,
    force_limit=20.0,
    well_top_z=-80.0,
    lock_xy_single_spot=True,
    lock_xy_position=(-120, -40.0),
)
```

## Output Files

Results are saved in `results/`:

```
results/
├── measurements/
│   └── run_XXX_YYYYMMDD_HHMMSS/
│       └── well_*.csv              # Raw measurement data
└── plots/
    └── run_XXX_YYYYMMDD_HHMMSS/
        ├── *_analysis*.png         # Analysis plots with fits
        ├── *_contact_detection*.png # Contact point detection
        ├── well_heatmap_original.png # Original E values
        ├── well_heatmap_corrected.png # System-corrected E values
        └── summary.csv              # Summary data for heatmaps
```

## Contact Detection Methods

- **`"retrospective"`**: Analyzes force data retrospectively from a threshold (recommended)
- **`"extrapolation"`**: Linear extrapolation from force threshold
- **`"simple_threshold"`**: Simple force threshold detection

## Fitting Methods

- **`"hertzian"`**: Calculates elastic modulus using Hertzian contact mechanics
  - With `apply_system_correction=True`: Generates both original and corrected fits/heatmaps
- **`"linear"`**: Calculates spring constant using linear fit (F = k * d)

## Project Structure

```
ASMI_new/
├── main_asmi.py              # Main entry point - edit parameters here
├── src/
│   ├── CNCController.py      # CNC machine control
│   ├── ForceSensor.py        # Force sensor interface
│   ├── ForceMonitoring.py    # Measurement protocols
│   ├── analysis.py           # Data analysis and fitting
│   └── plot.py               # Visualization functions
├── cad/                      # CAD files (STEP/STL)
└── results/                  # Output data and plots
```

## License

MIT License - Copyright (c) 2025 Hongrui Zhang
