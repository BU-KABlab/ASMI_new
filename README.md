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

**Important:** Before calling `main()`, you must create or comment out the hardware objects:

### For Measurements (Hardware Required)

```python
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor

# Create hardware objects
cnc = CNCController()
force_sensor = ForceSensor()

main(
    cnc=cnc,
    force_sensor=force_sensor,
    do_measure=True,
    wells_to_test=["A1", "A2", "B1"],
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
)
```

### For Analysis Only (No Hardware Connection)

```python
# Comment out hardware objects to avoid connection issues
# from src.CNCController import CNCController
# from src.ForceSensor import ForceSensor
# cnc = CNCController()
# force_sensor = ForceSensor()

main(
    cnc=None,  # Must be None for analysis-only
    force_sensor=None,  # Must be None for analysis-only
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
| `contact_method` | str | `"retrospective"` | `"extrapolation"`, `"retrospective"`, `"simple_threshold"`, or `"baseline_threshold"` |
| `fit_method` | str | `"hertzian"` | `"hertzian"` for elastic modulus, `"linear"` for spring constant |
| `apply_system_correction` | bool | `True` | Apply system compliance correction for Hertzian fits |

### Measurement Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `z_target` | float | `-15.0` | Target indentation depth (mm). **Measurement stops when either `z_target` or `force_limit` is reached** |
| `step_size` | float | `0.02` | Movement step size (mm) |
| `force_limit` | float | `5.0` | Maximum force limit (N). **Measurement stops when either `force_limit` or `z_target` is reached** |
| `well_top_z` | float | `-9.0` | Well top position before indentation (mm), or `None` to use current Z |
| `measure_with_return` | bool | `False` | Enable return measurements (up/down) |
| `retrospective_threshold` | float | `None` | Force threshold for retrospective contact detection (N) |

**Note:** Measurements automatically stop when either the `force_limit` (N) or `z_target` (mm) is reached, whichever comes first.

### Analysis Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `existing_run_folder` | str | `None` | Folder name for existing data (e.g., `"run_732_20251030_122001"`) |
| `existing_measured_with_return` | bool | `True` | Whether existing data has return measurements |
| `generate_heatmap` | bool | `True` | Generate 96-well plate heatmaps after analysis |
| `max_depth` | float | `0.5` | Maximum indentation depth (mm) used for fitting |
| `min_depth` | float | `0.25` | **Hertzian only:** Minimum depth (mm); 0.25 = legacy 0.25–0.5 mm range. Linear always uses 0–max_depth |
| `apply_force_correction` | bool | `False` | **Hertzian only:** Apply geometry-based force correction (KABlab legacy) before fit |
| `iterative_d0_refinement` | bool | `False` | **Hertzian only:** Iterative d0 refinement until \|d0\| < 0.01 mm (KABlab legacy) |
| `well_bottom_z` | float | `-85.0` | Well bottom Z (mm); sample height = \|contact_z - well_bottom_z\| |
| `poisson_ratio` | float | `None` | Sample Poisson's ratio; `None` = auto-detect from filename (e.g., 0.5 for hydrogel, 0.3 for glassy) |
| `use_legacy_height` | bool | `False` | Use original batch script approx_height for (b,c) lookup (match original E) |
| `legacy_height_step_mm` | float | `0.02` | Step size (mm) for legacy height formula; match `step_size` when measuring |
| `marker_scale` | float | `1.0` | Scale factor for plot marker sizes; use >1 when resizing in PowerPoint |

### Advanced Parameters

| Parameter | Type | Default | Description |
|----------|------|---------|-------------|
| `home_before_measure` | bool | `True` | Home CNC before measurements |
| `move_to_pickup` | bool | `False` | Move to pickup position after measurements |
| `pickup_position` | tuple | `(0.0, 0.0, 0.0)` | XYZ coordinates for pickup position (mm) |
| `lock_xy_single_spot` | bool | `False` | Lock XY position for single spot measurements |
| `lock_xy_position` | tuple | `None` | Specific XY coordinates to lock (mm) |
| `cnc` | CNCController | `None` | **For measurements:** Create `CNCController()` object. **For analysis-only:** Must be `None` to avoid connection errors |
| `force_sensor` | ForceSensor | `None` | **For measurements:** Create `ForceSensor()` object. **For analysis-only:** Must be `None` to avoid connection errors |

## Usage Examples

### Example 1: Measure Specific Wells

```python
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor

cnc = CNCController()
force_sensor = ForceSensor()

main(
    cnc=cnc,
    force_sensor=force_sensor,
    do_measure=True,
    wells_to_test=["E5", "E6", "E7"],
    contact_method="retrospective",
    retrospective_threshold=0.05,
    fit_method="hertzian",
    apply_system_correction=True,
    z_target=-80.0,
    step_size=0.01,
    force_limit=10.0,
    well_top_z=-70.0,
)
```

### Example 2: Analyze Existing Data with System Correction

```python
# Comment out hardware objects for analysis-only
# from src.CNCController import CNCController
# from src.ForceSensor import ForceSensor
# cnc = CNCController()
# force_sensor = ForceSensor()

main(
    cnc=None,  # Must be None to avoid connection attempts
    force_sensor=None,  # Must be None to avoid connection attempts
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
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor

cnc = CNCController()
force_sensor = ForceSensor()

main(
    cnc=cnc,
    force_sensor=force_sensor,
    do_measure=True,
    wells_to_test=["A1", "A2"],
    measure_with_return=True,
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=True,
)
```

### Example 4: Hertzian with KABlab Legacy Options (PDMS, etc.)

```python
main(
    do_measure=False,
    existing_run_folder="run_737_20260209_204208",
    wells_to_test=["E5", "E6", "E7"],
    contact_method="retrospective",
    fit_method="hertzian",
    apply_system_correction=False,
    min_depth=0.25,
    max_depth=0.5,
    apply_force_correction=True,
    iterative_d0_refinement=True,
    well_bottom_z=-85.0,
    poisson_ratio=0.5,
)
```

### Example 5: Measure System Compliance (Linear Fit)

```python
from src.CNCController import CNCController
from src.ForceSensor import ForceSensor

cnc = CNCController()
force_sensor = ForceSensor()

main(
    cnc=cnc,
    force_sensor=force_sensor,
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

### Example 6: Plot Customization for PowerPoint

```python
main(
    do_measure=False,
    existing_run_folder="run_737_...",
    marker_scale=1.5,  # Larger markers/lines for resizing in PPT
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
- **`"baseline_threshold"`**: Original KABlab formula: threshold = -baseline + 2×std

## Fitting Methods

- **`"hertzian"`**: Calculates elastic modulus using Hertzian contact mechanics
  - With `apply_system_correction=True`: Generates both original and corrected fits/heatmaps
  - `min_depth`, `max_depth`: Control depth range for fit (default 0.25–0.5 mm)
  - `apply_force_correction`: Geometry correction for finite-height samples (KABlab legacy)
  - `iterative_d0_refinement`: Iterative d0 refinement until convergence (KABlab legacy)
- **`"linear"`**: Calculates spring constant using linear fit (F = k * d); always uses 0–max_depth

## Sample Height

Sample height is computed as **\|contact_z - well_bottom_z\|**. Set `well_bottom_z` (default -85 mm) to match your plate geometry. Used for geometry force correction lookup and summary output.

## Batch Analysis (Original KABlab Pipeline)

For analysis using the original batch script pipeline (baseline_threshold contact, geometry correction, iterative d0):

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

- **`main_asmi_2.py`**: Entry point for batch analysis; uses `src/analysis_batch_2.py`
- **`src/analysis_batch_2.py`**: Original KABlab Hertzian fitting with geometry correction
- **`src/convert_measurement_format.py`**: Converts Z/Raw_Force/Corrected_Force → well/depth/force
- Output: `results/plots/<run_folder>/` (heatmap with E, ±std, R² per well)

## Project Structure

```
ASMI_new/
├── main_asmi.py              # Main entry point - measure & analyze (src/analysis)
├── main_asmi_2.py            # Batch analysis entry - original KABlab pipeline
├── src/
│   ├── CNCController.py      # CNC machine control
│   ├── ForceSensor.py        # Force sensor interface
│   ├── ForceMonitoring.py    # Measurement protocols
│   ├── analysis.py           # Data analysis and fitting (extrapolation, retrospective, etc.)
│   ├── analysis_batch_2.py   # Original KABlab batch analysis
│   ├── convert_measurement_format.py  # Format conversion for batch pipeline
│   └── plot.py               # Visualization functions
├── cad/                      # CAD files (STEP/STL)
└── results/                  # Output data and plots
```

## License

MIT License - Copyright (c) 2025 Hongrui Zhang
