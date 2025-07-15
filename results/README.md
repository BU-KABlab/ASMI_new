# Results Folder

This folder contains all measurement data and analysis results from the ASMI system.

## Folder Structure

```
results/
├── force_measurements/     # Force vs Z position data
├── well_measurements/      # Measurements at specific wells
│   ├── run_001/           # Individual test runs
│   ├── run_002/
│   └── ...
├── calibration/           # Calibration data
├── plots/                 # Analysis plots and visualizations
│   ├── run_007_20250711_053933/  # Plots for specific runs
│   └── ...
└── README.md             # This file
```

## File Naming Convention

### Force Measurements
- Force vs Z data: `force_vs_z_YYYYMMDD_HHMMSS.csv`
- Force cycle data: `force_cycle_YYYYMMDD_HHMMSS.csv`

### Well Measurements
- Individual well data: `well_A1_YYYYMMDD_HHMMSS.csv`
- Run summary: `well_summary_XXX_YYYYMMDD_HHMMSS.csv`
- Run folders: `run_XXX_YYYYMMDD_HHMMSS/`

### Calibration Data
- Calibration data: `calibration_YYYYMMDD_HHMMSS.csv`

### Analysis Plots
- Well analysis plots: `A5_analysis.png`, `B6_analysis.png`, etc.
- Run-specific plot folders: `run_XXX_YYYYMMDD_HHMMSS/`

## Data Format

### Force vs Z Measurements
- Timestamp(s): Time since start of measurement
- Z_Position(mm): Current Z-axis position
- Force(N): Force reading from sensor

### Well Measurements
Each well measurement file contains:
- **Metadata**: Test time, well identifier, target Z, force limit, sampling parameters
- **Force exceeded**: Whether force limit was exceeded
- **Stop Z**: Z position where movement stopped (if force limit exceeded)
- **Measurements**: Timestamp, Z position, force, movement direction (Down/Up)

### Run Summary Files
Contains results for all wells in a test run:
- Overall test parameters
- Results for each well (force exceeded, stop Z, max force, sample count)
- File paths to individual well data

### Analysis Plots
- Force vs Z position plots for each well
- Movement direction indicators
- Force limit thresholds
- Statistical analysis results

## Force Monitoring Tests

The force monitoring system generates organized data in the `well_measurements/` folder:

### Test Run Structure
Each test run creates a folder like `run_007_20250711_053933/` containing:
- Individual well data files (`well_A5_20250711_054103.csv`)
- Summary file (`well_summary_007_20250711_053933.csv`)

### Data Analysis
- Use `analyze_run_007.py` to analyze specific runs
- Use `analyze_well_run.py` for well-specific analysis
- Plots are automatically generated in `plots/` folder

### Safety Features
- Force limit monitoring with automatic stop
- Emergency stop procedures
- Stuck position detection
- Comprehensive logging of all events 