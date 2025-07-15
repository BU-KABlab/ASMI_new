# Results Folder

This folder contains all measurement data and analysis results from the ASMI system.

## Folder Structure

```
results/
├── force_measurements/     # Force vs Z position data
├── well_measurements/      # Measurements at specific wells
├── calibration/           # Calibration data
├── analysis/              # Analysis results and plots
└── README.md             # This file
```

## File Naming Convention

- Force measurements: `force_vs_z_YYYYMMDD_HHMMSS.csv`
- Well measurements: `well_A1_YYYYMMDD_HHMMSS.csv`
- Calibration data: `calibration_YYYYMMDD_HHMMSS.csv`

## Data Format

### Force vs Z Measurements
- Timestamp(s): Time since start of measurement
- Z_Position(mm): Current Z-axis position
- Force(N): Force reading from sensor

### Well Measurements
- Well: Well identifier (e.g., A1, B2)
- Timestamp: Measurement timestamp
- Force(N): Force reading
- Z_Position(mm): Z position when measurement taken 