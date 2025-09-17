# ASMI - Automated Soft Matter Indenter

A comprehensive automated system for measuring soft matter samples using CNC positioning and force sensors. The system provides precise indentation measurements with real-time force monitoring, data analysis, and visualization capabilities.

## Overview

The Automated Soft Matter Indenter (ASMI) is designed for high-throughput mechanical characterization of soft materials, particularly useful for:
- Hydrogel characterization
- Tissue engineering applications
- Soft material property analysis
- 96-well plate screening
- Time-series measurements

## Key Features

- **Automated CNC Control**: Precise positioning across 96-well plates
- **Real-time Force Monitoring**: Continuous force measurement during indentation
- **Multiple Contact Detection Methods**: Extrapolation, retrospective, and threshold-based
- **Bidirectional Measurements**: Both downward indentation and upward return measurements
- **Comprehensive Data Analysis**: Hertzian contact mechanics fitting with R² analysis
- **Advanced Visualization**: Raw data plots, contact detection plots, analysis results, and heatmaps
- **Flexible Workflows**: Measurement, analysis-only, or scheduled measurement modes
- **Data Management**: Automatic CSV splitting for directional measurements

## Project Structure

```
ASMI_new/
├── main_asmi.py              # Main entry point with parameter-based interface
├── src/                      # Core source code modules
│   ├── __init__.py
│   ├── CNCController.py      # CNC machine control and positioning
│   ├── ForceSensor.py        # Force sensor interface and calibration
│   ├── analysis.py           # Data analysis and Hertzian fitting
│   ├── force_monitoring.py   # Force measurement protocols
│   ├── plot.py               # Visualization and plotting functions
│   ├── run_count.txt         # Run counter for file naming
│   └── last_position.csv     # Last known CNC position
├── results/                  # Data output (local only, not in repo)
│   ├── measurements/         # Raw measurement data
│   └── plots/               # Generated plots and analysis results
├── archived/                 # Legacy code (local only)
├── tests/                    # Unit tests
└── README.md                 # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- CNC machine with USB connection
- GoDirect force sensor (Vernier)

### Dependencies
```bash
pip install pyserial godirect numpy scipy matplotlib pandas
```

### Hardware Setup
1. Connect CNC machine via USB
2. Connect GoDirect force sensor
3. Ensure proper power supply and grounding
4. Calibrate force sensor before first use

## Quick Start

### Basic Measurement
```python
from main_asmi import main

# Measure wells A1 and A2 with default settings
main(do_measure=True, wells_to_test=["A1", "A2"])

# Measure with return measurement (up/down)
main(do_measure=True, wells_to_test=["A1", "A2"], measure_with_return=True)
```

### Analysis of Existing Data
```python
# Analyze existing measurement data
main(do_measure=False, existing_run_folder="run_463_20250917_000017")
```

### Scheduled Measurements
```python
from main_asmi import measure_at_intervals

# Measure every hour for 24 hours
measure_at_intervals(
    interval_seconds=3600,  # 1 hour
    cycles=24,
    wells_to_test=["A1", "A2", "B1", "B2"]
)
```

## Usage Examples

### Parameter Configuration
```python
main(
    do_measure=True,                    # Enable measurement
    wells_to_test=["A1", "A2", "B1"],  # Wells to measure
    contact_method="extrapolation",     # Contact detection method
    measure_with_return=True,           # Enable return measurements
    z_target=-15.0,                     # Target indentation depth (mm)
    step_size=0.01,                     # Step size (mm)
    force_limit=5.0,                    # Force limit (N)
    well_top_z=-9.0,                    # Well top position (mm)
    generate_heatmap=True               # Generate heatmaps
)
```

### Contact Detection Methods
- **`"extrapolation"`**: Linear extrapolation from force threshold (default)
- **`"retrospective"`**: Retrospective analysis of force data
- **`"simple_threshold"`**: Simple force threshold detection

### Measurement Modes
1. **Simple Indentation**: Single downward measurement
2. **Return Measurement**: Both downward and upward measurements with directional analysis

## Data Output

### File Structure
```
results/
├── measurements/
│   └── run_XXX_YYYYMMDD_HHMMSS/
│       ├── well_A1_YYYYMMDD_HHMMSS.csv
│       ├── well_A1_YYYYMMDD_HHMMSS_down.csv  # Return measurements
│       └── well_A1_YYYYMMDD_HHMMSS_up.csv
└── plots/
    └── run_XXX_YYYYMMDD_HHMMSS/
        ├── well_heatmap_down.png
        ├── well_heatmap_up.png
        ├── A1_analysis_extrapolation_down.png
        └── A1_contact_detection_extrapolation_down.png
```

### Data Format
CSV files contain:
- Timestamp, Z position, Raw force, Corrected force
- Direction (for return measurements)
- Metadata (measurement parameters, total time)

## Analysis Features

### Hertzian Contact Mechanics
- Automatic fitting of force-depth curves
- Elastic modulus calculation
- R² quality assessment
- Uncertainty estimation

### Visualization
- Raw force vs position plots
- Contact detection visualization
- Analysis results with fitted curves
- 96-well plate heatmaps
- Directional analysis plots

### Data Processing
- Automatic baseline correction
- Direction-based data splitting
- Contact point detection
- Curve fitting with bounds

## Safety Features

- Force limit monitoring
- Emergency stop capabilities
- Automatic homing before/after measurements
- Position validation
- Error handling and recovery

## Configuration

### CNC Settings
Edit `src/CNCController.py`:
- Serial port configuration
- Well plate geometry
- Movement parameters

### Force Sensor Settings
Edit `src/ForceSensor.py`:
- Calibration parameters
- Sampling rates
- Threshold values

### Analysis Parameters
Edit `src/analysis.py`:
- Fitting bounds
- Contact detection thresholds
- Analysis ranges

## Troubleshooting

### Common Issues
1. **Serial Connection Error**: Check USB connection and port settings
2. **Force Sensor Not Found**: Verify GoDirect sensor connection
3. **Homing Failed**: Check CNC power and limit switches
4. **Import Errors**: Install missing dependencies

### Debug Mode
Enable verbose output by modifying print statements in the source code.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this software in your research, please cite:

```
ASMI - Automated Soft Matter Indenter
Hongrui Zhang, 2025
```

## Support

For technical support or questions:
- Check the troubleshooting section
- Review the source code documentation
- Open an issue on GitHub

## Changelog

### Version 2.0
- Parameter-based interface
- Return measurement support
- Enhanced visualization
- Improved data management
- Multiple contact detection methods

### Version 1.0
- Initial release
- Basic measurement capabilities
- CNC control integration
- Force sensor interface