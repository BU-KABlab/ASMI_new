# ASMI - Automated Soft Matter Indenter

An automated system for measuring samples using CNC positioning and force sensors.

## Project Structure

```
ASMI_new/
├── src/                    # Source code
│   ├── __init__.py
│   ├── CNCController.py    # CNC machine control
│   ├── ForceSensor.py      # Force sensor interface
│   ├── analysis.py         # Data analysis and visualization
│   └── force_monitoring.py # Force monitoring module
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_cnc_controller.py
│   └── test_force_sensor.py
├── test_force_monitoring.py           # Original comprehensive test script
├── test_force_monitoring_simple.py    # Simple force monitoring test
├── test_force_monitoring_advanced.py  # Advanced force monitoring test with CLI
├── main_asmi.py           # Main entry point
├── FORCE_MONITORING_README.md # Force monitoring documentation
└── README.md              # This file
```

## Installation

1. Install required packages:
```bash
pip install pyserial godirect
```

2. Ensure your CNC machine is connected via USB and the force sensor is plugged in.

## Usage

### Main Program
Run the main program:
```bash
python main_asmi.py
```

### Force Monitoring Tests

#### Simple Force Monitoring Test
For quick testing with predefined wells:
```bash
python test_force_monitoring_simple.py
```
Edit the script to modify `wells_to_test` and other parameters.

#### Advanced Force Monitoring Test
For flexible testing with command-line arguments:
```bash
# Test specific wells
python test_force_monitoring_advanced.py --wells A1,A2,B1,B2

# Test with custom parameters
python test_force_monitoring_advanced.py --wells A6,B6,C6 --target-z -20 --force-limit 50

# Test with high precision
python test_force_monitoring_advanced.py --wells A5,B5,C5 --period-ms 5 --feedrate 100
```

#### Original Comprehensive Test
For the full test suite:
```bash
python test_force_monitoring.py
```

## Testing

Run all unit tests:
```bash
python -m unittest discover tests
```

Run specific unit test:
```bash
python tests/test_force_sensor.py
python tests/test_cnc_controller.py
```

## Components

### CNCController
- Controls CNC machine movement
- Handles well plate positioning
- Manages G-code commands

### ForceSensor
- Interfaces with GoDirect force sensors
- Provides force measurements
- Handles sensor calibration

### Analysis
- Real-time data processing
- Statistical analysis
- Data visualization
- Force vs Z position analysis

### Force Monitoring
- Automated well testing with force limits
- Safety monitoring and emergency stops
- Batch processing of multiple wells
- Comprehensive data logging and analysis

## Configuration

Edit the configuration constants in each module:
- `CNCController.py`: Port, baud rate, well plate geometry
- `ForceSensor.py`: Threshold values, sampling parameters
- `force_monitoring.py`: Default force limits, sampling periods, feedrates

## Documentation

- `FORCE_MONITORING_README.md` - Detailed documentation for force monitoring tests
- `results/README.md` - Information about data output and file structure 