# ASMI - Automated Sample Measurement Interface

An automated system for measuring samples using CNC positioning and force sensors.

## Project Structure

```
ASMI/
├── src/                    # Source code
│   ├── __init__.py
│   ├── CNCController.py    # CNC machine control
│   ├── ForceSensor.py      # Force sensor interface
│   └── analysis.py         # Real-time data analysis (to be added)
├── tests/                  # Test files
│   ├── __init__.py
│   ├── test_cnc_controller.py
│   └── test_force_sensor.py
├── main_asmi.py           # Main entry point
└── README.md              # This file
```

## Installation

1. Install required packages:
```bash
pip install pyserial godirect
```

2. Ensure your CNC machine is connected via USB and the force sensor is plugged in.

## Usage

Run the main program:
```bash
python main_asmi.py
```

## Testing

Run all tests:
```bash
python -m unittest discover tests
```

Run specific test:
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

### Analysis (to be implemented)
- Real-time data processing
- Statistical analysis
- Data visualization

## Configuration

Edit the configuration constants in each module:
- `CNCController.py`: Port, baud rate, well plate geometry
- `ForceSensor.py`: Threshold values, sampling parameters 