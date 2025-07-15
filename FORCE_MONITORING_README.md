# Force Monitoring Test Scripts

This directory contains refactored force monitoring test scripts that make it easy to test force monitoring at multiple wells with just a simple input.

## Files

- `src/force_monitoring.py` - Main force monitoring module with all the core functionality
- `test_force_monitoring_simple.py` - Simple test script (just modify wells_to_test)
- `test_force_monitoring_advanced.py` - Advanced test script with command-line arguments
- `test_force_monitoring.py` - Original comprehensive test script (kept for reference)

## Quick Start

### Option 1: Simple Script (Recommended for beginners)

1. Edit `test_force_monitoring_simple.py` and modify the `wells_to_test` list:

```python
wells_to_test = ["A6", "B6", "C6", "C5", "B5", "A5"]
```

2. Optionally adjust other parameters:

```python
target_z = -15.0      # Target Z position in mm
force_limit = 45.0    # Force limit in N
period_ms = 10        # Sampling period in milliseconds
feedrate = 200        # Movement feedrate
```

3. Run the script:

```bash
python test_force_monitoring_simple.py
```

### Option 2: Advanced Script (Recommended for power users)

Use command-line arguments for flexible testing:

```bash
# Test specific wells
python test_force_monitoring_advanced.py --wells A1,A2,B1,B2

# Test with custom parameters
python test_force_monitoring_advanced.py --wells A6,B6,C6 --target-z -20 --force-limit 50

# Test with high precision
python test_force_monitoring_advanced.py --wells A5,B5,C5 --period-ms 5 --feedrate 100

# Test all wells in a pattern
python test_force_monitoring_advanced.py --wells A1,A2,A3,A4,A5,A6,B1,B2,B3,B4,B5,B6,C1,C2,C3,C4,C5,C6

# Skip confirmation prompt
python test_force_monitoring_advanced.py --wells A1,A2 --no-confirm
```

## Command Line Options (Advanced Script)

- `--wells`: Comma-separated list of wells to test (default: A6,B6,C6,C5,B5,A5)
- `--target-z`: Target Z position in mm (default: -15.0)
- `--force-limit`: Force limit in N (default: 45.0)
- `--period-ms`: Sampling period in milliseconds (default: 10)
- `--feedrate`: Movement feedrate (default: 200)
- `--no-confirm`: Skip confirmation prompt

## Output

The scripts will:

1. Create a new run folder in `results/well_measurements/run_XXX_YYYYMMDD_HHMMSS/`
2. Save individual well data files (e.g., `well_A1_20250711_123456.csv`)
3. Create a summary file (`well_summary_XXX_YYYYMMDD_HHMMSS.csv`)
4. Display real-time progress and results

## Example Output Structure

```
results/well_measurements/run_007_20250711_053933/
├── well_A5_20250711_054103.csv
├── well_A6_20250711_053955.csv
├── well_B5_20250711_054049.csv
├── well_B6_20250711_054009.csv
├── well_C5_20250711_054036.csv
├── well_C6_20250711_054023.csv
└── well_summary_007_20250711_053933.csv
```

## Safety Features

- **Force Limit Monitoring**: Automatically stops movement when force exceeds the limit
- **Emergency Stop**: Uses feed hold and machine unlock procedures
- **Stuck Detection**: Detects when Z position gets stuck and returns to safety
- **Confirmation Prompt**: Asks for user confirmation before starting (can be skipped with `--no-confirm`)

## Troubleshooting

### Force Sensor Not Connected
```
❌ Force sensor not connected. Please check USB connection.
```
- Check USB connection to force sensor
- Ensure force sensor is powered on
- Verify device permissions

### Machine Communication Issues
```
❌ Could not get current position
```
- Check serial connection to CNC machine
- Verify machine is powered on and in idle state
- Check baud rate and port settings in CNCController

### Force Limit Exceeded
```
⚠️ Force limit exceeded: 47.2N > 45.0N at Z=-12.3mm
```
- This is normal behavior - the system detected excessive force and stopped safely
- Check if there's an obstacle or if the force limit is too low
- Review the data files to understand the force profile

## Data Analysis

Each well's CSV file contains:
- Test metadata (time, well, parameters)
- Timestamp, Z position, and force measurements
- Movement direction (Down/Up)

The summary file contains:
- Overall test parameters
- Results for each well (force exceeded, stop Z, max force, etc.)

## Integration with Analysis

The output files are compatible with the existing analysis scripts in the `src/` folder and can be processed using the analysis tools. 