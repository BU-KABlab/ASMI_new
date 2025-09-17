# ASMI Source Code Documentation

This directory contains the core modules of the Automated Soft Matter Indenter (ASMI) system. Each module is designed to handle specific aspects of the measurement and analysis workflow.

## Module Overview

### Core Modules

#### `CNCController.py`
**Purpose**: Controls the CNC machine for precise positioning and movement operations.

**Key Features**:
- Serial communication with CNC machine
- G-code command generation and execution
- Well plate positioning (96-well format)
- Homing and safety operations
- Position tracking and validation

**Main Classes**:
- `CNCController`: Main controller class for CNC operations

**Key Methods**:
- `home()`: Home all axes with optional zeroing
- `move_to_well(col, row, z)`: Move to specific well position
- `move_to_z(z)`: Move Z-axis to specified position
- `get_current_position()`: Get current XYZ coordinates
- `wait_for_idle()`: Wait for machine to complete operations

**Dependencies**: `serial`, `time`, `csv`

---

#### `ForceSensor.py`
**Purpose**: Interfaces with GoDirect force sensors for real-time force measurements.

**Key Features**:
- GoDirect sensor communication
- Force calibration and baseline correction
- Real-time data acquisition
- Sensor status monitoring

**Main Classes**:
- `ForceSensor`: Main interface for force sensor operations

**Key Methods**:
- `get_force_reading()`: Get current force measurement
- `calibrate()`: Calibrate sensor baseline
- `is_connected()`: Check sensor connection status

**Dependencies**: `godirect`, `time`

---

#### `analysis.py`
**Purpose**: Performs data analysis using Hertzian contact mechanics and provides visualization.

**Key Features**:
- Hertzian contact mechanics fitting
- Multiple contact detection methods
- Elastic modulus calculation
- R² quality assessment
- Data filtering and preprocessing

**Main Classes**:
- `IndentationAnalyzer`: Main analysis class

**Key Methods**:
- `analyze_well()`: Analyze a single well's data
- `find_extraploation_contact_point()`: Extrapolation-based contact detection
- `fit_hertz_model()`: Fit Hertzian model to force-depth data
- `plot_contact_detection()`: Visualize contact detection
- `plot_results()`: Plot analysis results

**Contact Detection Methods**:
- **Extrapolation**: Linear extrapolation from force threshold
- **Retrospective**: Analysis of entire force curve
- **Simple Threshold**: Basic force threshold detection

**Dependencies**: `numpy`, `scipy`, `matplotlib`, `pandas`

---

#### `force_monitoring.py`
**Purpose**: Implements measurement protocols and data acquisition workflows.

**Key Features**:
- Automated indentation measurements
- Return measurement protocols
- Force limit monitoring
- Data logging and CSV output
- Safety monitoring

**Main Functions**:
- `simple_indentation_measurement()`: Single-direction indentation
- `simple_indentation_with_return_measurement()`: Bidirectional measurement
- `get_and_increment_run_count()`: Run counter management

**Measurement Parameters**:
- `z_target`: Target indentation depth
- `step_size`: Movement step size
- `force_limit`: Maximum allowed force
- `well_top_z`: Well top position before indentation

**Dependencies**: `time`, `csv`, `os`, `datetime`

---

#### `plot.py`
**Purpose**: Provides comprehensive visualization capabilities for measurement data and analysis results.

**Key Features**:
- Raw data visualization
- Contact detection plots
- Analysis result plots
- Heatmap generation
- Directional data plotting

**Main Classes**:
- `Plotter`: Main plotting interface

**Key Methods**:
- `plot_raw_data_all_wells()`: Plot all wells' raw data
- `plot_raw_force_individual_wells()`: Individual well force plots
- `plot_contact_detection()`: Contact point visualization
- `plot_results()`: Analysis results visualization
- `plot_well_heatmap()`: 96-well plate heatmaps

**Plot Types**:
- Raw force vs position
- Contact detection with thresholds
- Fitted curves with R² values
- Elastic modulus heatmaps
- Directional analysis plots

**Dependencies**: `matplotlib`, `numpy`, `pandas`

---

## Data Flow

```
1. CNCController.py
   ↓ (positioning)
2. ForceSensor.py
   ↓ (force data)
3. force_monitoring.py
   ↓ (measurement data)
4. analysis.py
   ↓ (analysis results)
5. plot.py
   ↓ (visualizations)
```

## Configuration Files

### `run_count.txt`
- Stores the current run counter
- Automatically incremented for each measurement batch
- Used for unique file naming

### `last_position.csv`
- Stores the last known CNC position
- Used for position recovery and validation
- Updated after each movement operation

## Dependencies

### Required Python Packages
```bash
pip install pyserial godirect numpy scipy matplotlib pandas
```

### Hardware Requirements
- CNC machine with USB serial connection
- GoDirect force sensor (Vernier)
- 96-well plate or compatible sample holder

## Usage Patterns

### Basic Measurement Workflow
1. Initialize `CNCController` and `ForceSensor`
2. Home the CNC machine
3. Call measurement functions from `force_monitoring.py`
4. Analyze data using `IndentationAnalyzer`
5. Generate plots using `Plotter`

### Analysis-Only Workflow
1. Load existing data using `IndentationAnalyzer`
2. Perform analysis with desired contact detection method
3. Generate visualizations using `Plotter`

### Batch Processing
1. Iterate through multiple wells
2. Perform measurements for each well
3. Aggregate results for heatmap generation
4. Generate comprehensive visualizations

## Error Handling

Each module includes comprehensive error handling:
- Serial communication errors
- Sensor connection issues
- Data validation errors
- File I/O errors
- Hardware timeout errors

## Performance Considerations

- **Serial Communication**: Optimized for minimal delays
- **Data Processing**: Efficient NumPy operations
- **Memory Management**: Streaming data processing for large datasets
- **Plotting**: Lazy loading and efficient rendering

## Testing

Unit tests are available in the `tests/` directory:
- `test_cnc_controller.py`: CNC control functionality
- `test_force_sensor.py`: Force sensor operations

## Extensibility

The modular design allows for easy extension:
- New contact detection methods in `analysis.py`
- Additional measurement protocols in `force_monitoring.py`
- Custom visualization types in `plot.py`
- New hardware interfaces following existing patterns

## Maintenance

### Regular Tasks
- Update run counter if needed
- Verify sensor calibration
- Check CNC positioning accuracy
- Monitor data file sizes

### Troubleshooting
- Check serial connections
- Verify sensor communication
- Validate data file integrity
- Review error logs

## Version History

### v2.0
- Enhanced contact detection methods
- Return measurement support
- Improved visualization
- Better error handling

### v1.0
- Initial implementation
- Basic measurement capabilities
- Core analysis functions