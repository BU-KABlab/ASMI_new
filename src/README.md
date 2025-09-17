# ASMI Contact Point Detection and Analysis

This document explains how contact point detection works in the ASMI system, including both real-time detection during measurement and post-processing analysis.

## Overview

Contact point detection is crucial for accurate material property calculations. The system uses two approaches:

1. **Real-time Contact Detection**: During measurement, detects when the indenter first touches the material
2. **Post-processing Analysis**: Refines the contact point after measurement for maximum accuracy

## Real-time Contact Detection

### Parameters

The `dynamic_indentation_measurement` function uses these key parameters:

- `contact_force_threshold` (default: 2.0N): Force threshold to detect contact
- `force_limit` (default: 45.0N): Maximum force before stopping
- `step_size` (default: 0.1mm): Distance between measurements

### How It Works

1. **Baseline Measurement**: Takes initial force readings to establish baseline
2. **Continuous Monitoring**: Moves indenter down in small steps while monitoring force
3. **Contact Detection**: When `corrected_force` (raw force - baseline) exceeds threshold, contact is detected
4. **Measurement Continuation**: Continues measuring until force limit or target Z is reached

### Code Example

```python
from src.force_monitoring import dynamic_indentation_measurement

# Run measurement with contact detection
success = dynamic_indentation_measurement(
    cnc=cnc_controller,
    force_sensor=force_sensor,
    well="A1",
    contact_force_threshold=2.0,  # Contact detected at 2N
    force_limit=45.0,             # Stop at 45N
    step_size=0.1                 # 0.1mm steps
)
```

## Post-processing Analysis

### Contact Point Detection Methods

The system provides two methods for finding the contact point in post-processing:

#### 1. Basic Contact Detection (`find_contact_point`)

Uses multiple strategies to robustly detect contact:

- **Sustained Force**: Looks for force staying above threshold for several points
- **Smoothed Force**: Uses rolling average to reduce noise
- **Force Derivative**: Detects rapid force changes
- **Simple Threshold**: Fallback to basic threshold detection

#### 2. True Contact Detection (`find_true_contact_point`)

**This is the key method for accurate analysis.**

1. **Threshold Detection**: Finds first point where force exceeds threshold
2. **Trend Analysis**: Analyzes force trend before threshold contact
3. **Extrapolation**: Extrapolates trend back to zero force
4. **True Contact**: Finds where indenter actually first touched material

**Why This Matters**: The threshold contact point is delayed due to the force threshold. Extrapolating back gives the true contact point where force should be zero.

### Code Example

```python
from src.analysis import IndentationAnalyzer

analyzer = IndentationAnalyzer()

# Load data
analyzer.load_data("measurement_data.csv")

# Find true contact point
contact_idx = analyzer.find_true_contact_point(z_positions, corrected_forces)
print(f"True contact at index {contact_idx}: Z={z_positions[contact_idx]:.3f}mm")
```

## Comprehensive Analysis Methods

### Original Script Analysis (`analyze_well_original_method`)

This implements the complete analysis pipeline from the original ASMI script:

#### 1. **Force Correction Tables**
Extensive lookup tables based on:
- **Poisson Ratio**: 0.3-0.5 range
- **Sample Height**: 3.5-9.5mm range
- **Correction Parameters**: b and c values for each combination

#### 2. **Iterative Depth Adjustment**
The core algorithm:
```
while |d0| > 0.01 mm:
    ├─ (re-)compute C·d^B for every depth point
    ├─ F_corr ← F_meas / (C·d^B)  # new force correction
    ├─ select points with 0.24 mm ≤ d ≤ 0.50 mm
    ├─ least-squares fit F_corr = A·(d – d0)^1.5
    └─ shift all depths d ← d – d0
end
```

#### 3. **Elastic Modulus Calculation**
- **Hertzian Model**: F = A·(d – d0)^1.5
- **Effective Modulus**: E* = (A × 0.75) / (r_sphere^0.5)
- **Material Modulus**: Accounts for sphere and sample properties
- **Empirical Correction**: Adjusts for softer samples

### Code Example

```python
# Use original script's comprehensive analysis
results = analyzer.analyze_well_original_method(
    depths=depths,
    forces=forces,
    p_ratio=0.4,
    well_name="A1"
)

if results:
    print(f"Elastic Modulus: {results['elastic_modulus']:,.0f} Pa")
    print(f"Uncertainty: {results['uncertainty']:,.0f} Pa")
    print(f"Converged: {results['converged']}")
```

## Data Output

### CSV Structure

Measurement data is saved with these columns:
- `Well`: Well identifier (e.g., "A1")
- `Z_Position(mm)`: Z position of indenter
- `Raw_Force(N)`: Raw force sensor reading
- `Corrected_Force(N)`: Force minus baseline
- `Timestamp`: Measurement timestamp

### Analysis Results

The `AnalysisResult` dataclass contains:
- `elastic_modulus`: Calculated elastic modulus (Pa)
- `uncertainty`: Uncertainty in measurement (Pa)
- `poisson_ratio`: Poisson ratio used
- `fit_A`: Hertzian fit parameter A
- `fit_d0`: Depth offset parameter
- `converged`: Whether iterative fitting converged
- `depth_range`: Analysis depth range
- `contact_z`: Z position at contact
- `contact_force`: Force at contact point

## Visualization

### Contact Detection Plot (`plot_contact_detection`)

Shows:
- Force vs. Z position
- Contact point highlighted
- Baseline force level
- Threshold line

### Analysis Results Plot (`plot_results`)

Shows:
- Original vs. corrected data
- Hertzian fit curve
- Analysis range highlighted
- Material properties

## Best Practices

### 1. **Threshold Selection**
- **Too Low**: False positives from noise
- **Too High**: Delayed contact detection
- **Recommended**: 2.0N for most materials

### 2. **Analysis Range**
- **Depth Range**: 0.24-0.50mm (optimal for Hertzian model)
- **Data Points**: At least 10 points in range
- **Force Range**: Should exceed 0.04N for reliable analysis

### 3. **Poisson Ratio**
- **Range**: 0.3-0.5 for most polymers
- **Default**: 0.4 if unknown
- **Impact**: Affects force correction tables

### 4. **Sample Height**
- **Estimation**: Based on contact Z position
- **Range**: 3.5-9.5mm for correction tables
- **Impact**: Critical for force corrections

## Troubleshooting

### Common Issues

1. **No Contact Detected**
   - Check force sensor calibration
   - Lower contact threshold
   - Verify sample is present

2. **Poor Fit Quality**
   - Ensure sufficient data in 0.24-0.50mm range
   - Check for sample surface contamination
   - Verify Poisson ratio

3. **Non-convergence**
   - Increase max iterations
   - Check data quality
   - Verify sample properties

4. **Unrealistic Modulus**
   - Check units (Pa vs kPa vs MPa)
   - Verify sphere radius
   - Check force sensor calibration

### Error Messages

- `"Not enough data points"`: Need at least 10 measurements
- `"Force range too small"`: Force variation < 0.04N
- `"Fit failed"`: Numerical issues in curve fitting
- `"No contact detected"`: Force never exceeded threshold

## Example Workflow

```python
# 1. Run measurement
success = dynamic_indentation_measurement(cnc, force_sensor, "A1")

# 2. Load and analyze data
analyzer = IndentationAnalyzer()
analyzer.load_data("well_A1_20240101_120000.csv")

# 3. Find contact point
contact_idx = analyzer.find_true_contact_point(z_positions, forces)

# 4. Calculate depths
depths, contact_z, shifted_forces = analyzer.calculate_indentation_depth(
    z_positions, contact_idx, forces
)

# 5. Perform comprehensive analysis
results = analyzer.analyze_well_original_method(
    depths, shifted_forces, p_ratio=0.4, well_name="A1"
)

# 6. Save and visualize results
analyzer.plot_results(results, save_plot=True)
```

This comprehensive approach ensures accurate contact point detection and reliable material property calculations. 