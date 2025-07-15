import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from src.analysis import IndentationAnalyzer
from matplotlib.lines import Line2D

# Map well to (Cm, intensity)
WELL_INFO = {
    'A5': ('5.0M', '1%'),
    'B5': ('5.0M', '3%'),
    'C5': ('5.0M', '5%'),
    'A6': ('6.0M', '5%'),
    'B6': ('6.0M', '3%'),
    'C6': ('6.0M', '1%'),
}

# Set Poisson's ratio for gels
POISSON_RATIO = 0.33

# Path to the run folder (edit as needed)
RUN_FOLDER = 'results/well_measurements/run_007_20250711_053933'

analyzer = IndentationAnalyzer(data_dir=RUN_FOLDER)

well_results = {}

for fname in os.listdir(RUN_FOLDER):
    if fname.startswith('well_') and fname.endswith('.csv') and 'summary' not in fname:
        # Extract well name
        well = fname.split('_')[1]
        # Load only down movement data
        with open(os.path.join(RUN_FOLDER, fname), 'r') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # Find header
        for i, row in enumerate(rows):
            if row and row[0] == 'Timestamp(s)':
                header_idx = i
                break
        else:
            continue
        header = rows[header_idx]
        movement_idx = header.index('Movement') if 'Movement' in header else -1
        if movement_idx == -1:
            print(f"No 'Movement' column in {fname}, skipping.")
            continue
        # Filter for down movement
        data_rows = [row for row in rows[header_idx+1:] if row and len(row) > movement_idx and row[movement_idx] == 'Down']
        # Write filtered data to temp file for analysis
        temp_file = os.path.join(RUN_FOLDER, f"temp_{well}.csv")
        with open(temp_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Well'] + header)
            for row in data_rows:
                writer.writerow([well] + row)
        # Analyze using IndentationAnalyzer
        analyzer.load_data(f"temp_{well}.csv")
        result = analyzer.analyze_well(well, POISSON_RATIO)
        if result:
            well_results[well] = result
            print(f"Well {well}: E = {result.elastic_modulus} Pa, R² = {result.fit_quality}")
            analyzer.plot_results(result)
        else:
            print(f"Analysis failed for well {well}")
        os.remove(temp_file)

# Summary plot
fig, ax = plt.subplots(figsize=(8,6))
colors = {'5.0M': 'tab:blue', '6.0M': 'tab:orange'}
markers = {'1%': 'o', '3%': 's', '5%': '^'}
for well, result in well_results.items():
    cm, intensity = WELL_INFO.get(well, ('?', '?')) # '?' is a placeholder for the well name
    ax.scatter([intensity], [result.elastic_modulus/1e6], color=colors[cm], marker=markers[intensity], s=100, label=f'{well} ({cm}, {intensity})')
    ax.text(intensity, result.elastic_modulus/1e6, well, fontsize=10, ha='center', va='bottom')

# Custom legend
handles = [Line2D([0],[0], color=colors['5.0M'], marker='o', linestyle='', label='Cm=5.0M'),
           Line2D([0],[0], color=colors['6.0M'], marker='o', linestyle='', label='Cm=6.0M')]
ax.legend(handles=handles, title='Composition', loc='upper left')
ax.set_xlabel('Intensity (%)')
ax.set_ylabel('Elastic Modulus (MPa)')
ax.set_title('Summary: Young\'s Modulus by Well')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show() 