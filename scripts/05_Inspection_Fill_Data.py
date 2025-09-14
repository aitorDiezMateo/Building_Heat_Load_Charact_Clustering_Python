import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 4:
    raise SystemExit(
        "Usage: 05_Inspection_Fill_Data.py <input_filled.csv> <output_plot1.jpg> <output_plot2.jpg> [--verbose]"
    )

input_file = sys.argv[1]
output_plot1 = sys.argv[2]
output_plot2 = sys.argv[3]
verbose = False
if len(sys.argv) > 4 and sys.argv[4] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

if verbose:
    print("Loading filled data for inspection visualization...")
    
# Load the data
dat = pd.read_csv(input_file, sep=";")

# Ensure DATE column is datetime if it exists
if 'Date' in dat.columns:
    dat['Date'] = pd.to_datetime(dat['Date'])
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])

if verbose:
    print("Creating visualization plots for data repair inspection...")
    
# Plot 1: Power vs Power_corrected colored by IS_Repaired
plt.figure(figsize=(6, 4), dpi=300)
colors = np.where(dat['IS_Repaired'], 'red', 'black')
plt.scatter(dat['Power'], dat['Power_corrected'], c=colors, s=6, alpha=0.8)
plt.xlabel('Power [kWh], original value')
plt.ylabel('Power [kWh], corrected value')

# Create custom legend
from matplotlib.lines import Line2D
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Not Repaired'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Repaired')]
plt.legend(handles=legend_elements, title='IS_Repaired')

plt.tight_layout()
os.makedirs(os.path.dirname(output_plot1), exist_ok=True)
plt.savefig(output_plot1, dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Temperature vs Power_corrected colored by IS_Repaired
plt.figure(figsize=(6, 4), dpi=300)
colors = np.where(dat['IS_Repaired'], 'red', 'black')
plt.scatter(dat['Temperature'], dat['Power_corrected'], c=colors, s=6, alpha=0.8)
plt.xlabel('Temperature')
plt.ylabel('Power')

# Create custom legend
legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Not Repaired'),
                   Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Repaired')]
plt.legend(handles=legend_elements, title='IS_Repaired')

plt.tight_layout()
os.makedirs(os.path.dirname(output_plot2), exist_ok=True)
plt.savefig(output_plot2, dpi=300, bbox_inches='tight')
plt.close()

if verbose:
    print("Data repair inspection visualization completed successfully.")
    print(f"Plots saved: {output_plot1}, {output_plot2}")
