import sys
import os
import pandas as pd
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Detect_Outliers_Model_Based_ch_3p import Detect_Outliers_Model_Based_CH_3P

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: 02_initial_changepoint_outliers.py <input_formatted.csv> <output_summary.csv> [--verbose]"
    )

input_file = sys.argv[1]
output_file = sys.argv[2]
verbose = False
if len(sys.argv) > 3 and sys.argv[3] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

df = pd.read_csv(input_file, sep=";")

unique_hours = df["DATE_hour_week"].unique()

Changepoint_Parameters_summary = pd.DataFrame({
    "DATE_hour_week": unique_hours,
    "slope_Temp": np.nan,
    "slope_Irrad": np.nan,
    "intercept": np.nan,
    "minimum": np.nan,
})

start_time = time.time()

if verbose:
    print("Initiating initial changepoint outlier detection process...")

if verbose:
    print(f"Processing {len(unique_hours)} unique hours of the week for changepoint analysis...")
# Iterate over each hour of the week
for j in unique_hours:
    # Subset by hour of the week
    df_subs_hw = df[df["DATE_hour_week"] == j].copy()
    
    # Get the optimal changepoint function parameters for each subset
    _, params = Detect_Outliers_Model_Based_CH_3P(
        df_subs_hw, threshold_outlier=1.96, pop_size=50, max_iter=100, verbose=verbose
    )
    
    # Take only the first solution if GA returns multiple solutions
    if len(params) > 4:
        params = params[:4]
    
    # Save the parameters to the summary pandas dataframe
    idx = Changepoint_Parameters_summary.index[Changepoint_Parameters_summary["DATE_hour_week"] == j]
    if len(idx) == 1:
        Changepoint_Parameters_summary.loc[idx, ["slope_Temp", "slope_Irrad", "intercept", "minimum"]] = params
    
    # Control process
    current_time = time.time()
    elapsed_time = current_time - start_time
    estimated_time = elapsed_time * (max(unique_hours) / j if j != 0 else float("inf"))
    
    if verbose:
        print(
            f"Progress: {j} / {len(unique_hours)}. "
            f"elapsed time: {elapsed_time:.2f} sec, "
            f"estimated time: {estimated_time:.2f} sec"
        )

if verbose:
    print("Saving changepoint parameter summary to output file...")
Changepoint_Parameters_summary.to_csv(output_file, sep=";", index=False)