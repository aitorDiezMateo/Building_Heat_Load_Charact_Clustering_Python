import sys
import os
import pandas as pd
import numpy as np
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Detect_Outliers_Model_Based_ch_3p import Detect_Outliers_Model_Based_CH_3P

if len(sys.argv) < 5:
    raise SystemExit(
        "Usage: 06_Final_Changepoint_Model.py <input_filled.csv> <output_data.csv> <output_params.csv> <use_slurm> [--verbose]"
    )

input_file = sys.argv[1]
output_data_file = sys.argv[2]
output_params_file = sys.argv[3]
use_slurm = sys.argv[4].lower() in ("true", "1", "yes", "on")
verbose = False
if len(sys.argv) > 5 and sys.argv[5] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

# Load the data
dat = pd.read_csv(input_file, sep=";")

# Ensure DATE column is datetime
if 'Date' in dat.columns and 'DATE' not in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['Date'])
    dat = dat.drop(columns=['Date'])  
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])

# Restructure dataframe
dat['Power_original'] = dat['Power'].copy()
dat['Power'] = dat['Power_corrected'].copy()

# Create a DataFrame with the parameters of changepoint models for each hour of the week
unique_hours = sorted(dat["DATE_hour_week"].unique())

Changepoint_Pars_summ_TOW = pd.DataFrame({
    "DATE_hour_week": unique_hours,
    "slope_Temp": np.nan,
    "slope_Irrad": np.nan,
    "intercept": np.nan,
    "minimum": np.nan,
})

# Update data frame with final format
dat['Power_fitted'] = np.nan
dat['Power_residuals'] = np.nan
dat['IS_Outlier'] = np.nan

# Empty dataframe for output
dat_output = pd.DataFrame(columns=dat.columns)

start_time = time.time()

if verbose:
    print("Initiating final changepoint model processing...")

if verbose:
    print(f"Processing {len(unique_hours)} unique time-of-week segments for final model...")

# Repeat process with optimized parameter set
for i in unique_hours:
    # Subset by hour of the week
    dat_subs_hw = dat[dat["DATE_hour_week"] == i].copy()
    
    # Get the optimal changepoint function parameters for each subset
    dat_processed, params = Detect_Outliers_Model_Based_CH_3P(
        dat_subs_hw, threshold_outlier=1.96, pop_size=50, max_iter=100, verbose=verbose, slurm_cluster=use_slurm
    )
    
    # Add processed data to output
    if dat_output.empty:
        dat_output = dat_processed
    else:
        dat_output = pd.concat([dat_output, dat_processed], ignore_index=True)
    
    # Take only the first solution if GA returns multiple solutions
    if len(params) > 4:
        params = params[:4]
    
    # Save the parameters to the summary pandas dataframe
    idx = Changepoint_Pars_summ_TOW.index[Changepoint_Pars_summ_TOW["DATE_hour_week"] == i]
    if len(idx) == 1:
        Changepoint_Pars_summ_TOW.loc[idx, ["slope_Temp", "slope_Irrad", "intercept", "minimum"]] = params
    
    # Control process
    current_time = time.time()
    elapsed_time = current_time - start_time
    estimated_time = elapsed_time * (len(unique_hours) / (unique_hours.index(i) + 1))
    
    if verbose:
        print(
            f"Progress: {unique_hours.index(i) + 1} / {len(unique_hours)}. "
            f"elapsed time: {elapsed_time:.2f} sec, "
            f"estimated time: {estimated_time:.2f} sec"
        )

# Reorder output dataframe by date
dat = dat_output.sort_values('DATE').reset_index(drop=True)
dat['Power_fitted_TOW'] = dat['Power_fitted'].copy()

if verbose:
    print("Saving final processed dataset and model parameters...")

# Save the processed data
dat.to_csv(output_data_file, index=False, sep=";")

# Save the parameters summary
Changepoint_Pars_summ_TOW.to_csv(output_params_file, sep=";", index=False)
