import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Repair_1h_double_side import repair_1h_double_side
from functions.Repair_multiple_h_double_side import repair_multiple_h_double_side

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: 04_Fill_Data.py <input_processed.csv> <output_filled.csv> [--verbose]"
    )

input_file = sys.argv[1]
output_file = sys.argv[2]
verbose = False
if len(sys.argv) > 3 and sys.argv[3] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

if verbose:
    print("Loading data for gap filling process...")
    
# Load the data
dat = pd.read_csv(input_file, sep=";")

# Ensure DATE column name consistency and convert to datetime
if 'Date' in dat.columns and 'DATE' not in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['Date'])
    dat = dat.drop(columns=['Date']) 
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])
else:
    raise ValueError("Neither 'Date' nor 'DATE' column found in input data")

if verbose:
    print("Filling 1-hour data gaps using interpolation...")
    
# Fill 1h voids with observations available for interpolation
string_date = "DATE"
vec_vars = ["DATE_YYYY_MM_DD", "Holiday", "Temperature", "Solar_Irradiation", "Power", "Power_fitted", "IS_Outlier"]
vec_repair = ["Power"]

dat = repair_1h_double_side(dat, string_date, vec_vars, vec_repair)

if verbose:
    print("Filling multi-hour data gaps (2-5 hours) using interpolation...")
    
# Fill 5h voids with observations available for interpolation
string_date = "DATE"
vec_vars = ["DATE_YYYY_MM_DD", "Holiday", "Temperature", "Solar_Irradiation", "Power", "Power_fitted", "IS_Outlier"]
vec_repair = ["Power"]

num_hours_max = 5
for cont_num_hours in range(2, num_hours_max + 1):
    num_hours = cont_num_hours
    for cont_num_hour_pre in range(1, num_hours + 1):
        nh_pre = cont_num_hour_pre
        nh_post = num_hours + 1 - nh_pre
        dat = repair_multiple_h_double_side(dat, string_date, vec_vars, vec_repair, nh_pre, nh_post)

if verbose:
    print("Saving filled data to output file...")
    
# Save the result
dat.to_csv(output_file, index=False, sep=";")

if verbose:
    print("Data gap filling process completed successfully.")
