import dask.dataframe as dd
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Format_Input import format_input

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: 01_load_data.py <input_data.csv> <output_formatted.csv> [--verbose]"
    )

input_file = sys.argv[1]
output_file = sys.argv[2]
verbose = False
if len(sys.argv) > 3 and sys.argv[3] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

if verbose:
    print("Loading and processing input data...")
    
df = dd.read_csv(input_file, sep = ";")

df['Date'] = df['Year'].astype(str) + ' ' + df['Month'].astype(str) + ' ' + df['Day_Month'].astype(str) + ' ' + df['Hour_Day'].astype(str)
df['Date'] = dd.to_datetime(df['Date'], format='%Y %m %d %H', utc=True)

# Keep only rows with __:00 times
df = df[df['Date'].dt.minute == 0]


df = format_input(df,
                  col_power = "Power")


if verbose:
    print("Saving processed data to output file...")
    
df.to_csv(output_file, index=False, sep=";", single_file=True)

if verbose:
    print("Data loading and formatting completed successfully.")