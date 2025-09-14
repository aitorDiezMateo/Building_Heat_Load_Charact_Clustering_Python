import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, pi, exp

if len(sys.argv) < 9:
    raise SystemExit(
        "Usage: 03_Inspection_Changepoint_Outliers.py <formatted.csv> <summary.csv> <out1> <out2> <out3> <out4> <out5> <out_data_csv> [--verbose]"
    )

input_formatted = sys.argv[1]
input_summary = sys.argv[2]
out_changepoint = sys.argv[3]
out_outliers1 = sys.argv[4]
out_outliers2 = sys.argv[5]
out_outliers3 = sys.argv[6]
out_outliers4 = sys.argv[7]
out_data_csv = sys.argv[8]
verbose = any(flag in ("--verbose", "-v", "true", "True", "1") for flag in sys.argv[9:])

if verbose:
    print("Loading input data and parameter files for analysis...")

df = pd.read_csv(input_formatted, sep=";")
# Ensure Date as datetime
if "Date" in df.columns:
    df["DATE"] = pd.to_datetime(df["Date"], errors="coerce")

params = pd.read_csv(input_summary, sep=";")

# Merge parameters onto data by DATE_hour_week
if verbose:
    print("Integrating changepoint parameters with data records...")
df = df.merge(
    params[["DATE_hour_week", "slope_Temp", "slope_Irrad", "intercept", "minimum"]],
    on="DATE_hour_week",
    how="left",
)

# Compute fitted power and residuals
if verbose:
    print("Computing fitted power values and residual analysis...")
linear_estimate = (
    df["intercept"].astype(float)
    + df["slope_Temp"].astype(float) * df["Temperature"].astype(float)
    + df["slope_Irrad"].astype(float) * df["Solar_Irradiation"].astype(float)
)
df["Power_fitted"] = np.maximum(df["minimum"].astype(float), linear_estimate)
df.loc[df["Power_fitted"] < 0, "Power_fitted"] = 0.0
df["Power_residuals"] = df["Power_fitted"] - df["Power"].astype(float)

# Outlier flag by residuals (95% CI)
resid = df["Power_residuals"].to_numpy()
resid_mean = np.nanmean(resid)
resid_std = np.nanstd(resid, ddof=0)
threshold = 1.96 * resid_std
df["IS_Outlier"] = np.abs(df["Power_residuals"] - resid_mean) > threshold

# ------------------------------------------------------------------
# Plot 1: Changepoint model lines per hour-of-week
# ------------------------------------------------------------------
if verbose:
    print("Generating changepoint model visualization...")
x1 = -50.0
x3 = 50.0
# Prepare coordinates from params (ignore slope_Irrad for this plot, like R)
p = params.copy()
p["y1"] = p["intercept"] + p["slope_Temp"] * x1
p["y2"] = p["minimum"]
p["y3"] = p["minimum"]

# Avoid division by zero for x2 when slope_Temp ~ 0
with np.errstate(divide="ignore", invalid="ignore"):
    p["x2"] = -(p["intercept"] - p["minimum"]) / p["slope_Temp"]

y_max = np.nanmax(p["y1"].to_numpy()) if len(p) else 1.0
if not np.isfinite(y_max) or y_max <= 0:
    y_max = 1.0

plt.figure(figsize=(6, 4), dpi=300)
plt.xlim(-50, 50)
plt.ylim(0, y_max)
plt.xlabel("Temperature [C]")
plt.ylabel("Heat Load [kWh]")
plt.title("Changepoint model per hour-of-week", pad=8)

for _, row in p.iterrows():
    slope_t = row.get("slope_Temp", np.nan)
    if np.isnan(slope_t) or np.isclose(slope_t, 0.0):
        continue
    xs = [x1, row["x2"], x3]
    ys = [row["y1"], row["y2"], row["y3"]]
    if any(np.isnan(xs)) or any(np.isnan(ys)):
        continue
    plt.plot(xs, ys, color="black", linewidth=0.8)

os.makedirs(os.path.dirname(out_changepoint), exist_ok=True)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_changepoint, dpi=300)
plt.close()

# ------------------------------------------------------------------
# Plot 2: Power vs Power_fitted colored by IS_Outlier
# ------------------------------------------------------------------
if verbose:
    print("Creating outlier analysis scatter plot (Power vs Fitted)...")
plt.figure(figsize=(6, 4), dpi=300)
colors = np.where(df["IS_Outlier"], "red", "black")
plt.scatter(df["Power"], df["Power_fitted"], c=colors, s=6, alpha=0.8)
plt.xlabel("Power")
plt.ylabel("Power_fitted")
plt.title("Power vs Fitted Power (outliers in red)", pad=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_outliers1, dpi=300)
plt.close()

# ------------------------------------------------------------------
# Plot 3: Temperature vs Power colored by IS_Outlier
# ------------------------------------------------------------------
if verbose:
    print("Creating outlier analysis scatter plot (Temperature vs Power)...")
plt.figure(figsize=(6, 4), dpi=300)
colors = np.where(df["IS_Outlier"], "red", "black")
plt.scatter(df["Temperature"], df["Power"], c=colors, s=6, alpha=0.8)
plt.xlabel("Temperature")
plt.ylabel("Power")
plt.title("Temperature vs Power (outliers in red)", pad=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_outliers2, dpi=300)
plt.close()

# ------------------------------------------------------------------
# Plot 4: Monthly aggregation - number of outliers and share (%)
# ------------------------------------------------------------------
if verbose:
    print("Generating monthly aggregation analysis plot...")
df_month = df.copy()
# Month label as YYYY-MM (robust to tz-aware datetimes)
dt_series = pd.to_datetime(df_month["DATE"], errors="coerce", utc=True)
df_month["Month"] = dt_series.dt.tz_convert(None).dt.to_period("M").astype(str)

agg = (
    df_month.groupby("Month", dropna=False)
    .agg(Sum_IS_Outlier=("IS_Outlier", "sum"), Total_Count=("IS_Outlier", "size"))
    .reset_index()
)
agg["Percentage_IS_Outlier"] = np.where(
    agg["Total_Count"] > 0,
    (agg["Sum_IS_Outlier"].astype(float) / agg["Total_Count"].astype(float)) * 100.0,
    0.0,
)
max_sum = float(agg["Sum_IS_Outlier"].max()) if len(agg) else 1.0
if max_sum <= 0:
    max_sum = 1.0
max_percentage = 100.0
agg["Scaled_Percentage"] = agg["Percentage_IS_Outlier"] * (max_sum / max_percentage)

# Plot with twin axes
plt.figure(figsize=(6, 4), dpi=300)
x_idx = np.arange(len(agg))
# Bars for scaled percentage
plt.bar(x_idx, agg["Scaled_Percentage"], color="red", alpha=0.5, width=0.8, label="Share (%) scaled")
# Points for count
plt.scatter(x_idx, agg["Sum_IS_Outlier"], color="blue", s=20, label="Number of outliers")
plt.xticks(x_idx, agg["Month"], rotation=45, ha="right")
plt.ylabel("Number of outliers")
ax1 = plt.gca()
ax2 = ax1.twinx()
# Map left scale to right scale
ax2.set_ylim(ax1.get_ylim()[0] * (max_percentage / max_sum), ax1.get_ylim()[1] * (max_percentage / max_sum))
ax2.set_ylabel("Share of Outliers (%)")
ax1.set_title("Monthly outliers: count and share", pad=8)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_outliers3, dpi=300)
plt.close()

# ------------------------------------------------------------------
# Plot 5: Histogram of residuals with normal overlay and 95% CI
# ------------------------------------------------------------------
if verbose:
    print("Creating residuals histogram with confidence intervals...")
mean_res = resid_mean
sd_res = resid_std if resid_std > 0 else 1e-9

plt.figure(figsize=(6, 4), dpi=300)
# Histogram as density
plt.hist(df["Power_residuals"].dropna().to_numpy(), bins=300, density=True, color="skyblue", edgecolor="black", alpha=0.7)
# Normal curve
r = df["Power_residuals"].dropna().to_numpy()
if r.size > 0:
    x_vals = np.linspace(r.min(), r.max(), 500)
    pdf_vals = (1.0 / (sd_res * sqrt(2 * pi))) * np.exp(-0.5 * ((x_vals - mean_res) / sd_res) ** 2)
    plt.plot(x_vals, pdf_vals, color="red", linewidth=1.0)
# CI lines
ci_lower = mean_res - 1.96 * sd_res
ci_upper = mean_res + 1.96 * sd_res
plt.axvline(ci_lower, linestyle="--", color="blue", linewidth=1.0)
plt.axvline(ci_upper, linestyle="--", color="blue", linewidth=1.0)
plt.title("Histogram of power residuals [kWh]\nover normally adjusted curve", pad=8)
plt.xlabel("Power_residuals")
plt.ylabel("Density")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(out_outliers4, dpi=300)
plt.close()

# ------------------------------------------------------------------
# Save the processed data with Power_fitted and IS_Outlier columns
# ------------------------------------------------------------------
if verbose:
    print("Saving processed dataset with outlier flags and fitted values...")

# Ensure DATE column name consistency for downstream processing
if 'Date' in df.columns and 'DATE' not in df.columns:
    df['DATE'] = df['Date']
    df = df.drop(columns=['Date'])  # Remove original Date column to avoid duplication

df.to_csv(out_data_csv, index=False, sep=";")

