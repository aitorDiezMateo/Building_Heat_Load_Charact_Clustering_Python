import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import seaborn as sns

if len(sys.argv) < 9:
    raise SystemExit(
        "Usage: 07_Clusterization.py <input_data.csv> <profiles1.jpg> <profiles2.jpg> <elbow.jpg> <silhouette.jpg> <cluster.jpg> <comprehensive.jpg> <assignments.csv> [--verbose]"
    )

input_file = sys.argv[1]
output_profiles1 = sys.argv[2]
output_profiles2 = sys.argv[3]
output_elbow = sys.argv[4]
output_silhouette = sys.argv[5]
output_cluster = sys.argv[6]
output_comprehensive = sys.argv[7]
output_assignments = sys.argv[8]
verbose = False
if len(sys.argv) > 9 and sys.argv[9] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

# Extract output directory from the first output file
import os
output_dir = os.path.dirname(output_profiles1)

# Load the data
dat = pd.read_csv(input_file, sep=";")

# Ensure DATE column is datetime
if 'Date' in dat.columns and 'DATE' not in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['Date'])
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])

# Create data frame with daily power vectors
if verbose:
    print("Creating daily power profiles for clustering analysis...")

# Ensure Power_corrected column is numeric
dat['Power_corrected'] = pd.to_numeric(dat['Power_corrected'], errors='coerce')

# Pivot data frame to get one observation per day, where 24 variables are obtained
dat_day = dat.pivot_table(
    index='DATE_YYYY_MM_DD', 
    columns='DATE_hour_day', 
    values='Power_corrected', 
    aggfunc='first'
).reset_index()

# Rename columns
try:
    dat_day.columns = ['DATE_YYYY_MM_DD'] + [f'Power.{int(col)}' for col in dat_day.columns[1:]]
except (ValueError, TypeError) as e:
    if verbose:
        print(f"Warning: Error in column renaming: {e}")
        print(f"Original columns: {dat_day.columns.tolist()}")
    # Fallback: use column names as they are
    dat_day.columns = ['DATE_YYYY_MM_DD'] + [f'Power.{col}' for col in dat_day.columns[1:]]

# Plot of daily profiles
if verbose:
    print("Generating daily power profile visualizations...")
plt.figure(figsize=(6, 4), dpi=300)
for idx in range(min(len(dat_day), 100)):  # Limit to first 100 days for readability
    hours = list(range(24))
    power_values = dat_day.iloc[idx, 1:25].values
    # Convert to numeric, replacing non-numeric values with NaN
    power_values = pd.to_numeric(power_values, errors='coerce')
    if not np.isnan(power_values).all():
        plt.plot(hours, power_values, alpha=0.3, linewidth=0.5)

plt.xlabel('Hour of the day')
plt.ylabel('Power')
plt.title('Daily profiles')
plt.grid(True, alpha=0.3)
plt.tight_layout()
os.makedirs(output_dir, exist_ok=True)
plt.savefig(output_profiles1, dpi=300, bbox_inches='tight')
plt.close()

# Create data frame to identify clusters
if verbose:
    print("Preparing data matrix for clustering algorithm...")

# Avoid missing values
dat_day_clean = dat_day.dropna()

# Min-max normalization function for each day (row-wise)
def normalize_day_profile(row):
    """Normalize each day's profile so min=0 and max=1 for that specific day"""
    power_values = row.values
    min_val = np.min(power_values)
    max_val = np.max(power_values)
    
    if max_val == min_val:
        # If all values are the same, return zeros
        return pd.Series(np.zeros(len(power_values)), index=row.index)
    else:
        # Normalize: (value - min) / (max - min)
        return pd.Series((power_values - min_val) / (max_val - min_val), index=row.index)

# 0-1 data normalization (applied to each day individually)
dat_day_normalized = dat_day_clean.copy()
power_columns = [col for col in dat_day_normalized.columns if col.startswith('Power.')]

# Apply row-wise normalization (each day gets normalized individually)
dat_day_normalized[power_columns] = dat_day_normalized[power_columns].apply(normalize_day_profile, axis=1)

# Plot of daily profiles after normalization
if verbose:
    print("Generating normalized daily profile visualizations...")
plt.figure(figsize=(6, 4), dpi=300)
for idx in range(min(len(dat_day_normalized), 100)):  # Limit for readability
    hours = list(range(24))
    power_values = dat_day_normalized.iloc[idx, 1:25].values
    # Convert to numeric, replacing non-numeric values with NaN
    power_values = pd.to_numeric(power_values, errors='coerce')
    if not np.isnan(power_values).all():
        plt.plot(hours, power_values, alpha=0.3, linewidth=0.5)

plt.xlabel('Hour of the day')
plt.ylabel('Power (normalized)')
plt.title('Normalized daily profiles')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_profiles2, dpi=300, bbox_inches='tight')
plt.close()

# Clusterization
if verbose:
    print("Initiating clustering optimization process...")

# Define the optimum number of clusters through various methods
k_min = 3  # Avoid a 2-cluster output, where only weekdays and weekends are separated
k_max = 30

# Prepare data for clustering (exclude DATE column)
clustering_data = dat_day_normalized[power_columns].values

# Elbow method
if verbose:
    print("Computing elbow method for optimal cluster determination...")
wss_values = []
for k in range(1, k_max + 1):
    kmeans = KMeans(n_clusters=k, random_state=25, n_init=10)
    kmeans.fit(clustering_data)
    wss_values.append(kmeans.inertia_)

# Difference between two consecutive WSS values
wss_diffs = np.abs(np.diff(wss_values))

# Define the optimal number of clusters as the position of the "elbow"
threshold = 0.05 * np.max(wss_diffs)
optimal_k_elbow = np.where(wss_diffs > threshold)[0][0] + 2 if len(np.where(wss_diffs > threshold)[0]) > 0 else k_min

# Silhouette method
if verbose:
    print("Computing silhouette analysis for cluster validation...")
silhouette_scores = []
for k in range(2, k_max + 1):
    kmeans = KMeans(n_clusters=k, random_state=25, n_init=10)
    cluster_labels = kmeans.fit_predict(clustering_data)
    silhouette_avg = silhouette_score(clustering_data, cluster_labels)
    silhouette_scores.append(silhouette_avg)

# Define the optimal number of clusters as the one that maximizes the silhouette score
optimal_k_silhouette = np.argmax(silhouette_scores) + 2

# Optimal number of clusters
optimal_k = max(optimal_k_elbow, optimal_k_silhouette, k_min)
if verbose:
    print(f"Optimal cluster configuration determined: {optimal_k} clusters (elbow method: {optimal_k_elbow}, silhouette analysis: {optimal_k_silhouette})")

# Graphics
# Elbow method plot
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(range(1, k_max + 1), wss_values, 'o-', color='blue', linewidth=1)
plt.axvline(x=optimal_k_elbow, color='darkgreen', linestyle='--', label=f'Optimal k = {optimal_k_elbow}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Intra Cluster sum (WSS)')
plt.title('Elbow method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_elbow, dpi=300, bbox_inches='tight')
plt.close()

# Silhouette method plot
plt.figure(figsize=(6, 4), dpi=300)
plt.plot(range(2, k_max + 1), silhouette_scores, 'o-', color='blue', linewidth=1)
plt.axvline(x=optimal_k_silhouette, color='darkgreen', linestyle='--', label=f'Optimal k = {optimal_k_silhouette}')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Average Silhouette value')
plt.title('Silhouette method')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_silhouette, dpi=300, bbox_inches='tight')
plt.close()

# Definition of clusters
if verbose:
    print(f"Executing K-means clustering algorithm with {optimal_k} clusters...")
kmeans_final = KMeans(n_clusters=optimal_k, random_state=25, n_init=10)
cluster_labels = kmeans_final.fit_predict(clustering_data)

# Assign clusters to the original dataset
dat_day_clean = dat_day_clean.copy()
dat_day_normalized = dat_day_normalized.copy()
dat_day_clean['cluster'] = cluster_labels
dat_day_normalized['cluster'] = cluster_labels

# Inspection
if verbose:
    print("Generating cluster analysis visualizations...")

# Cluster by date
plt.figure(figsize=(10, 6), dpi=300)
plt.plot(dat_day_normalized['cluster'], 'o', markersize=2)
plt.xlabel('Day index')
plt.ylabel('Cluster')
plt.title('Cluster assignment by day')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(output_cluster, dpi=300, bbox_inches='tight')
plt.close()

# Hourly load profiles by cluster
if verbose:
    print("Computing statistical summaries for cluster profiles...")

# Data preparation
cluster_stats = {}
for stat_name, stat_func in [('mean', 'mean'), ('std', 'std'), 
                           ('q05', lambda x: np.percentile(x, 5)), 
                           ('q95', lambda x: np.percentile(x, 95))]:
    cluster_stats[stat_name] = dat_day_normalized.groupby('cluster')[power_columns].agg(stat_func)

hours = list(range(24))

# Load profiles based on mean and standard deviation
if verbose:
    print("Generating cluster profile plots with statistical measures...")
for k in range(optimal_k):
    cluster_data = dat_day_normalized[dat_day_normalized['cluster'] == k]
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot mean line
    mean_values = cluster_stats['mean'].iloc[k].values
    std_values = cluster_stats['std'].iloc[k].values
    
    plt.plot(hours, mean_values, 'b-', linewidth=2, label='Mean')
    plt.plot(hours, mean_values - 1.96 * std_values, 'r--', alpha=0.7, label='Mean ± 1.96*SD')
    plt.plot(hours, mean_values + 1.96 * std_values, 'r--', alpha=0.7)
    
    # Plot individual days as points
    for idx in range(len(cluster_data)):
        day_values = cluster_data.iloc[idx][power_columns].values
        plt.plot(hours, day_values, 'k.', alpha=0.3, markersize=1)
    
    plt.xlim(0, 23)
    plt.ylim(0, 1)
    plt.xlabel('Hour')
    plt.ylabel('Normalized load (0-1)')
    plt.title(f'Variation range (mean & std.dev), cluster {k+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'07_ClusterN{k+1}_001.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

# Load profiles based on quantiles
if verbose:
    print("Generating cluster profile plots with quantile distributions...")
for k in range(optimal_k):
    cluster_data = dat_day_normalized[dat_day_normalized['cluster'] == k]
    
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Plot mean line and quantiles
    mean_values = cluster_stats['mean'].iloc[k].values
    q05_values = cluster_stats['q05'].iloc[k].values
    q95_values = cluster_stats['q95'].iloc[k].values
    
    plt.plot(hours, mean_values, 'b-', linewidth=2, label='Mean')
    plt.plot(hours, q05_values, 'g--', alpha=0.7, label='5th percentile')
    plt.plot(hours, q95_values, 'g--', alpha=0.7, label='95th percentile')
    
    # Plot individual days as points
    for idx in range(len(cluster_data)):
        day_values = cluster_data.iloc[idx][power_columns].values
        plt.plot(hours, day_values, 'k.', alpha=0.3, markersize=1)
    
    plt.xlim(0, 23)
    plt.ylim(0, 1)
    plt.xlabel('Hour')
    plt.ylabel('Normalized load (0-1)')
    plt.title(f'Variation range (5-95% quantiles), cluster {k+1}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'07_ClusterN{k+1}_002.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

# Additional visualization: All cluster profiles in subplots
if verbose:
    print("Generating comprehensive cluster analysis visualization...")
fig, axes = plt.subplots(optimal_k, 1, figsize=(12, 4*optimal_k), dpi=300)
if optimal_k == 1:
    axes = [axes]  # Make it iterable for single cluster case

for k in range(optimal_k):
    ax = axes[k]
    plt.sca(ax)  # Set current axis
    
    # Get periods belonging to this cluster
    cluster_data = dat_day_normalized[dat_day_normalized['cluster'] == k]
    cluster_power_data = cluster_data[power_columns]
    
    # Plot each period in this cluster
    for idx, (_, row) in enumerate(cluster_power_data.iterrows()):
        plt.plot(range(len(row)), row.values, alpha=0.3, color='#c690d1')
    
    # Calculate and plot the average profile for this cluster (the red line that stands out)
    cluster_mean = cluster_power_data.mean()
    plt.plot(range(len(cluster_mean)), cluster_mean.values, color='red', linewidth=3, label='Cluster Mean')
    
    plt.title(f'Cluster {k+1} ({len(cluster_data)} days)')
    plt.xlabel('Hour of the day')
    plt.ylabel('Normalized Power (0-1)')
    
    # Set fixed y-axis scale with specific ticks
    plt.ylim(0.0, 1.0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Set x-axis labels to show time periods (sample every few to avoid crowding)
    n_ticks = min(12, len(power_columns))  # Show max 12 ticks
    tick_indices = np.linspace(0, len(power_columns)-1, n_ticks, dtype=int)
    hour_labels = [str(i) for i in tick_indices]
    plt.xticks(tick_indices, hour_labels)
    
    plt.grid(True, alpha=0.3)
    plt.legend()

plt.tight_layout()
plt.savefig(output_comprehensive, dpi=300, bbox_inches='tight')
plt.close()

# Individual cluster visualization files
if verbose:
    print("Generating individual cluster profile visualizations...")
for k in range(optimal_k):
    plt.figure(figsize=(10, 6), dpi=300)
    
    # Get periods belonging to this cluster
    cluster_data = dat_day_normalized[dat_day_normalized['cluster'] == k]
    cluster_power_data = cluster_data[power_columns]
    
    # Plot each period in this cluster
    for idx, (_, row) in enumerate(cluster_power_data.iterrows()):
        plt.plot(range(len(row)), row.values, alpha=0.3, color='#c690d1')
    
    # Calculate and plot the average profile for this cluster (the red line that stands out)
    cluster_mean = cluster_power_data.mean()
    plt.plot(range(len(cluster_mean)), cluster_mean.values, color='red', linewidth=3, label='Cluster Mean')
    
    plt.title(f'Cluster {k+1} Daily Profiles ({len(cluster_data)} days)')
    plt.xlabel('Hour of the day')
    plt.ylabel('Normalized Power (0-1)')
    
    # Set fixed y-axis scale with specific ticks
    plt.ylim(0.0, 1.0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Set x-axis labels to show time periods (sample every few to avoid crowding)
    n_ticks = min(12, len(power_columns))  # Show max 12 ticks
    tick_indices = np.linspace(0, len(power_columns)-1, n_ticks, dtype=int)
    hour_labels = [str(i) for i in tick_indices]
    plt.xticks(tick_indices, hour_labels)
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'07_ClusterN{k+1}_003.jpg'), dpi=300, bbox_inches='tight')
    plt.close()

if verbose:
    print(f"Clustering analysis completed successfully. {optimal_k} clusters identified.")
    print(f"Visualization outputs saved to: {output_dir}")

# Save cluster assignments
dat_day_clean[['DATE_YYYY_MM_DD', 'cluster']].to_csv(output_assignments, index=False, sep=';')
if verbose:
    print(f"Cluster assignments saved to: {output_assignments}")