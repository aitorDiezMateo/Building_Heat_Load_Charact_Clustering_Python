import sys
import os
import pandas as pd
import numpy as np
import time
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from functions.Detect_Outliers_Model_Based_ch_3p import Detect_Outliers_Model_Based_CH_3P

if len(sys.argv) < 5:
    raise SystemExit(
        "Usage: 09_Changepoint_Clusters.py <input_data.csv> <clusters.csv> <output_clust_params.csv> <output_pred_params.csv> [--verbose]"
    )

# Get command line arguments
input_data_file = sys.argv[1]
input_cluster_file = sys.argv[2]
output_clust_params = sys.argv[3]
output_clust_pred_params = sys.argv[4]
verbose = False
if len(sys.argv) > 5 and sys.argv[5] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

if verbose:
    print("Loading data for cluster-based changepoint analysis...")

# Load the data
dat = pd.read_csv(input_data_file, sep=";")
dat_cluster = pd.read_csv(input_cluster_file, sep=";")

# Ensure DATE column is datetime
if 'Date' in dat.columns and 'DATE' not in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['Date'])
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])

# Add cluster information to Dat & remove incomplete cases (those without cluster information) into Dat_clean
if verbose:
    print("Processing cluster assignment information...")

# Add a new column extracting only the date (without time) from the DATE variable
# To be used as index
dat['Date_full'] = dat['DATE'].copy()
dat['DATE'] = dat['DATE'].dt.date

# Convert cluster DATE to date format for merging
dat_cluster['DATE'] = pd.to_datetime(dat_cluster['DATE_YYYY_MM_DD']).dt.date

# Create CART predictions similar to 08_CART.py
if verbose:
    print("Generating CART model predictions for cluster validation...")

# Create daily summary data for CART
dat_day_cart = dat.groupby('DATE_YYYY_MM_DD').agg({
    'Temperature': 'mean',  # Average daily temperature
    'Solar_Irradiation': 'sum',  # Total (Cumulated) daily solar irradiation
    'DATE_day_year': 'first',
    'DATE_day_week': 'first', 
    'DATE_weekday': 'first',
    'Holiday': 'first'
}).reset_index()

# Convert DATE_YYYY_MM_DD to date for merging
dat_day_cart['DATE'] = pd.to_datetime(dat_day_cart['DATE_YYYY_MM_DD']).dt.date

# Merge with cluster information
dat_day_cart = dat_day_cart.merge(
    dat_cluster[['DATE', 'cluster']], 
    on='DATE', 
    how='left'
)

# Ensure that all the observations have data from both processes
dat_day_cart = dat_day_cart.dropna()

# Convert variables to appropriate types
dat_day_cart['Holiday'] = dat_day_cart['Holiday'].astype('category')
dat_day_cart['cluster'] = dat_day_cart['cluster'].astype('category')
dat_day_cart['DATE_weekday'] = dat_day_cart['DATE_weekday'].astype('category')

# Build CART model for predictions
features = ['Temperature', 'Solar_Irradiation', 'Holiday', 'DATE_day_year', 'DATE_day_week', 'DATE_weekday']
X = dat_day_cart[features].copy()

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['Holiday', 'DATE_weekday'], drop_first=False)
y = dat_day_cart['cluster']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build the CART model
cart_model = DecisionTreeClassifier(
    criterion='gini',
    min_samples_split=5,
    min_samples_leaf=5,
    max_depth=5,
    random_state=42
)

# Fit the model and predict
cart_model.fit(X_train, y_train)
cluster_pred = cart_model.predict(X)
dat_day_cart['cluster_PRED'] = cluster_pred

# Merge cluster information with original data
dat = dat.merge(dat_day_cart[['DATE', 'cluster', 'cluster_PRED']], on='DATE', how='left')

# Remove the temporary date column and restore original DATE
dat['DATE'] = dat['Date_full']
dat = dat.drop(columns=['Date_full'])

# Remove observations without cluster first
dat_clean = dat.dropna().copy()  # Explicitly create a copy to avoid warnings

# Code cluster+hour information (now safe to convert to int)
dat_clean['ClusterHour'] = dat_clean['cluster'].astype(int) * 100 + dat_clean['DATE_hour_day']
dat_clean['ClusterHour_PRED'] = dat_clean['cluster_PRED'].astype(int) * 100 + dat_clean['DATE_hour_day']

if verbose:
    print(f"Clean data shape: {dat_clean.shape}")
    print("Starting changepoint processing with actual clusters...")

# Process with clusters as identified
start_time = time.time()

# Create a data frame with the parameters of changepoint models for each cluster
unique_cluster_hours = sorted(dat_clean['ClusterHour'].unique())
changepoint_pars_summ_clust = pd.DataFrame({
    'ClusterHour': unique_cluster_hours,
    'slope_Temp': np.nan,
    'slope_Irrad': np.nan,
    'intercept': np.nan,
    'minimum': np.nan
})

# Update data frame with final format
dat_clean['Power_fitted'] = np.nan
dat_clean['Power_residuals'] = np.nan
dat_clean['IS_Outlier'] = np.nan

# Empty dataframe for output
dat_output = pd.DataFrame(columns=dat_clean.columns)

for i, cluster_hour in enumerate(unique_cluster_hours):
    dat_subs_clust = dat_clean[dat_clean['ClusterHour'] == cluster_hour].copy()
    
    if verbose:
        print(f"Processing cluster hour {cluster_hour} ({i+1}/{len(unique_cluster_hours)}) with {len(dat_subs_clust)} observations")
    
    # Get the optimal changepoint function parameters for each subset
    dat_processed, params = Detect_Outliers_Model_Based_CH_3P(
        dat_subs_clust, threshold_outlier=1.96, pop_size=50, max_iter=100, verbose=False
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
    idx = changepoint_pars_summ_clust.index[changepoint_pars_summ_clust['ClusterHour'] == cluster_hour]
    if len(idx) == 1:
        changepoint_pars_summ_clust.loc[idx, ['slope_Temp', 'slope_Irrad', 'intercept', 'minimum']] = params
    
    # Control process
    if verbose:
        current_time = time.time()
        elapsed_time = current_time - start_time
        estimated_time = elapsed_time * (len(unique_cluster_hours) / (i + 1))
        print(f"Progress: {i+1} / {len(unique_cluster_hours)}. "
              f"elapsed time: {elapsed_time:.2f} sec, "
              f"estimated time: {estimated_time:.2f} sec")

# Reorder output dataframe by date
dat_clean = dat_output.sort_values('DATE').reset_index(drop=True)
dat_clean['Power_fitted_ClustH'] = dat_clean['Power_fitted'].copy()

if verbose:
    print("Starting changepoint processing with predicted clusters...")

# Process with clusters as predicted by the CART process
start_time = time.time()

# Create a data frame with the parameters of changepoint models for each predicted cluster
unique_cluster_hours_pred = sorted(dat_clean['ClusterHour_PRED'].unique())
changepoint_pars_summ_clust_pred = pd.DataFrame({
    'ClusterHour_PRED': unique_cluster_hours_pred,
    'slope_Temp': np.nan,
    'slope_Irrad': np.nan,
    'intercept': np.nan,
    'minimum': np.nan
})

# Update data frame with final format
dat_clean['Power_fitted'] = np.nan
dat_clean['Power_residuals'] = np.nan
dat_clean['IS_Outlier'] = np.nan

# Empty dataframe for output
dat_output = pd.DataFrame(columns=dat_clean.columns)

for i, cluster_hour_pred in enumerate(unique_cluster_hours_pred):
    dat_subs_clust = dat_clean[dat_clean['ClusterHour_PRED'] == cluster_hour_pred].copy()
    
    if verbose:
        print(f"Processing predicted cluster hour {cluster_hour_pred} ({i+1}/{len(unique_cluster_hours_pred)}) with {len(dat_subs_clust)} observations")
    
    # Get the optimal changepoint function parameters for each subset
    dat_processed, params = Detect_Outliers_Model_Based_CH_3P(
        dat_subs_clust, threshold_outlier=1.96, pop_size=50, max_iter=100, verbose=False
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
    idx = changepoint_pars_summ_clust_pred.index[changepoint_pars_summ_clust_pred['ClusterHour_PRED'] == cluster_hour_pred]
    if len(idx) == 1:
        changepoint_pars_summ_clust_pred.loc[idx, ['slope_Temp', 'slope_Irrad', 'intercept', 'minimum']] = params
    
    # Control process
    if verbose:
        current_time = time.time()
        elapsed_time = current_time - start_time
        estimated_time = elapsed_time * (len(unique_cluster_hours_pred) / (i + 1))
        print(f"Progress: {i+1} / {len(unique_cluster_hours_pred)}. "
              f"elapsed time: {elapsed_time:.2f} sec, "
              f"estimated time: {estimated_time:.2f} sec")

# Reorder output dataframe by date
dat_clean = dat_output.sort_values('DATE').reset_index(drop=True)
dat_clean['Power_fitted_ClustH_PRED'] = dat_clean['Power_fitted'].copy()

# Write to file
if verbose:
    print("Saving results...")

changepoint_pars_summ_clust.to_csv(output_clust_params, sep=";", index=False)
changepoint_pars_summ_clust_pred.to_csv(output_clust_pred_params, sep=";", index=False)

if verbose:
    print("Changepoint clustering analysis completed successfully!")
    print(f"Outputs saved:")
    print(f"  - Cluster parameters: {output_clust_params}")
    print(f"  - Predicted cluster parameters: {output_clust_pred_params}")
