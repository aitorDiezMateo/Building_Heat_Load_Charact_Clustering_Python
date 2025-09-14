import sys
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid Qt issues
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns
import joblib

if len(sys.argv) < 9:
    raise SystemExit(
        "Usage: 08_CART.py <input_data.csv> <clusters.csv> <confusion_matrix.csv> <tree_plot.jpg> <comparison_plot.jpg> <accuracy_plot.jpg> <report.csv> <model.pkl> [--verbose]"
    )

input_data_file = sys.argv[1]  
input_cluster_file = sys.argv[2]  
output_confusion_matrix = sys.argv[3]
output_cart_plot1 = sys.argv[4]   
output_cart_plot2 = sys.argv[5]  
output_cart_plot3 = sys.argv[6]  
output_classification_report_csv = sys.argv[7]  
output_model_file = sys.argv[8]  
verbose = False
if len(sys.argv) > 9 and sys.argv[9] in ("--verbose", "-v", "true", "True", "1"):
    verbose = True

# Extract output directory from the first output file
output_dir = os.path.dirname(output_confusion_matrix)

# Load the data
if verbose:
    print("Loading data for CART model development...")
dat = pd.read_csv(input_data_file, sep=";")
dat_cluster = pd.read_csv(input_cluster_file, sep=";")

# Ensure DATE column is datetime
if 'Date' in dat.columns and 'DATE' not in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['Date'])
elif 'DATE' in dat.columns:
    dat['DATE'] = pd.to_datetime(dat['DATE'])

# Create data frame with the required information
if verbose:
    print("Aggregating data into daily summary statistics...")

# Summarize data by daily values
dat_day_cart = dat.groupby('DATE_YYYY_MM_DD').agg({
    'Temperature': 'mean',  # Average daily temperature
    'Solar_Irradiation': 'sum',  # Total (Cumulated) daily solar irradiation
    'DATE_day_year': 'first',
    'DATE_day_week': 'first', 
    'DATE_weekday': 'first',
    'Holiday': 'first'
}).reset_index()

# Convert DATE_YYYY_MM_DD to datetime for merging
dat_day_cart['DATE'] = pd.to_datetime(dat_day_cart['DATE_YYYY_MM_DD'])

# Merge with cluster information from the previous process
if verbose:
    print("Integrating cluster assignment information...")
dat_cluster['DATE'] = pd.to_datetime(dat_cluster['DATE_YYYY_MM_DD'])

# Join the summarized daily data with cluster assignments
dat_day_cart = dat_day_cart.merge(
    dat_cluster[['DATE', 'cluster']], 
    on='DATE', 
    how='left'
)

# Ensure that all the observations have data from both processes
dat_day_cart = dat_day_cart.dropna()

# Convert variables to appropriate types (factors)
dat_day_cart['Holiday'] = dat_day_cart['Holiday'].astype('category')
dat_day_cart['cluster'] = dat_day_cart['cluster'].astype('category')
dat_day_cart['DATE_weekday'] = dat_day_cart['DATE_weekday'].astype('category')

if verbose:
    print(f"Final dataset shape: {dat_day_cart.shape}")
    print(f"Cluster distribution:\n{dat_day_cart['cluster'].value_counts().sort_index()}")

# Attribute clusters
if verbose:
    print("Constructing Classification and Regression Tree model...")

# Prepare features for the model
features = ['Temperature', 'Solar_Irradiation', 'Holiday', 'DATE_day_year', 'DATE_day_week', 'DATE_weekday']
X = dat_day_cart[features].copy()

# Convert categorical variables to dummy variables
X = pd.get_dummies(X, columns=['Holiday', 'DATE_weekday'], drop_first=False)

y = dat_day_cart['cluster']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if verbose:
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

# Build the CART model
cart_model = DecisionTreeClassifier(
    criterion='gini',  # Similar to R's default
    min_samples_split=5,  # minsplit in R
    min_samples_leaf=5,   # minbucket in R
    max_depth=5,          # maxdepth in R
    random_state=42
)

# Fit the model
cart_model.fit(X_train, y_train)

# Save the trained model
if verbose:
    print("Persisting trained CART model to disk...")
model_data = {
    'model': cart_model,
    'feature_names': X.columns.tolist(),
    'class_names': sorted(y.unique()),
    'training_info': {
        'train_size': X_train.shape[0],
        'test_size': X_test.shape[0],
        'features': features,
        'random_state': 42
    }
}
joblib.dump(model_data, output_model_file)

# Predict the cluster for all data
cluster_pred = cart_model.predict(X)
dat_day_cart['cluster_PRED'] = cluster_pred

if verbose:
    print("Model training process completed successfully.")

# Inspection
if verbose:
    print("Conducting model performance evaluation...")

# Make predictions on the test data
y_pred = cart_model.predict(X_test)

# Create a confusion matrix to evaluate the model
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

if verbose:
    print(f"Model accuracy: {accuracy:.4f}")
    print("Confusion matrix:")
    print(cm)

# Save confusion matrix
cm_df = pd.DataFrame(cm, 
                     index=[f'Actual_{i}' for i in sorted(y.unique())],
                     columns=[f'Predicted_{i}' for i in sorted(y.unique())])
cm_df.to_csv(output_confusion_matrix, sep=";")

# Classification report
report = classification_report(y_test, y_pred)
if verbose:
    print("Classification report:")
    print(report)

# Save classification report as CSV
report_dict = classification_report(y_test, y_pred, output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()
# Round numeric columns to 2 decimal places
numeric_columns = ['precision', 'recall', 'f1-score']
for col in numeric_columns:
    if col in report_df.columns:
        report_df[col] = report_df[col].round(2)
report_df.to_csv(output_classification_report_csv, sep=";")

# Visualize the decision tree
if verbose:
    print("Generating model visualization outputs...")
plt.figure(figsize=(20, 12))
plot_tree(cart_model, 
          feature_names=X.columns,
          class_names=[str(c) for c in sorted(y.unique())],
          filled=True,
          rounded=True,
          fontsize=10)
plt.title("CART Decision Tree", fontsize=16)
plt.tight_layout()
plt.savefig(output_cart_plot1, dpi=150, bbox_inches='tight')
plt.close()

# Plot: Actual vs Predicted clusters over time
plt.figure(figsize=(12, 6))
plt.scatter(dat_day_cart['DATE'], dat_day_cart['cluster'].astype(int), 
           alpha=0.6, label='Actual', s=30)
plt.scatter(dat_day_cart['DATE'], dat_day_cart['cluster_PRED'].astype(int), 
           alpha=0.6, color='red', label='Predicted', s=30)
plt.xlabel('Date')
plt.ylabel('Cluster')
plt.title('Actual vs Predicted Clusters Over Time')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_cart_plot2, dpi=150, bbox_inches='tight')
plt.close()

# Plot: Prediction accuracy over time
correct_predictions = dat_day_cart['cluster'] == dat_day_cart['cluster_PRED']
plt.figure(figsize=(12, 6))
plt.scatter(dat_day_cart['DATE'], correct_predictions.astype(int), 
           alpha=0.6, s=30)
plt.xlabel('Date')
plt.ylabel('Correct Prediction (1=Yes, 0=No)')
plt.title('Prediction Accuracy Over Time')
plt.yticks([0, 1], ['Incorrect', 'Correct'])
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(output_cart_plot3, dpi=150, bbox_inches='tight')
plt.close()

if verbose:
    print("CART analysis completed successfully!")
    print("Model outputs saved:")
    print(f"  - Confusion matrix: {output_confusion_matrix}")
    print(f"  - Classification report: {output_classification_report_csv}")
    print(f"  - Decision tree visualization: {output_cart_plot1}")
    print(f"  - Prediction comparison plot: {output_cart_plot2}")
    print(f"  - Temporal accuracy plot: {output_cart_plot3}")
    print(f"  - Trained model file: {output_model_file}")
