# -------------------------------------------------------------------------
# AUTHOR: Daniel Appel
# FILENAME: PCA
# SPECIFICATION: A program for performing PCA
# FOR: CS 5990 (Advanced Data Mining) - Assignment #2
# TIME SPENT: how long it took you to complete the assignment
# -----------------------------------------------------------*/
import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Get the absolute path of the current script (pca.py)
base_path = os.path.dirname(os.path.abspath(__file__))

# Construct the correct path to the CSV file
file_path = os.path.join(base_path, "..", "data", "heart_disease_dataset.csv")

# Load the dataset
df = pd.read_csv(file_path)

# Remove the target variable ('Heart Disease') if it exists
if 'Heart Disease' in df.columns:
    df_features = df.drop(columns=['Heart Disease'])
else:
    df_features = df.copy()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df_features)

# Get the number of features
num_features = df_features.shape[1]

# Store PC1 variances
pc1_variances = {}

# Run PCA for each feature removal
for i in range(num_features):
    # Create a new dataset by dropping the i-th feature
    reduced_data = np.delete(scaled_data, i, axis=1)
    
    # Run PCA
    pca = PCA()
    pca.fit(reduced_data)
    
    # Store PC1 variance and the feature removed
    pc1_variances[df_features.columns[i]] = pca.explained_variance_ratio_[0]

# Find the feature removal that gives the highest PC1 variance
max_feature = max(pc1_variances, key=pc1_variances.get)
max_variance = pc1_variances[max_feature]

# Print results
print(f'Highest PC1 variance found: {max_variance:.4f} when removing {max_feature}')