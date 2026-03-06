"""
DBSCAN clustering implementation using scikit-learn.
"""

import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def run_dbscan(data, eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, skip_preprocessing=False):
    """
    Run DBSCAN clustering on the input data.
    
    Args:
        data (np.ndarray): Input data matrix of shape (n_samples, n_features)
        eps (float, optional): Maximum distance between samples for neighborhood. Defaults to 0.5.
        min_samples (int, optional): Minimum number of samples in a neighborhood. Defaults to 5.
        metric (str, optional): Metric to compute distances. Defaults to 'euclidean'.
        metric_params (dict, optional): Additional keyword arguments for the metric. Defaults to None.
        algorithm (str, optional): Algorithm to compute nearest neighbors. Defaults to 'auto'.
        leaf_size (int, optional): Leaf size for BallTree or KDTree. Defaults to 30.
        p (float, optional): Power of the Minkowski metric. Defaults to None.
        skip_preprocessing (bool, optional): Skip imputation and standardization if data is already processed. Defaults to False.
    
    Returns:
        dict: Dictionary containing:
            - labels: Cluster labels for each sample (-1 for noise points)
            - core_sample_indices: Indices of core samples
            - components: Core samples
            - n_clusters: Number of clusters (excluding noise)
            - metrics: Dictionary of clustering quality metrics (only for non-noise points)
                - silhouette: Silhouette score
                - calinski_harabasz: Calinski-Harabasz score
                - davies_bouldin: Davies-Bouldin score
    """
    # Apply preprocessing if not skipped
    if skip_preprocessing:
        # Data is already processed, use as-is
        data_scaled = data
        print("Skipping preprocessing - using pre-processed data")
    else:
        # Handle missing values
        imputer = SimpleImputer(strategy='mean')
        data_imputed = imputer.fit_transform(data)

        # Standardize the data
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_imputed)
    
    # Initialize and fit DBSCAN
    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples,
        metric=metric,
        metric_params=metric_params,
        algorithm=algorithm,
        leaf_size=leaf_size,
        p=p
    )
    
    # Fit and predict
    labels = dbscan.fit_predict(data_scaled)
    
    # Calculate metrics only for non-noise points
    non_noise_mask = labels != -1
    if np.sum(non_noise_mask) > 1:  # Need at least 2 samples for metrics
        metrics = {
            'silhouette': silhouette_score(data_scaled[non_noise_mask], labels[non_noise_mask]),
            'calinski_harabasz': calinski_harabasz_score(data_scaled[non_noise_mask], labels[non_noise_mask]),
            'davies_bouldin': davies_bouldin_score(data_scaled[non_noise_mask], labels[non_noise_mask])
        }
    else:
        metrics = {
            'silhouette': None,
            'calinski_harabasz': None,
            'davies_bouldin': None
        }
    
    return {
        'labels': labels,
        'core_sample_indices': dbscan.core_sample_indices_,
        'components': dbscan.components_,
        'n_clusters': len(set(labels)) - (1 if -1 in labels else 0),
        'metrics': metrics
    } 