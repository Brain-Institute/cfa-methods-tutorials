"""
K-means clustering implementation using scikit-learn.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def run_kmeans(data, n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-4, random_state=None, skip_preprocessing=False):
    """
    Run K-means clustering on the input data.
    
    Args:
        data (np.ndarray): Input data matrix of shape (n_samples, n_features)
        n_clusters (int, optional): Number of clusters to form. Defaults to 3.
        init (str, optional): Method for initialization. Defaults to 'k-means++'.
        n_init (int, optional): Number of times the k-means algorithm will be run. Defaults to 10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 300.
        tol (float, optional): Relative tolerance with regards to inertia. Defaults to 1e-4.
        random_state (int, optional): Random state for reproducibility. Defaults to None.
        skip_preprocessing (bool, optional): Skip imputation and standardization if data is already processed. Defaults to False.
    
    Returns:
        dict: Dictionary containing:
            - labels: Cluster labels for each sample
            - cluster_centers: Coordinates of cluster centers
            - inertia: Sum of squared distances of samples to their closest cluster center
            - n_iter: Number of iterations run
            - metrics: Dictionary of clustering quality metrics
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
    
    # Initialize and fit K-means
    kmeans = KMeans(
        n_clusters=n_clusters,
        init=init,
        n_init=n_init,
        max_iter=max_iter,
        tol=tol,
        random_state=random_state
    )
    
    # Fit and predict
    labels = kmeans.fit_predict(data_scaled)
    
    # Calculate clustering quality metrics
    metrics = {
        'silhouette': silhouette_score(data_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(data_scaled, labels),
        'davies_bouldin': davies_bouldin_score(data_scaled, labels)
    }
    
    return {
        'labels': labels,
        'cluster_centers': kmeans.cluster_centers_,
        'inertia': kmeans.inertia_,
        'n_iter': kmeans.n_iter_,
        'metrics': metrics
    } 