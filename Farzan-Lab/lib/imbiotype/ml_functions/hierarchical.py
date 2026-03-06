"""
Hierarchical clustering implementation using scikit-learn.
"""

import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def run_hierarchical(data, n_clusters=3, linkage='ward', affinity='euclidean', compute_full_tree=True, distance_threshold=None, skip_preprocessing=False):
    """
    Run Hierarchical clustering on the input data.
    
    Args:
        data (np.ndarray): Input data matrix of shape (n_samples, n_features)
        n_clusters (int, optional): Number of clusters to find. Defaults to 3.
        linkage (str, optional): Linkage criterion. Defaults to 'ward'.
        affinity (str, optional): Metric used for distance computation. Defaults to 'euclidean'.
        compute_full_tree (bool, optional): Whether to compute the full tree. Defaults to True.
        distance_threshold (float, optional): Distance threshold for cutting the tree. Defaults to None.
        skip_preprocessing (bool, optional): Skip imputation and standardization if data is already processed. Defaults to False.
    
    Returns:
        dict: Dictionary containing:
            - labels: Cluster labels for each sample
            - n_leaves: Number of leaves in the hierarchical tree
            - n_connected_components: Number of connected components
            - children: Children of each non-leaf node
            - distances: Distances between nodes in the tree (if available)
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
    
    # Initialize and fit Hierarchical clustering
    hierarchical = AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage=linkage,
        affinity=affinity,
        compute_full_tree=compute_full_tree,
        distance_threshold=distance_threshold
    )
    
    # Fit and predict
    labels = hierarchical.fit_predict(data_scaled)
    
    # Calculate clustering quality metrics
    metrics = {
        'silhouette': silhouette_score(data_scaled, labels),
        'calinski_harabasz': calinski_harabasz_score(data_scaled, labels),
        'davies_bouldin': davies_bouldin_score(data_scaled, labels)
    }
    
    # Create results dictionary with required attributes
    results = {
        'labels': labels,
        'n_leaves': hierarchical.n_leaves_,
        'n_connected_components': hierarchical.n_connected_components_,
        'children': hierarchical.children_,
        'metrics': metrics
    }
    
    # Add distances if available (only when distance_threshold is set)
    if hasattr(hierarchical, 'distances_'):
        results['distances'] = hierarchical.distances_
    
    return results 