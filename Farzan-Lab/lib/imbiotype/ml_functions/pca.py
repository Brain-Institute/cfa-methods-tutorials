"""
PCA implementation using scikit-learn.
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

def run_pca(data, settings=None):
    """
    Run PCA on the input data.
    
    Args:
        data (np.ndarray): Input data matrix of shape (n_samples, n_features)
        settings (PCASettings, optional): PCA settings object. If None, uses default settings.
    
    Returns:
        dict: Dictionary containing:
            - components: PCA components (n_features, n_components)
            - explained_variance_ratio: Total explained variance ratio (float)
            - explained_variance_ratios: Explained variance ratio for each component (array)
            - transformed_data: Transformed data (n_samples, n_components)
    """
    # Handle missing values first
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data)
    
    # Standardize the data
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_imputed)
    
    # Determine number of components based on settings
    if settings is None or settings.comp_cutoff_method == "explained_variance":
        # First fit PCA with all components
        pca_full = PCA()
        pca_full.fit(data_scaled)
        
        # Find number of components needed to explain variance
        cumsum = np.cumsum(pca_full.explained_variance_ratio_)
        target_variance = settings.explained_variance if settings else 0.95
        n_components = np.argmax(cumsum >= target_variance) + 1
        
        # Refit with determined number of components
        pca = PCA(n_components=n_components)
    elif settings.comp_cutoff_method == "num_components":
        pca = PCA(n_components=settings.num_components)
    else:  # keep_all
        pca = PCA()
    
    # Fit and transform the data
    transformed_data = pca.fit_transform(data_scaled)
    
    return {
        'components': pca.components_.T,  # Transpose to match expected shape
        'explained_variance_ratio': float(np.sum(pca.explained_variance_ratio_)),  # Total ratio as float
        'explained_variance_ratios': pca.explained_variance_ratio_,  # Individual ratios as array
        'transformed_data': transformed_data
    } 