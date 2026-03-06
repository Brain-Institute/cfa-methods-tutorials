"""
Clustering model implementation using scikit-learn with integrated feature processing.
"""

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Union, Any
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# Import feature processing components
try:
    from ..processing.feature_processor import PreprocessingConfig, FeatureProcessor
    FEATURE_PROCESSING_AVAILABLE = True
except ImportError:
    FEATURE_PROCESSING_AVAILABLE = False
    PreprocessingConfig = None
    FeatureProcessor = None

# Import ML functions for clustering
try:
    from ..ml_functions.kmeans import run_kmeans
    from ..ml_functions.hierarchical import run_hierarchical
    from ..ml_functions.dbscan import run_dbscan
    ML_FUNCTIONS_AVAILABLE = True
except ImportError:
    ML_FUNCTIONS_AVAILABLE = False
    run_kmeans = None
    run_hierarchical = None
    run_dbscan = None

@dataclass
class KMeansSettings:
    n_clusters: int = 3
    init: str = "k-means++"
    n_init: int = 10
    max_iter: int = 300
    random_state: Optional[int] = 42

@dataclass
class HierarchicalSettings:
    n_clusters: int = 3
    linkage: str = "ward"
    metric: str = "euclidean"

@dataclass
class DBSSCANSettings:
    eps: float = 0.5
    min_samples: int = 5
    metric: str = "euclidean"

@dataclass
class ClusteringConfig:
    name: str
    method: str  # "kmeans", "hierarchical", or "dbscan"
    kmeans_settings: Optional[KMeansSettings] = None
    hierarchical_settings: Optional[HierarchicalSettings] = None
    dbscan_settings: Optional[DBSSCANSettings] = None
    fusion_run_id: Optional[str] = None
    fusion_run_name: Optional[str] = None

    # Feature selection and processing configuration
    selected_features: Optional[List[str]] = None  # List of selected feature names
    feature_processing_configs: Optional[Dict[str, Dict[str, Any]]] = None  # feature_name -> processing config
    processing_mode: str = "no_hpo"  # "with_hpo" or "no_hpo"
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary for storage"""
        config_dict = {
            "name": self.name,
            "method": self.method,
            "fusion_run_id": self.fusion_run_id,
            "fusion_run_name": self.fusion_run_name,
            "processing_mode": self.processing_mode
        }

        # Add method-specific settings
        if self.kmeans_settings:
            config_dict["kmeans_settings"] = asdict(self.kmeans_settings)
        if self.hierarchical_settings:
            config_dict["hierarchical_settings"] = asdict(self.hierarchical_settings)
        if self.dbscan_settings:
            config_dict["dbscan_settings"] = asdict(self.dbscan_settings)

        # Add selected features
        if self.selected_features:
            config_dict["selected_features"] = self.selected_features

        # Add feature processing configurations
        if self.feature_processing_configs:
            config_dict["feature_processing_configs"] = self.feature_processing_configs

        return config_dict

    @classmethod
    def from_dict(cls, data: Dict) -> 'ClusteringConfig':
        """Create a ClusteringConfig object from a dictionary."""
        method = data.get("method")
        kmeans_settings = None
        hierarchical_settings = None
        dbscan_settings = None
        selected_features = data.get("selected_features")

        if method == "kmeans" and "kmeans_settings" in data:
            kmeans_settings = KMeansSettings(**data["kmeans_settings"])
        elif method == "hierarchical" and "hierarchical_settings" in data:
            # Handle potential 'affinity' vs 'metric' discrepancy if needed
            h_settings_dict = data["hierarchical_settings"]
            if "affinity" in h_settings_dict and "metric" not in h_settings_dict:
                 h_settings_dict["metric"] = h_settings_dict.pop("affinity")
            hierarchical_settings = HierarchicalSettings(**h_settings_dict)
        elif method == "dbscan" and "dbscan_settings" in data:
            dbscan_settings = DBSSCANSettings(**data["dbscan_settings"])

        return cls(
            name=data.get("name", "Unnamed"),
            method=method,
            kmeans_settings=kmeans_settings,
            hierarchical_settings=hierarchical_settings,
            dbscan_settings=dbscan_settings,
            fusion_run_id=data.get("fusion_run_id"),
            fusion_run_name=data.get("fusion_run_name"),
            selected_features=selected_features,
            feature_processing_configs=data.get("feature_processing_configs"),
            processing_mode=data.get("processing_mode", "no_hpo")
        )

class ClusteringModel:
    """A class for performing clustering analysis on fused data with integrated feature processing."""

    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_processor = None
        self.processed_feature_info = {}
    
    def combine_feature_data(self, clustering_input: Dict) -> Dict:
        """
        Combine feature data from fusion results into a 2D array suitable for clustering.
        
        Args:
            clustering_input: Dictionary containing feature data from fusion results
            
        Returns:
            dict: Dictionary containing combined feature data and metadata
        """
        # Get feature data and names
        feature_data = clustering_input["feature_data"]
        feature_names = clustering_input["feature_names"]
        feature_types = clustering_input["feature_types"]
        
        # Combine feature data into a 2D array
        combined_data = np.hstack(feature_data)
        
        # Create a mapping of column indices to feature names
        col_to_feature = {}
        current_col = 0
        for i, (name, data) in enumerate(zip(feature_names, feature_data)):
            n_cols = data.shape[1]
            for j in range(n_cols):
                col_to_feature[current_col + j] = (name, j)
            current_col += n_cols
        
        return {
            "data": combined_data,
            "feature_names": feature_names,
            "feature_types": feature_types,
            "col_to_feature": col_to_feature
        }

    def apply_feature_processing(self, clustering_input: Dict, config: ClusteringConfig) -> Dict:
        """
        Apply feature processing pipeline to clustering input data.

        Args:
            clustering_input: Dictionary containing feature data from fusion results
            config: ClusteringConfig with feature processing configurations

        Returns:
            dict: Dictionary containing processed feature data
        """
        if not FEATURE_PROCESSING_AVAILABLE:
            print("Warning: Feature processing not available, using raw data")
            return clustering_input

        if not config.feature_processing_configs:
            print("No feature processing configurations found, using raw data")
            return clustering_input

        processed_input = clustering_input.copy()
        processed_feature_data = []
        processed_feature_names = []
        processed_feature_types = []

        feature_data = clustering_input["feature_data"]
        feature_names = clustering_input["feature_names"]
        feature_types = clustering_input["feature_types"]

        print(f"Applying feature processing to {len(feature_names)} features...")

        for i, (feature_name, data, feature_type) in enumerate(zip(feature_names, feature_data, feature_types)):
            # Check if this feature has processing configuration
            if feature_name in config.feature_processing_configs:
                processing_config = config.feature_processing_configs[feature_name]

                try:
                    # Convert data to DataFrame for processing
                    if isinstance(data, np.ndarray):
                        df = pd.DataFrame(data, columns=[f"{feature_name}_{j}" for j in range(data.shape[1])])
                    else:
                        df = pd.DataFrame(data)

                    # Convert to FeatureProcessingSettings first, then to PreprocessingConfig
                    from ..models.feature_processing import FeatureProcessingSettings
                    feature_settings = FeatureProcessingSettings.from_dict(processing_config)

                    # Convert to PreprocessingConfig format
                    preprocessing_config = PreprocessingConfig(
                        transform={"kind": feature_settings.transform_method} if feature_settings.transform_method != "none" else None,
                        standardize={"with_mean": True, "with_std": True} if feature_settings.standardize else None,
                        pca={"n_components": feature_settings.pca_num_components} if feature_settings.pca_enabled else None
                    )

                    # Create and apply feature processor
                    processor = FeatureProcessor(feature_settings)

                    # Apply processing (no_hpo mode - all steps applied together)
                    processed_result = processor.fit_transform(df)

                    # Convert back to numpy array
                    if hasattr(processed_result, 'values'):
                        processed_data = processed_result.values
                    else:
                        processed_data = processed_result

                    processed_feature_data.append(processed_data)
                    processed_feature_names.append(feature_name)
                    processed_feature_types.append(feature_type)

                    # Store processing info
                    self.processed_feature_info[feature_name] = {
                        'original_shape': data.shape,
                        'processed_shape': processed_data.shape,
                        'processing_config': processing_config
                    }

                except Exception as e:
                    # Fall back to original data
                    processed_feature_data.append(data)
                    processed_feature_names.append(feature_name)
                    processed_feature_types.append(feature_type)
            else:
                # No processing configuration, use original data
                processed_feature_data.append(data)
                processed_feature_names.append(feature_name)
                processed_feature_types.append(feature_type)
                print(f"  → Using raw data for {feature_name}: {data.shape}")

        # Update the clustering input with processed data
        processed_input["feature_data"] = processed_feature_data
        processed_input["feature_names"] = processed_feature_names
        processed_input["feature_types"] = processed_feature_types

        return processed_input
    
    def run(self, clustering_input: Dict, config: Union[ClusteringConfig, List[ClusteringConfig]], 
           output_dir: str, run_id: Optional[str] = None) -> str:
        """
        Run clustering analyses on the fused data.
        
        Args:
            clustering_input: Dictionary containing feature data from fusion results
            config: ClusteringConfig object or list of ClusteringConfig objects
            output_dir: Base directory containing clustering runs
            run_id: Optional run_id to use (if None, a new one will be generated)
            
        Returns:
            str: The run_id of the clustering analysis
        """
        if not clustering_input or "feature_data" not in clustering_input:
            raise ValueError("No feature data provided for clustering")
        
        # Convert single config to list for consistent processing
        configs = [config] if not isinstance(config, list) else config
        
        # Use provided run_id or generate a new one
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = os.path.join(output_dir, run_id)
        os.makedirs(run_dir, exist_ok=True)

        # Initialize results dictionary
        results = {
            "configs": [asdict(config) for config in configs],
            "clustering_results": {},
            "feature_processing_info": {}
        }

        # Run each clustering method
        for config in configs:
            print(f"\nProcessing clustering configuration: {config.name}")

            # Apply feature processing if configured
            processed_input = self.apply_feature_processing(clustering_input, config)

            # Prepare data
            data_dict = self.combine_feature_data(processed_input)
            X = data_dict["data"]

            # Standardize the data (note: feature processing may have already done this)
            X_scaled = self.scaler.fit_transform(X)

            # Store data info for this configuration
            if "scaled_data" not in results:  # Store from first config
                results.update({
                    "scaled_data": X_scaled,
                    "feature_names": data_dict["feature_names"],
                    "feature_types": data_dict["feature_types"],
                    "col_to_feature": data_dict["col_to_feature"]
                })

            # Store feature processing info
            results["feature_processing_info"][config.name] = self.processed_feature_info.copy()

            # Determine if we should skip preprocessing in ml_functions
            # Skip if we have feature processing configs (data is already processed)
            skip_preprocessing = bool(config.feature_processing_configs)

            try:
                # Use ml_functions for clustering with proper preprocessing control
                if config.method == "kmeans":
                    if not config.kmeans_settings:
                        raise ValueError("K-means settings not provided")

                    if ML_FUNCTIONS_AVAILABLE:
                        # Use ml_functions with preprocessing control
                        settings = asdict(config.kmeans_settings)
                        clustering_results = run_kmeans(X_scaled, skip_preprocessing=skip_preprocessing, **settings)
                        labels = clustering_results["labels"]
                        metrics = clustering_results.get("metrics", {})
                    else:
                        # Fallback to direct sklearn
                        self.model = KMeans(**asdict(config.kmeans_settings))
                        labels = self.model.fit_predict(X_scaled)
                        metrics = {}
                    
                elif config.method == "hierarchical":
                    if not config.hierarchical_settings:
                        raise ValueError("Hierarchical clustering settings not provided")

                    if ML_FUNCTIONS_AVAILABLE:
                        # Use ml_functions with preprocessing control
                        settings = asdict(config.hierarchical_settings)
                        # Convert affinity to metric for compatibility
                        if "affinity" in settings:
                            settings["metric"] = settings.pop("affinity")
                        clustering_results = run_hierarchical(X_scaled, skip_preprocessing=skip_preprocessing, **settings)
                        labels = clustering_results["labels"]
                        metrics = clustering_results.get("metrics", {})
                    else:
                        # Fallback to direct sklearn
                        settings = asdict(config.hierarchical_settings)
                        if "affinity" in settings:
                            settings["metric"] = settings.pop("affinity")
                        self.model = AgglomerativeClustering(**settings)
                        labels = self.model.fit_predict(X_scaled)
                        metrics = {}
                    
                elif config.method == "dbscan":
                    if not config.dbscan_settings:
                        raise ValueError("DBSCAN settings not provided")

                    if ML_FUNCTIONS_AVAILABLE:
                        # Use ml_functions with preprocessing control
                        settings = asdict(config.dbscan_settings)
                        clustering_results = run_dbscan(X_scaled, skip_preprocessing=skip_preprocessing, **settings)
                        labels = clustering_results["labels"]
                        metrics = clustering_results.get("metrics", {})
                        n_clusters = clustering_results.get("n_clusters", 0)
                        n_noise = list(labels).count(-1)
                    else:
                        # Fallback to direct sklearn
                        self.model = DBSCAN(**asdict(config.dbscan_settings))
                        labels = self.model.fit_predict(X_scaled)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_noise = list(labels).count(-1)
                        metrics = {}
                    
                else:
                    raise ValueError(f"Unsupported clustering method: {config.method}")
                
                # Calculate clustering quality metrics if not already provided by ml_functions
                if not metrics:  # Only calculate if ml_functions didn't provide them
                    if config.method != "dbscan":  # DBSCAN doesn't guarantee clusters
                        if len(np.unique(labels)) > 1:  # Need at least 2 clusters for metrics
                            try:
                                metrics["silhouette_score"] = float(silhouette_score(X_scaled, labels))
                                metrics["calinski_harabasz_score"] = float(calinski_harabasz_score(X_scaled, labels))
                                metrics["davies_bouldin_score"] = float(davies_bouldin_score(X_scaled, labels))
                            except Exception as e:
                                print(f"Warning: Could not calculate metrics for {config.name}: {str(e)}")
                    else:
                        metrics["n_clusters"] = int(n_clusters)
                        metrics["n_noise"] = int(n_noise)
                
                # Store results for this method
                results["clustering_results"][config.name] = {
                    "labels": labels.tolist(),  # Convert to list immediately
                    "metrics": metrics
                }
                
            except Exception as e:
                print(f"Error in {config.method} clustering: {str(e)}")
                continue
        
        # Save results
        # Save numerical results
        results_path = os.path.join(run_dir, "results.npz")
        np.savez(results_path, 
                 scaled_data=results["scaled_data"],
                 feature_names=results["feature_names"],
                 feature_types=results["feature_types"],
                 col_to_feature=results["col_to_feature"])
        
        # Save clustering results separately
        clustering_results_path = os.path.join(run_dir, "clustering_results.npz")
        # Convert all numpy arrays to lists before saving
        clustering_results = {}
        for method_name, method_results in results["clustering_results"].items():
            clustering_results[method_name] = method_results
        np.savez(clustering_results_path, **clustering_results)
        
        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "configs": [asdict(config) for config in configs]
            }, f, indent=4)
        
        # Save metadata with only run information
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump({
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "fusion_run_id": configs[0].fusion_run_id if configs else None
            }, f, indent=4)
            
        return run_id
    
    def load_results(self, output_dir: str, run_id: str) -> Dict:
        """
        Load clustering results from a directory.
        
        Args:
            output_dir: Base directory containing clustering results
            run_id: The run_id of the clustering analysis to load
            
        Returns:
            dict: Dictionary containing loaded results
        """
        run_dir = os.path.join(output_dir, run_id)
        
        # Check for results.npz
        results_path = os.path.join(run_dir, "results.npz")
        if not os.path.exists(results_path):
            # Return a minimal valid result structure instead of failing
            print(f"Note: Detailed clustering results not found for run {run_id}. Returning basic information.")
            
            # Create minimal results dictionary with required keys
            minimal_results = {
                "scaled_data": np.array([]),
                "feature_names": np.array([]),
                "feature_types": np.array([]),
                "col_to_feature": {},
                "clustering_results": {}
            }
            
            # Check if there's a metadata file we can at least show
            metadata_path = os.path.join(run_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    minimal_results["metadata"] = metadata
                except Exception as e:
                    print(f"Error reading metadata: {str(e)}")
                    
            # Check if there's a config file
            config_path = os.path.join(run_dir, "config.json")
            if os.path.exists(config_path):
                try:
                    with open(config_path, 'r') as f:
                        config = json.load(f)
                    minimal_results["config"] = config
                except Exception as e:
                    print(f"Error reading config: {str(e)}")
            
            return minimal_results
        
        # If results.npz exists, load it
        results = dict(np.load(results_path, allow_pickle=True))
        
        # Load clustering results
        clustering_results_path = os.path.join(run_dir, "clustering_results.npz")
        if os.path.exists(clustering_results_path):
            loaded_data = np.load(clustering_results_path, allow_pickle=True)
            clustering_results = {}
            
            # Convert numpy arrays to dictionaries
            for method_name in loaded_data.files:
                method_data = loaded_data[method_name].item()  # Convert numpy array to dict
                clustering_results[method_name] = method_data
            
            results["clustering_results"] = clustering_results
        else:
            results["clustering_results"] = {}
        
        return results
    
    def print_results(self, results: Dict):
        """
        Print clustering results and metrics in a concise format.
        
        Args:
            results: Dictionary containing clustering results
        """
        print("\nClustering Results:")
        print("-" * 50)
        
        # First check if we have a minimal result structure
        if "metadata" in results and not results.get("scaled_data", np.array([])).size:
            print("Clustering completed, but detailed results are not available.")
            print(f"Run ID: {results.get('metadata', {}).get('run_id', 'unknown')}")
            print(f"Timestamp: {results.get('metadata', {}).get('timestamp', 'unknown')}")
            
            # Print config information if available
            if "config" in results:
                print("\nConfiguration:")
                configs = results.get("config", {}).get("configs", [])
                for i, config in enumerate(configs):
                    print(f"Method {i+1}: {config.get('method', 'unknown')}")
            
            print("-" * 50)
            return
        
        # If we have full results, continue with normal printing
        # Print basic information
        if "scaled_data" in results and hasattr(results["scaled_data"], "shape"):
            print(f"Number of samples: {results['scaled_data'].shape[0]}")
            print(f"Number of features: {results['scaled_data'].shape[1]}")
        else:
            print("Sample and feature data not available")
        
        # Print results for each clustering method
        if "clustering_results" in results:
            for method_name, method_results in results["clustering_results"].items():
                print(f"\n{method_name}:")
                if "labels" in method_results:
                    n_clusters = len(set(method_results["labels"])) - (1 if -1 in method_results["labels"] else 0)
                    print(f"Number of clusters: {n_clusters}")
                    if -1 in method_results["labels"]:
                        n_noise = method_results["labels"].count(-1)
                        print(f"Number of noise points: {n_noise}")
                
                if "metrics" in method_results:
                    metrics = method_results["metrics"]
                    if metrics:  # Only print if there are metrics
                        print("Metrics:")
                        for metric_name, value in metrics.items():
                            if isinstance(value, (int, float)):
                                print(f"  {metric_name}: {value:.4f}")
                            else:
                                print(f"  {metric_name}: {value}")
                    else:
                        print("No metrics available")
                else:
                    print("No metrics available")
        else:
            print("\nNo clustering results available")
        
        # Print feature information
        if "feature_names" in results and "feature_types" in results:
            if len(results["feature_names"]) > 0:
                print("\nFeature Information:")
                for name, ftype in zip(results["feature_names"], results["feature_types"]):
                    print(f"{name} ({ftype})")
        
        print("-" * 50) 