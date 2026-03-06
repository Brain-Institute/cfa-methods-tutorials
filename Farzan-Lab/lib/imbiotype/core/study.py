"""
Core Study class for managing IMBiotype analysis
"""

import os
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np

from .data_manager import DataManager
from ..models.fusion import FusionModel, FusionConfig
from ..models.clustering import ClusteringModel, ClusteringConfig
from ..pipeline.pipeline import Pipeline
from ..functions.preprocess import (
    apply_scaling, apply_imputation, select_columns, create_formula_column,
    filter_rows_by_condition, join_datasets, calculate_grouped_difference
)

class Study:
    """
    Main class for managing an IMBiotype study.
    Handles data import, feature management, and analysis pipelines.
    """
    
    def __init__(self, study_path: str):
        """
        Initialize a study
        
        Args:
            study_path: Path to the study directory
        """
        self.study_path = study_path
        self.metadata_path = os.path.join(study_path, "metadata.json")
        self.raw_data_path = os.path.join(study_path, "raw_data")
        self.fusion_path = os.path.join(study_path, "fusion")
        self.clustering_path = os.path.join(study_path, "clustering")
        self.evaluation_path = os.path.join(study_path, "evaluation")
        # Initialize DataManager with the base study path
        self.data_manager = DataManager(self.study_path)
        self.fusion_model = FusionModel()
        self.clustering_model = ClusteringModel()
        self._load_metadata()

        # Create necessary directories if they don't exist
        os.makedirs(self.fusion_path, exist_ok=True)
        os.makedirs(self.clustering_path, exist_ok=True)
        os.makedirs(self.evaluation_path, exist_ok=True)
        # Ensure raw_data and processed_data exist (might be redundant if created by Study.create)
        os.makedirs(self.raw_data_path, exist_ok=True)
        os.makedirs(os.path.join(self.study_path, "processed_data"), exist_ok=True)
        
    def _load_metadata(self):
        """Load study metadata"""
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, 'r') as f:
                self._metadata = json.load(f)
        else:
            self._metadata = {}
            
    def _save_metadata(self):
        """Save study metadata"""
        with open(self.metadata_path, 'w') as f:
            json.dump(self._metadata, f, indent=2)
            
    @classmethod
    def create(cls, study_path: str, study_name: Optional[str] = None) -> 'Study':
        """
        Create a new study
        
        Args:
            study_path: Path where the study should be created
            study_name: Optional name for the study
            
        Returns:
            Newly created Study instance
        """
        if not os.path.exists(study_path):
            os.makedirs(study_path)
            
        # Create raw_data directory
        raw_data_path = os.path.join(study_path, "raw_data")
        os.makedirs(raw_data_path, exist_ok=True)
        # Create processed_data directory
        processed_data_path = os.path.join(study_path, "processed_data")
        os.makedirs(processed_data_path, exist_ok=True)
        
        # Create metadata
        metadata = {
            "study_id": str(uuid.uuid4()),
            "study_name": study_name or os.path.basename(study_path),
            "created_at": datetime.now().isoformat(),
            "last_modified": datetime.now().isoformat()
        }
        
        # Save metadata
        metadata_path = os.path.join(study_path, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return cls(study_path)
        
    @classmethod
    def load(cls, study_path: str) -> 'Study':
        """
        Load an existing study
        
        Args:
            study_path: Path to the study directory
            
        Returns:
            Loaded Study instance
            
        Raises:
            ValueError: If study directory is invalid
        """
        if not os.path.exists(study_path):
            raise ValueError(f"Study directory not found: {study_path}")
            
        metadata_path = os.path.join(study_path, "metadata.json")
        if not os.path.exists(metadata_path):
            raise ValueError(f"Invalid study directory: {study_path}")
            
        return cls(study_path)
        
    def get_metadata(self) -> Dict:
        """Get study metadata"""
        return self._metadata.copy()
        
    def update_metadata(self, updates: Dict) -> None:
        """
        Update study metadata
        
        Args:
            updates: Dictionary of metadata updates
        """
        self._metadata.update(updates)
        self._metadata["last_modified"] = datetime.now().isoformat()
        self._save_metadata()
        
    def get_datasets(self) -> List[str]:
        """Get list of all datasets in the study"""
        return self.data_manager.get_datasets()
        
    def import_dataset(self, file_path: str, 
                      dataset_name: Optional[str] = None) -> str:
        """
        Import a dataset into the study
        
        Args:
            file_path: Path to the dataset file
            dataset_name: Optional name for the dataset
            
        Returns:
            Name of the imported dataset
        """
        dataset_name, overwritten = self.data_manager.import_dataset(file_path, dataset_name)
        self.update_metadata({
            "last_modified": datetime.now().isoformat()
        })
        return dataset_name, overwritten
        
    def preview_dataset(self, filename: str):
        """
        Preview a dataset's contents
        
        Args:
            filename: Name of the dataset to preview
            
        Returns:
            DataFrame with preview data
        """
        return self.data_manager.preview_dataset(filename)
        
    def check_dataset_quality(self, filename: str):
        """
        Check dataset quality metrics
        
        Args:
            filename: Name of the dataset to check
            
        Returns:
            Tuple of (no_missing_values, is_standardized)
        """
        return self.data_manager.check_dataset_quality(filename)
        
    def get_dataset_columns(self, filename: str):
        """
        Get column information for a dataset
        
        Args:
            filename: Dataset to get columns for
            
        Returns:
            List of column info dicts with name and type
        """
        return self.data_manager.get_dataset_columns(filename)
        
    def create_feature_vector(self, name: str) -> Dict:
        """
        Create a new feature vector
        
        Args:
            name: Name for the feature vector
            
        Returns:
            Feature vector metadata
        """
        vector = self.data_manager.create_feature_vector(name)
        self.update_metadata({
            "last_modified": datetime.now().isoformat()
        })
        return vector
        
    def add_features_to_vector(self, vector_name: str, 
                             features: List[Dict[str, str]]) -> None:
        """
        Add features to an existing vector
        
        Args:
            vector_name: Name of the target vector
            features: List of features to add, each with name and source
        """
        self.data_manager.add_features_to_vector(vector_name, features)
        self.update_metadata({
            "last_modified": datetime.now().isoformat()
        })
        
    def get_feature_vectors(self) -> List[Dict]:
        """Get all feature vectors"""
        return self.data_manager.get_feature_vectors()
        
    def load_feature_data(self, vector_name: str):
        """
        Load the data for a feature vector
        
        Args:
            vector_name: Name of the feature vector
            
        Returns:
            DataFrame containing the feature data
        """
        return self.data_manager.load_feature_data(vector_name)
        
    def get_features_by_prefix(self, vector_name: str, prefix: str) -> List[str]:
        """
        Get all feature names for a given prefix in a feature vector
        
        Args:
            vector_name: Name of the feature vector
            prefix: Feature prefix (e.g., 'x', 'y', 'z')
            
        Returns:
            List of feature names with the given prefix
        """
        return self.data_manager.get_features_by_prefix(vector_name, prefix)
        
    def create_feature(self, name: str, dataset: str, columns: List[str]) -> Dict:
        """
        Create a new feature from selected columns in a dataset
        
        Args:
            name: Name for the feature
            dataset: Name of the source dataset
            columns: List of column names to include
            
        Returns:
            Feature metadata
        """
        feature = self.data_manager.create_feature(name, dataset, columns)
        self.update_metadata({
            "last_modified": datetime.now().isoformat()
        })
        return feature
        
    def find_columns_by_pattern(self, dataset: str, pattern: str, 
                              pattern_type: str = "prefix") -> List[str]:
        """
        Find columns in a dataset matching a pattern
        
        Args:
            dataset: Name of the dataset to search
            pattern: Pattern to match against
            pattern_type: Type of pattern matching ("prefix", "suffix", or "contains")
            
        Returns:
            List of matching column names
        """
        return self.data_manager.find_columns_by_pattern(dataset, pattern, pattern_type)
        
    def create_feature_simple(self, name: str, dataset: str, 
                           pattern: Optional[str] = None,
                           pattern_type: Optional[str] = None,
                           columns: Optional[List[str]] = None) -> Dict:
        """
        Create a feature using a simplified interface. Can create features either by:
        1. Pattern matching (prefix, suffix, or contains)
        2. Manual column selection
        
        Args:
            name: Name for the feature
            dataset: Name of the source dataset
            pattern: Optional pattern to match columns (e.g., "theta_" for prefix)
            pattern_type: Type of pattern matching ("prefix", "suffix", or "contains")
            columns: Optional list of column names to include manually
            
        Returns:
            Feature metadata
            
        Examples:
            # Create feature from columns matching a prefix
            study.create_feature_simple("eeg_theta", "eeg.csv", pattern="theta_", pattern_type="prefix")
            
            # Create feature from columns matching a suffix
            study.create_feature_simple("behavior_scores", "behavior.csv", pattern="_score", pattern_type="suffix")
            
            # Create feature from columns containing a substring
            study.create_feature_simple("attention_metrics", "behavior.csv", pattern="attention", pattern_type="contains")
            
            # Create feature from manually specified columns
            study.create_feature_simple("combined_metrics", "behavior.csv", 
                                     columns=["reaction_time", "accuracy", "confidence"])
        """
        if pattern and pattern_type:
            # Use pattern matching
            columns = self.find_columns_by_pattern(dataset, pattern, pattern_type)
        elif not columns:
            raise ValueError("Either pattern/pattern_type or columns must be provided")
            
        return self.create_feature(name, dataset, columns)
        
    def run_fusion(self, config: FusionConfig, progress_callback=None) -> str:
        """
        Run a data fusion analysis
        
        Args:
            config: Configuration for the fusion analysis
            progress_callback: Optional callback function for progress updates
            
        Returns:
            ID of the fusion run
        """
        # Generate unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = os.path.join(self.fusion_path, run_id)
        os.makedirs(run_dir)
        
        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
            
        # Load feature data
        feature_data = {}
        if config.feature_processing_configs: # If feature_processing_configs are provided, load based on them
            for processing_config in config.feature_processing_configs:
                feature_name = processing_config.feature_name
                try:
                    feature_data[feature_name] = \
                        self.data_manager.get_feature_data(processing_config.feature_name)
                except Exception as e:
                    raise ValueError(f"Error loading feature vector '{feature_name}' specified in feature_processing_configs: {e}")
        else: # No feature_processing_configs, load based on CCA settings
            print("No feature_processing_configs in FusionConfig. Loading features based on cca_settings.")
            feature_names_to_load = set() # Use a set to avoid duplicates
            if config.cca_settings.x_main_feature_vector_name:
                feature_names_to_load.add(config.cca_settings.x_main_feature_vector_name)
            if config.cca_settings.y_main_feature_vector_name:
                feature_names_to_load.add(config.cca_settings.y_main_feature_vector_name)
            if config.cca_settings.x_covariate_feature_vector_name:
                feature_names_to_load.add(config.cca_settings.x_covariate_feature_vector_name)
            if config.cca_settings.y_covariate_feature_vector_name:
                feature_names_to_load.add(config.cca_settings.y_covariate_feature_vector_name)
            
            if not feature_names_to_load:
                raise ValueError("FusionConfig has no feature_processing_configs and no feature vectors specified in cca_settings for CCA.")

            for feature_name in feature_names_to_load:
                try:
                    loaded_data = self.data_manager.get_feature_data(feature_name)
                    if loaded_data is None:
                        raise ValueError(f"Feature vector '{feature_name}' loaded as None.")
                    feature_data[feature_name] = loaded_data
                except Exception as e:
                    raise ValueError(f"Error loading feature vector '{feature_name}' specified in cca_settings: {e}")

        # Ensure all loaded data are pandas DataFrames before passing to fusion_model.run
        for name, data in feature_data.items():
            if not isinstance(data, pd.DataFrame):
                # This case should ideally not happen if load_feature_data always returns DataFrames
                # or raises an error. Adding a safeguard.
                if isinstance(data, np.ndarray):
                    print(f"Warning: Loaded data for '{name}' is a NumPy array. Converting to DataFrame.")
                    feature_data[name] = pd.DataFrame(data)
                else:
                    raise TypeError(f"Loaded data for feature vector '{name}' is not a pandas DataFrame or NumPy array. Got {type(data)}.")
                
        # Run fusion analysis
        results = self.fusion_model.run(
            feature_data,
            config,
            run_dir,
            progress_callback=progress_callback
        )
        
        # Set all other runs to non-default before marking this one as default
        self._clear_default_fusion_runs()
        
        # Save results metadata (only essential run information)
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "default": True  # Mark this run as the default
        }
        
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return run_id
        
    def _clear_default_fusion_runs(self):
        """
        Clear the default flag from all fusion runs
        """
        if not os.path.exists(self.fusion_path):
            return
        
        for run_id in os.listdir(self.fusion_path):
            run_dir = os.path.join(self.fusion_path, run_id)
            metadata_path = os.path.join(run_dir, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    metadata["default"] = False
                    
                    with open(metadata_path, 'w') as f:
                        json.dump(metadata, f, indent=2)
                except Exception as e:
                    print(f"Error clearing default flag for run {run_id}: {str(e)}")

    def set_fusion_run_as_default(self, run_id: str) -> bool:
        """
        Set a specific fusion run as the default
        
        Args:
            run_id: ID of the fusion run to set as default
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Check if the run exists
        run_dir = os.path.join(self.fusion_path, run_id)
        metadata_path = os.path.join(run_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            return False
        
        # Clear default flag from all runs
        self._clear_default_fusion_runs()
        
        # Set this run as default
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            metadata["default"] = True
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error setting run {run_id} as default: {str(e)}")
            return False

    def get_default_fusion_run(self) -> Optional[Dict]:
        """
        Get the metadata for the default fusion run
        
        Returns:
            Dict: Metadata for the default fusion run, or None if no default is set
        """
        runs = self.get_fusion_runs()
        for run in runs:
            if run.get("default", False):
                return run
        
        # If no run is explicitly marked as default, return the most recent one
        return runs[0] if runs else None
        
    def get_fusion_runs(self) -> List[Dict]:
        """
        Get a list of all fusion runs
        
        Returns:
            List of run metadata dictionaries
        """
        runs = []
        
        if not os.path.exists(self.fusion_path):
            return runs
        
        # Get list of fusion runs from directory structure
        for run_id in os.listdir(self.fusion_path):
            run_dir = os.path.join(self.fusion_path, run_id)
            metadata_path = os.path.join(run_dir, "metadata.json")
            
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                    
                    if "default" not in metadata:
                        # Handle legacy runs without default flag
                        metadata["default"] = False
                    
                    runs.append(metadata)
                except Exception as e:
                    print(f"Error loading metadata for run {run_id}: {str(e)}")
        
        # Sort runs by timestamp (newest first)
        runs = sorted(runs, key=lambda x: x.get("timestamp", ""), reverse=True)
        
        # Ensure at least one run is marked as default
        if runs and not any(run.get("default", False) for run in runs):
            # Mark the most recent run as default
            runs[0]["default"] = True
            # Update the metadata file for this run
            run_id = runs[0]["run_id"]
            run_dir = os.path.join(self.fusion_path, run_id)
            metadata_path = os.path.join(run_dir, "metadata.json")
            try:
                with open(metadata_path, 'w') as f:
                    json.dump(runs[0], f, indent=2)
            except Exception as e:
                print(f"Error updating metadata for run {run_id}: {str(e)}")
        
        return runs

    def get_fusion_run(self, run_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific fusion run
        
        Args:
            run_id: ID of the fusion run
            
        Returns:
            Run metadata if found, None otherwise
        """
        run_dir = os.path.join(self.fusion_path, run_id)
        metadata_path = os.path.join(run_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
        
    def load_fusion_results(self, run_id: str) -> Dict:
        """
        Load results from a fusion run
        
        Args:
            run_id: ID of the fusion run
            
        Returns:
            Dictionary of results
        """
        run_dir = os.path.join(self.fusion_path, run_id)
        return self.fusion_model.load_results(run_dir)
        
    def print_fusion_results(self, run_id: str, view_type: str = ""):
        """
        Print results from a fusion run
        
        Args:
            run_id: ID of the fusion run
            view_type: Type of view (e.g., "Two-view" or "Multiview")
        """
        results = self.load_fusion_results(run_id)
        self.fusion_model.print_results(results, view_type)
        
    def run_clustering(
        self,
        config: ClusteringConfig,
        fusion_run_id: Optional[str] = None,
        default_to_default_run: bool = True
    ) -> str:
        """
        Run a clustering analysis
        
        Args:
            config: Configuration for the clustering analysis
            fusion_run_id: ID of the fusion run to use (optional)
            default_to_default_run: Whether to use the default fusion run if fusion_run_id is not provided
            
        Returns:
            ID of the clustering run
        """
        # If no fusion_run_id is provided, use the default run
        if fusion_run_id is None and default_to_default_run:
            default_run = self.get_default_fusion_run()
            if default_run:
                fusion_run_id = default_run.get("run_id")
        
        if not fusion_run_id:
            raise ValueError("fusion_run_id must be provided or a default fusion run must exist")
        
        # Generate unique run ID (include config name to avoid collision when running multiple configs)
        base_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{base_id}_{config.name}" if hasattr(config, 'name') and config.name else base_id

        # Create run directory
        run_dir = os.path.join(self.clustering_path, run_id)
        os.makedirs(run_dir, exist_ok=True)
        
        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.to_dict(), f, indent=2)
        
        # Load fusion results
        fusion_run_dir = os.path.join(self.fusion_path, fusion_run_id)
        
        # Load the clustering_input.npz file which contains data specifically prepared for clustering
        clustering_input_path = os.path.join(fusion_run_dir, "clustering_input.npz")
        if not os.path.exists(clustering_input_path):
            raise FileNotFoundError(f"Clustering input not found: {clustering_input_path}")
            
        clustering_input = dict(np.load(clustering_input_path, allow_pickle=True))
        
        # Run clustering analysis - pass the parent directory and run_id separately
        # to avoid creating nested directories
        results = self.clustering_model.run(
            clustering_input,
            config,
            self.clustering_path,  # Pass the parent directory
            run_id=run_id  # Pass the run_id separately
        )
        
        # Save results metadata
        metadata = {
            "run_id": run_id,
            "fusion_run_id": fusion_run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return run_id
    
    def get_clustering_runs(self) -> List[Dict]:
        """
        Get list of all clustering runs in the study.
        
        Returns:
            List of clustering run metadata dictionaries
        """
        runs = []
        for run_id in os.listdir(self.clustering_path):
            run_dir = os.path.join(self.clustering_path, run_id)
            if os.path.isdir(run_dir):
                metadata_path = os.path.join(run_dir, "metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        metadata = json.load(f)
                        runs.append(metadata)
        return sorted(runs, key=lambda x: x["timestamp"], reverse=True)
    
    def get_clustering_run(self, run_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific clustering run.
        
        Args:
            run_id: ID of the clustering run
            
        Returns:
            Dictionary containing run metadata, or None if not found
        """
        run_dir = os.path.join(self.clustering_path, run_id)
        metadata_path = os.path.join(run_dir, "metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                return json.load(f)
        return None
    
    def load_clustering_results(self, run_id: str) -> Dict:
        """
        Load clustering results from a specific run.
        
        Args:
            run_id: ID of the clustering run
            
        Returns:
            Dictionary containing clustering results
        """
        return self.clustering_model.load_results(self.clustering_path, run_id)
    
    def print_clustering_results(self, run_id: str, title: str = "Clustering Results") -> None:
        """
        Print clustering results from a specific run.
        
        Args:
            run_id: ID of the clustering run
            title: Optional title for the results output
        """
        results = self.load_clustering_results(run_id)
        self.clustering_model.print_results(results)
        
    def evaluate_results(self, fusion_id: str, 
                        clustering_ids: List[str]) -> Dict:
        """
        Evaluate results from fusion and clustering
        
        Args:
            fusion_id: ID of the fusion run
            clustering_ids: List of clustering run IDs
            
        Returns:
            Evaluation metrics and results
        """
        return self.pipeline.evaluate_results(fusion_id, clustering_ids)
        
    @property
    def clustering_runs(self) -> List[Dict]:
        """Get all clustering analysis runs"""
        return self.pipeline.get_clustering_runs()

    # =======================================================
    # Evaluation Methods
    # =======================================================
    def run_evaluation(
        self,
        fusion_run_id: str,
        clustering_configs,
        config=None,
        progress_callback=None
    ) -> str:
        """Run evaluation on a fusion run. Returns evaluation run_id."""
        from ..evaluation.evaluator import BiotypeEvaluator
        from ..evaluation.config import EvaluationConfig

        config = config or EvaluationConfig()
        if progress_callback:
            config.progress_callback = progress_callback

        evaluator = BiotypeEvaluator(self)
        results = evaluator.validate(
            fusion_run_id=fusion_run_id,
            clustering_configs=clustering_configs,
            config=config
        )

        # Save results
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.evaluation_path, run_id)
        os.makedirs(run_dir, exist_ok=True)

        results.save(os.path.join(run_dir, "results.json"))

        eval_config = {
            'fusion_run_id': fusion_run_id,
            'clustering_configs': (
                clustering_configs if isinstance(clustering_configs[0], str)
                else [c if isinstance(c, dict) else str(c) for c in clustering_configs]
            ) if clustering_configs else [],
            'evaluation_config': config.to_dict()
        }
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(eval_config, f, indent=2)

        return run_id

    def get_evaluation_runs(self) -> List[Dict]:
        """Get all evaluation runs."""
        runs = []
        if not os.path.exists(self.evaluation_path):
            return runs
        for entry in sorted(os.listdir(self.evaluation_path), reverse=True):
            run_dir = os.path.join(self.evaluation_path, entry)
            if os.path.isdir(run_dir):
                run_info = {'run_id': entry, 'timestamp': entry}
                config_path = os.path.join(run_dir, "config.json")
                if os.path.exists(config_path):
                    try:
                        with open(config_path, 'r') as f:
                            run_info['config'] = json.load(f)
                    except Exception:
                        pass
                runs.append(run_info)
        return runs

    def load_evaluation_results(self, run_id: str):
        """Load evaluation results by run_id."""
        from ..evaluation.results import EvaluationResults
        results_path = os.path.join(self.evaluation_path, run_id, "results.json")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Evaluation results not found: {results_path}")
        return EvaluationResults.load(results_path)

    # =======================================================
    # Preprocessing Methods
    # =======================================================
    def apply_scaling_to_dataset(
        self,
        dataset_name: str,
        column_operations: Dict[str, str],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Applies scaling (standardization or normalization) to columns of a dataset
        within the processed_data directory and saves the result.

        Args:
            dataset_name: The name of the dataset file in processed_data to modify.
            column_operations: Dictionary mapping column names to operations
                               ('standardize', 'normalize', or 'none').
            output_filename: If provided, save the result to a new file with this name
                             in processed_data. If None (default), overwrite the
                             original dataset_name file in processed_data.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If the input dataset is not found in processed_data.
            ValueError: If invalid operations or columns are specified.
            TypeError: If trying to scale non-numeric data.
            IOError: If loading or saving the dataset fails.
        """
        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded from processed_data.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}' for scaling: {e}") from e

        # Apply the scaling using the library function
        try:
            scaled_df = apply_scaling(df, column_operations)
        except (ValueError, TypeError, RuntimeError) as e:
             # Re-raise exceptions from apply_scaling
             raise e

        # Determine output filename
        save_filename = output_filename if output_filename else dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        if not save_filename.lower().endswith('.csv'):
             # Add .csv if missing, or raise error if original wasn't csv?
             # For now, default to adding .csv if no extension
             if '.' not in save_filename:
                  save_filename += '.csv'
             else:
                  # Or raise error if existing extension is not supported
                  raise ValueError(f"Output filename '{save_filename}' must have a supported extension (e.g., .csv)")


        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(scaled_df, save_filename)
            # Add history entry (optional)
            # self.data_manager.add_cleaning_history_entry(save_filename, f"Applied scaling: {column_operations}")
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save scaled dataset to '{save_filename}': {e}") from e

    def apply_imputation_to_dataset(
        self,
        dataset_name: str,
        column_configs: Dict[str, Dict[str, Union[str, float, int, bool]]],
        output_filename: Optional[str] = None
    ) -> str:
        """
        Applies imputation to columns of a dataset based on per-column configurations
        within the processed_data directory and saves the result.

        Args:
            dataset_name: The name of the dataset file in processed_data.
            column_configs: Dictionary specifying imputation for each column.
                            Example: {
                                'col1': {'strategy': 'mean'},
                                'col2': {'strategy': 'constant', 'value': 0}
                            }
            output_filename: If provided, save the result to a new file. If None,
                             overwrite the original dataset_name file.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If the input dataset is not found in processed_data.
            ValueError: If the configuration is invalid.
            TypeError: If imputation is attempted on incompatible types.
            IOError: If loading or saving the dataset fails.
        """
        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded from processed_data.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}' for imputation: {e}") from e

        # Apply imputation using the library function with the new config
        try:
            imputed_df = apply_imputation(df, column_configs=column_configs)
        except (ValueError, TypeError, RuntimeError) as e:
            # Re-raise exceptions from apply_imputation
            raise e

        # Determine output filename
        save_filename = output_filename if output_filename else dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        if not save_filename.lower().endswith('.csv'):
            if '.' not in save_filename:
                 save_filename += '.csv'
            else:
                 raise ValueError(f"Output filename '{save_filename}' must have a supported extension (e.g., .csv)")

        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(imputed_df, save_filename)
            # Optionally add history entry
            # history_desc = f"Applied imputation with config: {column_configs}"
            # self.data_manager.add_cleaning_history_entry(save_filename, history_desc)
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save imputed dataset to '{save_filename}': {e}") from e

    def apply_select_columns_to_dataset(
        self,
        dataset_name: str,
        keep_columns: Optional[List[str]] = None,
        remove_columns: Optional[List[str]] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Selects or removes columns from a dataset within the processed_data
        directory and saves the result.

        Args:
            dataset_name: The name of the dataset file in processed_data.
            keep_columns: Optional list of column names to keep.
            remove_columns: Optional list of column names to remove.
            output_filename: If provided, save the result to a new file. If None,
                             overwrite the original dataset_name file.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If the input dataset is not found in processed_data.
            ValueError: If invalid arguments are provided (e.g., both keep/remove).
            IOError: If loading or saving the dataset fails.
        """
        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded from processed_data.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}' for column selection: {e}") from e

        # Apply column selection using the library function
        try:
            selected_df = select_columns(df, keep_columns=keep_columns, remove_columns=remove_columns)
        except (ValueError, TypeError) as e:
            # Re-raise exceptions from select_columns
            raise e

        # Determine output filename
        save_filename = output_filename if output_filename else dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        # Use the original extension if overwriting, default to .csv for new files
        if output_filename:
             _, ext = os.path.splitext(save_filename)
             if not ext:
                  save_filename += '.csv' # Default to csv if new file has no extension
             elif ext.lower() != '.csv': # Currently only support saving CSV
                  raise ValueError(f"Output filename '{save_filename}' must have a .csv extension.")
        else:
             # Overwriting: ensure we use the original extension if possible
             _, original_ext = os.path.splitext(dataset_name)
             if original_ext.lower() != '.csv':
                  # This case shouldn't happen if load_dataset worked, but handle defensively
                  raise ValueError(f"Cannot overwrite non-CSV file '{dataset_name}' with selected columns.")

        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(selected_df, save_filename)
            # Optionally add history entry
            # action_desc = f"Kept columns: {keep_columns}" if keep_columns else f"Removed columns: {remove_columns}"
            # self.data_manager.add_cleaning_history_entry(save_filename, action_desc)
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save dataset with selected columns to '{save_filename}': {e}") from e

    def apply_formula_column_to_dataset(
        self,
        dataset_name: str,
        new_column_name: str,
        formula: str,
        pre_cast_to_str: Optional[List[str]] = None,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Adds a new column to a dataset based on a formula evaluated using pandas.eval.
        Operates on and saves to the processed_data directory.

        Args:
            dataset_name: The name of the dataset file in processed_data.
            new_column_name: The name for the new column to create.
            formula: The formula string to evaluate (e.g., 'col_a + col_b / 2').
            pre_cast_to_str: An optional list of column names that should be cast
                             to string type before evaluating the formula.
            output_filename: If provided, save the result to a new file. If None,
                             overwrite the original dataset_name file.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If the input dataset is not found in processed_data.
            ValueError: If the formula is invalid, uses non-existent columns, or the
                        new column name already exists.
            IOError: If loading or saving the dataset fails.
        """
        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded from processed_data.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}' for formula column creation: {e}") from e

        # Create the formula column using the library function
        try:
            modified_df = create_formula_column(df, new_column_name, formula, pre_cast_to_str=pre_cast_to_str)
        except (ValueError, TypeError, RuntimeError) as e:
            # Re-raise exceptions from create_formula_column
            raise e

        # Determine output filename
        save_filename = output_filename if output_filename else dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        if output_filename:
             _, ext = os.path.splitext(save_filename)
             if not ext:
                  save_filename += '.csv'
             elif ext.lower() != '.csv':
                  raise ValueError(f"Output filename '{save_filename}' must have a .csv extension.")
        else:
             _, original_ext = os.path.splitext(dataset_name)
             if original_ext.lower() != '.csv':
                  raise ValueError(f"Cannot overwrite non-CSV file '{dataset_name}' with formula column.")

        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(modified_df, save_filename)
            # Optionally add history entry
            # self.data_manager.add_cleaning_history_entry(save_filename, f"Created formula column '{new_column_name}' with formula: {formula}")
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save dataset with formula column to '{save_filename}': {e}") from e

    def apply_filter_rows_to_dataset(
        self,
        dataset_name: str,
        query_string: str,
        output_filename: Optional[str] = None
    ) -> str:
        """
        Filters rows in a dataset based on a query string and saves the result.
        Operates on and saves to the processed_data directory.

        Args:
            dataset_name: The name of the dataset file in processed_data.
            query_string: The query string for filtering rows (pandas.query syntax).
            output_filename: If provided, save the result to a new file. If None,
                             overwrite the original dataset_name file.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If the input dataset is not found in processed_data.
            ValueError: If the query string is invalid.
            IOError: If loading or saving the dataset fails.
        """
        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded from processed_data.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}' for filtering: {e}") from e

        # Apply row filtering using the library function
        try:
            filtered_df = filter_rows_by_condition(df, query_string)
        except (ValueError, TypeError) as e:
            # Re-raise exceptions from filter_rows_by_condition
            raise e

        # Determine output filename
        save_filename = output_filename if output_filename else dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        if output_filename:
             _, ext = os.path.splitext(save_filename)
             if not ext:
                  save_filename += '.csv'
             elif ext.lower() != '.csv':
                  raise ValueError(f"Output filename '{save_filename}' must have a .csv extension.")
        else:
             _, original_ext = os.path.splitext(dataset_name)
             if original_ext.lower() != '.csv':
                  raise ValueError(f"Cannot overwrite non-CSV file '{dataset_name}' with filtered rows.")

        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(filtered_df, save_filename)
            # Optionally add history entry
            # self.data_manager.add_cleaning_history_entry(save_filename, f"Filtered rows with query: {query_string}")
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save filtered dataset to '{save_filename}': {e}") from e

    def apply_join_datasets_to_dataset(
        self,
        anchor_dataset_name: str,
        anchor_on: Union[str, List[str], None] = None,
        adjoining_dataset_name: str = None,
        adjoining_on: Union[str, List[str], None] = None,
        adjoining_columns_to_keep: Optional[List[str]] = None,
        how: str = 'left',
        output_filename: Optional[str] = None
    ) -> str:
        """
        Joins two datasets within the processed_data directory and saves the result.

        Args:
            anchor_dataset_name: Filename of the primary dataset.
            anchor_on: Column(s) in the anchor dataset to join on.
                       Not required if how='concat'.
            adjoining_dataset_name: Filename of the dataset to join.
            adjoining_on: Column(s) in the adjoining dataset to join on.
                          Not required if how='concat'.
            adjoining_columns_to_keep: List of columns from the adjoining dataset
                                         to include (must include join columns for standard joins).
                                         If None and how='concat', all columns are kept.
            how: Type of join ('left', 'right', 'inner', 'outer', 'concat'). Defaults to 'left'.
                 - 'concat': Horizontal concatenation by row index (no join keys needed).
            output_filename: If provided, save result to a new file. If None,
                             overwrite the anchor_dataset_name file.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError: If input datasets are not found.
            ValueError: If join parameters are invalid.
            IOError: If loading or saving fails.
        """
        # Load anchor dataset
        try:
            anchor_df = self.data_manager.load_dataset(anchor_dataset_name)
            if anchor_df is None:
                raise FileNotFoundError(f"Anchor dataset '{anchor_dataset_name}' could not be loaded.")
        except Exception as e:
            raise IOError(f"Failed to load anchor dataset '{anchor_dataset_name}': {e}") from e

        # Load adjoining dataset
        try:
            adjoining_df = self.data_manager.load_dataset(adjoining_dataset_name)
            if adjoining_df is None:
                raise FileNotFoundError(f"Adjoining dataset '{adjoining_dataset_name}' could not be loaded.")
        except Exception as e:
            raise IOError(f"Failed to load adjoining dataset '{adjoining_dataset_name}': {e}") from e

        # Perform the join using the library function
        try:
            merged_df = join_datasets(
                anchor_df=anchor_df,
                adjoining_df=adjoining_df,
                anchor_on=anchor_on,
                adjoining_on=adjoining_on,
                adjoining_columns_to_keep=adjoining_columns_to_keep,
                how=how
            )
        except (ValueError, TypeError, RuntimeError) as e:
            # Re-raise exceptions from join_datasets
            raise e

        # Determine output filename
        save_filename = output_filename if output_filename else anchor_dataset_name

        # Ensure output filename has a valid extension (e.g., .csv)
        if output_filename:
             _, ext = os.path.splitext(save_filename)
             if not ext:
                  save_filename += '.csv'
             elif ext.lower() != '.csv':
                  raise ValueError(f"Output filename '{save_filename}' must have a .csv extension.")
        else:
             _, original_ext = os.path.splitext(anchor_dataset_name)
             if original_ext.lower() != '.csv':
                  raise ValueError(f"Cannot overwrite non-CSV file '{anchor_dataset_name}' with joined data.")

        # Save the result back to processed_data
        try:
            self.data_manager.save_processed_dataset(merged_df, save_filename)
            # Optionally add history entry
            # history_cols = ", ".join(adjoining_columns_to_keep)
            # self.data_manager.add_cleaning_history_entry(save_filename, f"Joined with {adjoining_dataset_name} on {anchor_on}/{adjoining_on}, kept: {history_cols}")
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save joined dataset to '{save_filename}': {e}") from e

    def apply_grouped_difference_to_dataset(
        self,
        dataset_name: str,
        group_by_col: str,
        value_col_pattern: str,
        value_col_pattern_type: str,
        condition_col: str,
        condition_val_a: any,
        condition_val_b: any,
        new_column_name: str,
        output_filename: str,
        columns_to_keep_from_condition_b: Optional[List[str]] = None
    ) -> str:
        """
        Calculates grouped differences for multiple columns based on pattern matching,
        keeps specified additional columns, filters for complete groups, and saves the result.

        Args:
            dataset_name: Name of the dataset file in processed_data.
            group_by_col: Column to group by (e.g., 'SUBJLABEL').
            value_col_pattern: Pattern to identify value columns (e.g., "QIDS_SR_").
            value_col_pattern_type: Type of pattern ("prefix", "suffix", "contains").
            condition_col: Column defining the conditions (e.g., 'EVENTNAME').
            condition_val_a: Value representing the minuend (e.g., 'Wk8').
            condition_val_b: Value representing the subtrahend (e.g., 'Baseline').
            new_column_name: Base name for the resulting difference columns. The actual
                             difference columns will be named like '{value_col_identified}_diff'.
                             This parameter might be re-evaluated for its utility.
            output_filename: Filename to save the result to (must be provided).
            columns_to_keep_from_condition_b: Optional list of columns from condition_val_b
                                              data to include in the output.

        Returns:
            The filename of the saved processed dataset.

        Raises:
            FileNotFoundError, ValueError, TypeError, IOError, RuntimeError as appropriate.
        """
        # Validate output_filename is provided (as output_type is removed)
        if not output_filename:
            raise ValueError("output_filename must be provided for saving the grouped difference result.")

        # Load the dataset from processed_data
        try:
            df = self.data_manager.load_dataset(dataset_name)
            if df is None:
                raise FileNotFoundError(f"Dataset '{dataset_name}' could not be loaded.")
        except Exception as e:
            raise IOError(f"Failed to load dataset '{dataset_name}': {e}") from e

        # Calculate the grouped difference using the updated preprocess function
        try:
            result_df = calculate_grouped_difference(
                df=df,
                group_by_col=group_by_col,
                value_col_pattern=value_col_pattern,
                value_col_pattern_type=value_col_pattern_type,
                condition_col=condition_col,
                condition_val_a=condition_val_a,
                condition_val_b=condition_val_b,
                new_column_name=new_column_name,
                columns_to_keep_from_condition_b=columns_to_keep_from_condition_b
            )
        except (ValueError, TypeError, RuntimeError) as e:
            raise e # Re-raise errors from calculation

        # output_df is now directly result_df, as output_type logic is removed
        output_df = result_df

        # Determine output filename and check extension (already handled by output_filename being mandatory)
        save_filename = output_filename # Already validated to be present
        if not save_filename.lower().endswith('.csv'):
            if '.' not in save_filename:
                 save_filename += '.csv'
            else:
                 raise ValueError(f"Output filename '{save_filename}' must have a .csv extension.")

        # Save the result
        try:
            self.data_manager.save_processed_dataset(output_df, save_filename)
            # Optionally add history entry (consider what to log for this more complex op)
            # self.data_manager.add_cleaning_history_entry(save_filename, f"Calculated grouped diff using pattern '{value_col_pattern}'")
            return save_filename
        except Exception as e:
            raise IOError(f"Failed to save grouped difference result to '{save_filename}': {e}") from e

    def delete_processed_dataset(self, dataset_name: str) -> bool:
        """
        Deletes a specific dataset file from the processed_data directory.

        Args:
            dataset_name: The name of the dataset file to delete.

        Returns:
            True if deletion was successful, False otherwise.

        Raises:
            FileNotFoundError: If the file doesn't exist in processed_data.
            IOError: If there's a permission error or other OS issue during deletion.
        """
        try:
            deleted = self.data_manager.delete_processed_dataset(dataset_name)
            # Update study metadata timestamp?
            if deleted:
                self.update_metadata({
                    "last_modified": datetime.now().isoformat()
                })
            return deleted
        except (FileNotFoundError, IOError) as e:
            # Re-raise specific errors from DataManager
            raise e
        except Exception as e:
            # Catch any other unexpected errors
            raise RuntimeError(f"An unexpected error occurred while deleting dataset '{dataset_name}': {e}") from e

    def restore_all_raw_data(self) -> Tuple[int, int, List[str]]:
        """
        Restores all raw data to the processed_data directory.
        This effectively resets the processed_data directory to match raw_data.

        Returns:
            A tuple from DataManager.restore_raw_data_to_processed:
                (deleted_count, copied_count, errors_list)
        """
        try:
            deleted_count, copied_count, errors = self.data_manager.restore_raw_data_to_processed()
            self.update_metadata({
                "last_modified": datetime.now().isoformat(),
                "last_raw_restore": datetime.now().isoformat()
            })
            # After restoring, features defined in features.json might be invalid
            # if their source datasets (from processed_data) were based on files that
            # are no longer present after the restore, or if their columns changed.
            # The DataManager.restore_raw_data_to_processed attempts to re-export features,
            # which will generate warnings if source data is missing.
            # No specific feature cleanup is done here in Study, relying on DataManager's behavior.
            return deleted_count, copied_count, errors
        except Exception as e:
            raise RuntimeError(f"An unexpected error occurred during raw data restoration: {e}") from e 

    def plot_cca_results(self, fusion_run_id: str, max_modes_to_plot: int = 5):
        """
        Plots CCA results from a specified fusion run.

        Args:
            fusion_run_id: ID of the fusion run.
            max_modes_to_plot: Maximum number of CCA modes to plot for variates and weights.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np # Make sure numpy is imported for np.corrcoef and np.ceil
        except ImportError as e:
            print(f"Matplotlib and NumPy are required for plotting. Please install them. Error: {e}")
            return

        results = self.load_fusion_results(fusion_run_id)

        # Handle both old and new result structures
        cca_data = None
        if 'cca' in results and results['cca'] is not None:
            # Old structure
            cca_data = results['cca']
        elif 'cca_results' in results and results['cca_results'] is not None:
            # New structure
            cca_data = results['cca_results']

        if cca_data is None:
            print(f"No CCA data found in fusion run '{fusion_run_id}'.")
            return
        if isinstance(cca_data, np.ndarray) and cca_data.size == 1:
             # Handle case where cca_data might be a 0-d array containing the dict
            cca_data = cca_data.item()

        if not isinstance(cca_data, dict):
            print(f"CCA data in fusion run '{fusion_run_id}' is not in the expected dictionary format.")
            return

        variates = cca_data.get('transformed_data')
        weights = cca_data.get('weights')
        correlations = cca_data.get('correlations')
        x_feature_names = cca_data.get('x_feature_names')
        y_feature_names = cca_data.get('y_feature_names')

        if variates is None or len(variates) < 2 or weights is None or correlations is None:
            print("CCA results are incomplete (missing variates, weights, or correlations).")
            return

        variate_U = variates[0]
        variate_V = variates[1]

        # Handle both old and new weights formats
        if isinstance(weights, dict):
            # New format: weights is a dictionary with 'x_weights' and 'y_weights'
            coeff_A = weights.get('x_weights')
            coeff_B = weights.get('y_weights')
            if coeff_A is None or coeff_B is None:
                print("CCA weights dictionary missing 'x_weights' or 'y_weights' keys.")
                return
        elif isinstance(weights, (list, tuple)) and len(weights) >= 2:
            # Old format: weights is a list/tuple with [x_weights, y_weights]
            coeff_A = weights[0]
            coeff_B = weights[1]
        else:
            print("CCA weights are not in expected format (dict with x_weights/y_weights or list with 2 elements).")
            return

        if not isinstance(variate_U, np.ndarray) or not isinstance(variate_V, np.ndarray) or \
           not isinstance(coeff_A, np.ndarray) or not isinstance(coeff_B, np.ndarray):
            print("CCA variates or weights are not NumPy arrays as expected.")
            return
            
        n_modes_available = variate_U.shape[1]
        n_modes_to_plot = min(n_modes_available, max_modes_to_plot)

        if n_modes_to_plot == 0:
            print("No CCA modes available to plot.")
            return

        # Plot scatter plots of variates
        print(f"Plotting scatter of CCA variates for top {n_modes_to_plot} modes...")
        cols_scatter = int(np.ceil(np.sqrt(n_modes_to_plot)))
        rows_scatter = int(np.ceil(n_modes_to_plot / cols_scatter))
        fig_scatter, axs_scatter = plt.subplots(rows_scatter, cols_scatter, figsize=(cols_scatter*4, rows_scatter*4), squeeze=False)
        axs_scatter_flat = axs_scatter.flatten()
        for i in range(n_modes_to_plot):
            ax = axs_scatter_flat[i]
            ax.scatter(variate_U[:, i], variate_V[:, i], alpha=0.7)
            # Calculate correlation for the current mode
            # Correlations array from cca-zoo score might be a single value for 1 component
            # or an array of values for multiple components.
            current_corr = correlations[i] if isinstance(correlations, (list, np.ndarray)) and i < len(correlations) else correlations
            ax.set_title(f'Mode {i + 1}\nCorr = {current_corr:.3f}')
            ax.set_xlabel("X Variate (e.g., EEG)")
            ax.set_ylabel("Y Variate (e.g., Clinical)")
            ax.grid(True)
        # Hide unused subplots
        for i in range(n_modes_to_plot, len(axs_scatter_flat)):
            axs_scatter_flat[i].set_visible(False)
        fig_scatter.suptitle('CCA Variate Scatter Plots', fontsize=16)
        fig_scatter.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

        # Plot bar plots of Y-weights (e.g., clinical features)
        if y_feature_names is not None and len(y_feature_names) == coeff_B.shape[0]:
            print(f"Plotting Y-weights (e.g., clinical features) for top {n_modes_to_plot} modes...")
            cols_bar_y = int(np.ceil(np.sqrt(n_modes_to_plot)))
            rows_bar_y = int(np.ceil(n_modes_to_plot / cols_bar_y))
            fig_bar_y, axs_bar_y = plt.subplots(rows_bar_y, cols_bar_y, figsize=(cols_bar_y*5, rows_bar_y*4), squeeze=False)
            axs_bar_y_flat = axs_bar_y.flatten()
            for i in range(n_modes_to_plot):
                ax = axs_bar_y_flat[i]
                current_weights = coeff_B[:, i]
                current_feature_names = list(y_feature_names)

                if len(current_feature_names) > 25: # Threshold for too many features
                    # Sort by absolute weight and take top 25
                    sorted_indices = np.argsort(np.abs(current_weights))[::-1][:25]
                    current_weights = current_weights[sorted_indices]
                    current_feature_names = [current_feature_names[j] for j in sorted_indices]
                    ax.set_title(f'Mode {i + 1} Y-Weights (Top 25)')
                else:
                    ax.set_title(f'Mode {i + 1} Y-Weights')
                
                ax.bar(current_feature_names, current_weights)
                ax.set_xlabel("Features (e.g., Clinical)")
                ax.set_ylabel("Weight")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.grid(True, axis='y')
            for i in range(n_modes_to_plot, len(axs_bar_y_flat)):
                axs_bar_y_flat[i].set_visible(False)
            fig_bar_y.suptitle('CCA Y-Feature Weights', fontsize=16)
            fig_bar_y.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("Y-feature names not available or mismatch with weights, skipping Y-weights plot.")

        # Plot bar plots of X-weights (e.g., EEG features)
        if x_feature_names is not None and len(x_feature_names) == coeff_A.shape[0]:
            print(f"Plotting X-weights (e.g., EEG features) for top {n_modes_to_plot} modes...")
            cols_bar_x = int(np.ceil(np.sqrt(n_modes_to_plot)))
            rows_bar_x = int(np.ceil(n_modes_to_plot / cols_bar_x))
            fig_bar_x, axs_bar_x = plt.subplots(rows_bar_x, cols_bar_x, figsize=(cols_bar_x*5, rows_bar_x*4), squeeze=False)
            axs_bar_x_flat = axs_bar_x.flatten()
            for i in range(n_modes_to_plot):
                ax = axs_bar_x_flat[i]
                current_weights = coeff_A[:, i]
                current_feature_names = list(x_feature_names)

                if len(current_feature_names) > 25: # Threshold for too many features
                    # Sort by absolute weight and take top 25
                    sorted_indices = np.argsort(np.abs(current_weights))[::-1][:25]
                    current_weights = current_weights[sorted_indices]
                    current_feature_names = [current_feature_names[j] for j in sorted_indices]
                    ax.set_title(f'Mode {i + 1} X-Weights (Top 25)')
                else:
                    ax.set_title(f'Mode {i + 1} X-Weights')
                
                ax.bar(current_feature_names, current_weights)
                ax.set_xlabel("Features (e.g., EEG)")
                ax.set_ylabel("Weight")
                plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
                ax.grid(True, axis='y')
            for i in range(n_modes_to_plot, len(axs_bar_x_flat)):
                axs_bar_x_flat[i].set_visible(False)
            fig_bar_x.suptitle('CCA X-Feature Weights', fontsize=16)
            fig_bar_x.tight_layout(rect=[0, 0, 1, 0.96])
            plt.show()
        else:
            print("X-feature names not available or mismatch with weights, skipping X-weights plot.") 