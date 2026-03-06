"""
Pipeline management for IMBiotype analysis workflow
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional
import uuid
import numpy as np

from ..models.fusion import FusionModel, FusionConfig
from ..models.clustering import ClusteringModel, ClusteringConfig

class Pipeline:
    """
    Manages the analysis pipeline including fusion and clustering
    """
    
    def __init__(self, study):
        """
        Initialize the pipeline manager
        
        Args:
            study: Reference to the parent Study object
        """
        self.study = study
        self.fusion_model = FusionModel()
        self.clustering_model = ClusteringModel()
        
    def run_fusion(self, config: FusionConfig) -> str:
        """
        Run a data fusion analysis
        
        Args:
            config: Configuration for the fusion analysis
            
        Returns:
            ID of the fusion run
        """
        # Generate unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = os.path.join(self.study.fusion_path, run_id)
        os.makedirs(run_dir)
        
        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config.__dict__, f, indent=2)
            
        # Load feature data
        feature_data = {}
        for fv_config in config.dim_reduction_settings:
            feature_data[fv_config.name] = \
                self.study.data_manager.load_feature_data(fv_config.name)
                
        # Run fusion analysis
        results = self.fusion_model.run(
            feature_data,
            config,
            run_dir
        )
        
        # Save results metadata (only essential run information)
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat()
        }
        
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return run_id
        
    def run_clustering(self, configs: List[ClusteringConfig]) -> str:
        """
        Run clustering analysis with multiple configurations
        
        Args:
            configs: List of clustering configurations
            
        Returns:
            ID of the clustering run
        """
        if not configs:
            raise ValueError("No clustering configurations provided")
            
        # Generate unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = os.path.join(self.study.clustering_path, run_id)
        os.makedirs(run_dir)
        
        # Save configuration
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump({
                "configs": [config.__dict__ for config in configs]
            }, f, indent=2)
            
        # Load fusion results if specified
        fusion_id = configs[0].fusion_run_id if configs else None
        if fusion_id:
            fusion_dir = os.path.join(self.study.fusion_path, fusion_id)
            clustering_input = np.load(os.path.join(fusion_dir, "clustering_input.npz"))
        else:
            raise ValueError("Fusion run ID must be specified in clustering config")
            
        # Run clustering analysis
        clustering_run_id = self.clustering_model.run(
            clustering_input,
            configs,
            run_dir
        )
        
        # Save results metadata (only essential run information)
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "fusion_run_id": fusion_id
        }
        
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return run_id
        
    def evaluate_results(self, fusion_id: str, 
                        clustering_ids: List[str]) -> str:
        """
        Evaluate results from fusion and clustering runs
        
        Args:
            fusion_id: ID of the fusion run
            clustering_ids: List of clustering run IDs
            
        Returns:
            ID of the evaluation run
        """
        # Generate unique run ID
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create run directory
        run_dir = os.path.join(self.study.evaluation_path, run_id)
        os.makedirs(run_dir)
        
        # Load fusion results
        fusion_dir = os.path.join(self.study.fusion_path, fusion_id)
        fusion_results = np.load(os.path.join(fusion_dir, "results.npz"))
        
        # Load clustering results
        clustering_results = []
        for c_id in clustering_ids:
            c_dir = os.path.join(self.study.clustering_path, c_id)
            results = self.clustering_model.load_results(c_dir)
            clustering_results.append(results)
            
        # Compute evaluation metrics
        metrics = {
            "fusion_metrics": fusion_results.get("metrics", {}),
            "clustering_metrics": [
                r.get("metrics", {}) for r in clustering_results
            ]
        }
        
        # Save results
        results_path = os.path.join(run_dir, "results.npz")
        np.savez(results_path, **metrics)
        
        # Save configuration
        config = {
            "fusion_run_id": fusion_id,
            "clustering_run_ids": clustering_ids
        }
        config_path = os.path.join(run_dir, "config.json")
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
            
        # Save metadata
        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "fusion_run_id": fusion_id,
            "clustering_run_ids": clustering_ids
        }
        metadata_path = os.path.join(run_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        return run_id
        
    def get_fusion_runs(self) -> List[Dict]:
        """Get metadata for all fusion runs"""
        runs = []
        for run_id in os.listdir(self.study.fusion_path):
            metadata_path = os.path.join(
                self.study.fusion_path,
                run_id,
                "metadata.json"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    runs.append(json.load(f))
        return runs
        
    def get_clustering_runs(self) -> List[Dict]:
        """Get metadata for all clustering runs"""
        runs = []
        for run_id in os.listdir(self.study.clustering_path):
            metadata_path = os.path.join(
                self.study.clustering_path,
                run_id,
                "metadata.json"
            )
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    runs.append(json.load(f))
        return runs 