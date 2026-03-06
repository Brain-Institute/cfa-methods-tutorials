"""
Evaluation model for assessing fusion and clustering combinations.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

class EvaluationModel:
    """
    A model for evaluating combinations of fusion and clustering results.
    """
    
    def __init__(self):
        self.fusion_config: str = ""
        self.cluster_configs: List[str] = []
        self.evaluation_results: Dict[str, Any] = {}
        
    def update_configs(self, fusion_config: str, cluster_configs: List[str]) -> None:
        """
        Update the selected fusion and clustering configurations.
        
        Args:
            fusion_config: Name of the selected fusion configuration
            cluster_configs: List of selected clustering configuration names
        """
        self.fusion_config = fusion_config.strip() if fusion_config else ""
        self.cluster_configs = cluster_configs if cluster_configs else []
        self.evaluation_results = {}
        
    def is_combo_valid(self) -> bool:
        """
        Check if the current combination of fusion and clustering configs is valid.
        
        Returns:
            bool: True if both fusion and at least one clustering config are selected
        """
        return bool(self.fusion_config) and len(self.cluster_configs) > 0
    
    def evaluate_clustering(self, data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering results using multiple metrics.
        
        Args:
            data: The feature matrix (n_samples, n_features)
            labels: Cluster assignments (n_samples,)
            
        Returns:
            Dict containing evaluation metrics:
                - silhouette_score: Higher is better
                - calinski_harabasz_score: Higher is better
                - davies_bouldin_score: Lower is better
        """
        # Standardize the data for fair comparison
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data)
        
        # Calculate metrics
        metrics = {
            'silhouette_score': silhouette_score(data_scaled, labels),
            'calinski_harabasz_score': calinski_harabasz_score(data_scaled, labels),
            'davies_bouldin_score': davies_bouldin_score(data_scaled, labels)
        }
        
        return metrics
    
    def evaluate_fusion_clustering_combo(self, 
                                      fusion_data: np.ndarray,
                                      fusion_labels: np.ndarray,
                                      clustering_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate a combination of fusion and clustering results.
        
        Args:
            fusion_data: The fused feature matrix
            fusion_labels: Labels from fusion process
            clustering_labels: Labels from clustering process
            
        Returns:
            Dict containing:
                - fusion_metrics: Metrics for fusion results
                - clustering_metrics: Metrics for clustering results
                - agreement_score: Agreement between fusion and clustering labels
        """
        # Evaluate fusion results
        fusion_metrics = self.evaluate_clustering(fusion_data, fusion_labels)
        
        # Evaluate clustering results
        clustering_metrics = self.evaluate_clustering(fusion_data, clustering_labels)
        
        # Calculate agreement between fusion and clustering labels
        # Using adjusted mutual information for label agreement
        from sklearn.metrics import adjusted_mutual_info_score
        agreement_score = adjusted_mutual_info_score(fusion_labels, clustering_labels)
        
        return {
            'fusion_metrics': fusion_metrics,
            'clustering_metrics': clustering_metrics,
            'agreement_score': agreement_score
        }
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the evaluation results.
        
        Returns:
            Dict containing evaluation summary with interpretation
        """
        if not self.evaluation_results:
            return {}
            
        summary = {
            'fusion_config': self.fusion_config,
            'cluster_configs': self.cluster_configs,
            'metrics': self.evaluation_results,
            'interpretation': {
                'fusion_quality': self._interpret_metrics(self.evaluation_results['fusion_metrics']),
                'clustering_quality': self._interpret_metrics(self.evaluation_results['clustering_metrics']),
                'agreement': self._interpret_agreement(self.evaluation_results['agreement_score'])
            }
        }
        
        return summary
    
    def _interpret_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Interpret clustering metrics and provide a quality assessment.
        
        Args:
            metrics: Dictionary of clustering metrics
            
        Returns:
            String interpretation of the metrics
        """
        silhouette = metrics['silhouette_score']
        calinski = metrics['calinski_harabasz_score']
        davies = metrics['davies_bouldin_score']
        
        # Interpret silhouette score
        if silhouette > 0.7:
            sil_quality = "Excellent"
        elif silhouette > 0.5:
            sil_quality = "Good"
        elif silhouette > 0.3:
            sil_quality = "Fair"
        else:
            sil_quality = "Poor"
            
        # Interpret Davies-Bouldin score
        if davies < 0.3:
            db_quality = "Excellent"
        elif davies < 0.5:
            db_quality = "Good"
        elif davies < 0.7:
            db_quality = "Fair"
        else:
            db_quality = "Poor"
            
        return f"Overall quality: {sil_quality} (Silhouette: {silhouette:.3f}, Davies-Bouldin: {db_quality})"
    
    def _interpret_agreement(self, agreement_score: float) -> str:
        """
        Interpret the agreement score between fusion and clustering labels.
        
        Args:
            agreement_score: Agreement score between labels
            
        Returns:
            String interpretation of the agreement
        """
        if agreement_score > 0.8:
            return "Strong agreement between fusion and clustering"
        elif agreement_score > 0.5:
            return "Moderate agreement between fusion and clustering"
        elif agreement_score > 0.2:
            return "Weak agreement between fusion and clustering"
        else:
            return "Little to no agreement between fusion and clustering" 