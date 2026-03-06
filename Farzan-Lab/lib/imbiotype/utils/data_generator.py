"""
Data generation utilities for IMBiotype
"""

import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime

class DataGenerator:
    """
    Generates synthetic data for testing and examples with ground truth clusters
    that are recoverable through fusion.
    """
    
    def __init__(self, n_samples: int = 100, n_clusters: int = 3, random_seed: int = 42):
        """
        Initialize the data generator
        
        Args:
            n_samples: Number of samples to generate
            n_clusters: Number of ground truth clusters to generate
            random_seed: Random seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_clusters = n_clusters
        np.random.seed(random_seed)
        
        # Generate cluster assignments
        self.cluster_assignments = np.random.randint(0, n_clusters, size=n_samples)
        
        # Generate cluster-specific parameters
        self.cluster_params = {
            'eeg': {
                'power_scale': np.random.uniform(0.8, 1.2, size=n_clusters),
                'phase_shift': np.random.uniform(-np.pi/4, np.pi/4, size=n_clusters),
                'frequency_shift': np.random.uniform(0.9, 1.1, size=n_clusters)
            },
            'behavior': {
                'reaction_time_scale': np.random.uniform(0.8, 1.2, size=n_clusters),
                'accuracy_bias': np.random.uniform(0.7, 0.9, size=n_clusters),
                'confidence_scale': np.random.uniform(0.8, 1.2, size=n_clusters)
            },
            'clinical': {
                'depression_scale': np.random.uniform(0.8, 1.2, size=n_clusters),
                'anxiety_scale': np.random.uniform(0.8, 1.2, size=n_clusters),
                'cognitive_scale': np.random.uniform(0.8, 1.2, size=n_clusters)
            }
        }
        
    def generate_from_columns(self, columns: List[str], 
                            output_dir: str,
                            filename: str) -> str:
        """
        Generate synthetic data based on column names with cluster-specific patterns
        
        Args:
            columns: List of column names to generate
            output_dir: Directory to save the data
            filename: Name for the output file
            
        Returns:
            Path to the generated file
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate data based on column name patterns
        data = {}
        for col in columns:
            # Get cluster-specific parameters
            cluster_idx = self.cluster_assignments
            
            if any(x in col.lower() for x in ['power', 'amplitude', 'magnitude']):
                # EEG-like power values with cluster-specific scaling
                base_power = np.random.gamma(shape=2, scale=1, size=self.n_samples)
                cluster_scales = self.cluster_params['eeg']['power_scale'][cluster_idx]
                data[col] = base_power * cluster_scales
                
            elif any(x in col.lower() for x in ['phase', 'angle']):
                # Phase/angle values with cluster-specific shifts
                base_phase = np.random.uniform(-np.pi, np.pi, size=self.n_samples)
                cluster_shifts = self.cluster_params['eeg']['phase_shift'][cluster_idx]
                data[col] = (base_phase + cluster_shifts) % (2 * np.pi)
                
            elif any(x in col.lower() for x in ['time', 'latency', 'duration']):
                # Time-like values with cluster-specific scaling
                base_time = np.random.normal(loc=200, scale=50, size=self.n_samples)
                cluster_scales = self.cluster_params['behavior']['reaction_time_scale'][cluster_idx]
                data[col] = np.maximum(0, base_time * cluster_scales)
                
            elif any(x in col.lower() for x in ['accuracy', 'correct', 'hit']):
                # Binary values with cluster-specific bias
                cluster_bias = self.cluster_params['behavior']['accuracy_bias'][cluster_idx]
                data[col] = np.random.binomial(n=1, p=cluster_bias, size=self.n_samples)
                
            elif any(x in col.lower() for x in ['confidence', 'certainty']):
                # Confidence values with cluster-specific scaling
                base_conf = np.random.beta(a=2, b=2, size=self.n_samples)
                cluster_scales = self.cluster_params['behavior']['confidence_scale'][cluster_idx]
                data[col] = np.clip(base_conf * cluster_scales, 0, 1)
                
            elif any(x in col.lower() for x in ['depression', 'anxiety', 'stress']):
                # Clinical scales with cluster-specific scaling
                base_scale = np.random.normal(loc=50, scale=15, size=self.n_samples)
                if 'depression' in col.lower():
                    cluster_scales = self.cluster_params['clinical']['depression_scale'][cluster_idx]
                elif 'anxiety' in col.lower():
                    cluster_scales = self.cluster_params['clinical']['anxiety_scale'][cluster_idx]
                else:
                    cluster_scales = np.random.uniform(0.8, 1.2, size=self.n_samples)
                data[col] = np.clip(base_scale * cluster_scales, 0, 100)
                
            elif any(x in col.lower() for x in ['cognitive', 'function']):
                # Cognitive function with cluster-specific scaling
                base_cog = np.random.normal(loc=50, scale=15, size=self.n_samples)
                cluster_scales = self.cluster_params['clinical']['cognitive_scale'][cluster_idx]
                data[col] = np.clip(base_cog * cluster_scales, 0, 100)
                
            else:
                # Default: standard normal distribution
                data[col] = np.random.normal(size=self.n_samples)
                
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Add some missing values randomly (5% of data)
        mask = np.random.random(size=df.shape) < 0.05
        df[mask] = np.nan
        
        # Save to CSV
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        
        return output_path
        
    def generate_test_study(self, output_dir: str) -> Tuple[str, str, str]:
        """
        Generate a complete test study with EEG, behavior, and clinical data
        containing ground truth clusters that are recoverable through fusion
        
        Args:
            output_dir: Directory to save the data
            
        Returns:
            Tuple of (eeg_file, behavior_file, clinical_file)
        """
        # Generate EEG data
        eeg_columns = [
            "alpha_power",
            "beta_power",
            "theta_power",
            "delta_power",
            "gamma_power",
            "alpha_phase",
            "beta_phase",
            "theta_phase",
            "delta_phase",
            "gamma_phase"
        ]
        eeg_file = self.generate_from_columns(
            columns=eeg_columns,
            output_dir=output_dir,
            filename="eeg.csv"
        )
        
        # Generate behavior data
        behavior_columns = [
            "reaction_time",
            "accuracy",
            "confidence",
            "attention_score",
            "memory_score",
            "task_performance",
            "subjective_rating",
            "error_rate",
            "response_bias",
            "decision_certainty"
        ]
        behavior_file = self.generate_from_columns(
            columns=behavior_columns,
            output_dir=output_dir,
            filename="behavior.csv"
        )
        
        # Generate clinical data
        clinical_columns = [
            "depression_scale",
            "anxiety_scale",
            "stress_scale",
            "sleep_quality",
            "cognitive_function",
            "emotional_stability",
            "social_functioning",
            "quality_of_life",
            "treatment_response",
            "symptom_severity"
        ]
        clinical_file = self.generate_from_columns(
            columns=clinical_columns,
            output_dir=output_dir,
            filename="clinical.csv"
        )
        
        return eeg_file, behavior_file, clinical_file 