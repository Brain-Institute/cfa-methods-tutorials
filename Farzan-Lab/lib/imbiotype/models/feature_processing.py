"""
Unified Feature Processing Models

This module provides a unified architecture for feature processing that eliminates
duplication across CCA, clustering, and other analysis types. It replaces the
scattered preprocessing logic with a clean, modular system.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union, Any, TYPE_CHECKING
import json

if TYPE_CHECKING:
    from imbiotype.processing.feature_processor import PreprocessingConfig


@dataclass
class FeatureProcessingSettings:
    """
    Unified settings for feature processing pipeline.
    
    This replaces PCASettings, InputPreprocessingSettings, and DimReductionConfig
    with a single, comprehensive configuration class.
    """
    
    # Basic identification
    feature_name: str = ""
    enabled: bool = True
    
    # Data transformations (applied first)
    transform_method: str = "none"  # "none", "log", "sqrt", "zscore", "robust_scale"
    
    # Feature selection (applied second)
    feature_selection_enabled: bool = False
    feature_selection_method: str = "none"  # "none", "correlation", "variance", "univariate"
    
    # Feature selection parameters
    correlation_threshold: float = 0.95  # For correlation-based selection
    variance_threshold: float = 0.01     # For variance-based selection
    univariate_k: int = 10               # Number of features for univariate selection
    univariate_score_func: str = "f_classif"  # Score function for univariate selection
    max_features: Optional[int] = None   # Maximum number of features to keep
    
    # Dimensionality reduction (applied third)
    pca_enabled: bool = False
    pca_method: str = "explained_variance"  # "explained_variance", "num_components", "keep_all"
    pca_explained_variance: float = 0.95
    pca_num_components: int = 3
    
    # Standardization (applied last)
    standardize: bool = True
    standardization_method: str = "zscore"  # "zscore", "robust", "minmax"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary for storage and serialization"""
        return {
            "feature_name": self.feature_name,
            "enabled": self.enabled,
            "transform_method": self.transform_method,
            "feature_selection_enabled": self.feature_selection_enabled,
            "feature_selection_method": self.feature_selection_method,
            "correlation_threshold": self.correlation_threshold,
            "variance_threshold": self.variance_threshold,
            "univariate_k": self.univariate_k,
            "univariate_score_func": self.univariate_score_func,
            "max_features": self.max_features,
            "pca_enabled": self.pca_enabled,
            "pca_method": self.pca_method,
            "pca_explained_variance": self.pca_explained_variance,
            "pca_num_components": self.pca_num_components,
            "standardize": self.standardize,
            "standardization_method": self.standardization_method
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureProcessingSettings':
        """Create settings from dictionary"""
        return cls(
            feature_name=data.get("feature_name", ""),
            enabled=data.get("enabled", True),
            transform_method=data.get("transform_method", "none"),
            feature_selection_enabled=data.get("feature_selection_enabled", False),
            feature_selection_method=data.get("feature_selection_method", "none"),
            correlation_threshold=data.get("correlation_threshold", 0.95),
            variance_threshold=data.get("variance_threshold", 0.01),
            univariate_k=data.get("univariate_k", 10),
            univariate_score_func=data.get("univariate_score_func", "f_classif"),
            max_features=data.get("max_features"),
            pca_enabled=data.get("pca_enabled", False),
            pca_method=data.get("pca_method", "explained_variance"),
            pca_explained_variance=data.get("pca_explained_variance", 0.95),
            pca_num_components=data.get("pca_num_components", 3),
            standardize=data.get("standardize", True),
            standardization_method=data.get("standardization_method", "zscore")
        )
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'FeatureProcessingSettings':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))
    
    def copy(self) -> 'FeatureProcessingSettings':
        """Create a deep copy of the settings"""
        return self.from_dict(self.to_dict())
    
    def is_processing_enabled(self) -> bool:
        """Check if any processing is enabled"""
        return (
            self.enabled and (
                self.transform_method != "none" or
                self.feature_selection_enabled or
                self.pca_enabled or
                self.standardize
            )
        )
    
    def get_processing_summary(self) -> str:
        """Get a human-readable summary of processing steps"""
        if not self.is_processing_enabled():
            return "No processing"
        
        steps = []
        
        if self.transform_method != "none":
            steps.append(f"Transform: {self.transform_method}")
        
        if self.feature_selection_enabled and self.feature_selection_method != "none":
            if self.feature_selection_method == "correlation":
                steps.append(f"Feature Selection: Correlation (threshold={self.correlation_threshold})")
            elif self.feature_selection_method == "variance":
                steps.append(f"Feature Selection: Variance (threshold={self.variance_threshold})")
            elif self.feature_selection_method == "univariate":
                steps.append(f"Feature Selection: Univariate (k={self.univariate_k}, func={self.univariate_score_func})")
        
        if self.pca_enabled:
            if self.pca_method == "explained_variance":
                steps.append(f"PCA: {self.pca_explained_variance*100:.1f}% variance")
            elif self.pca_method == "num_components":
                steps.append(f"PCA: {self.pca_num_components} components")
            else:
                steps.append("PCA: Keep all")
        
        if self.standardize:
            steps.append(f"Standardize: {self.standardization_method}")
        
        return " → ".join(steps)


@dataclass
class FeatureProcessingConfig:
    """
    Configuration linking a feature vector to its processing settings.

    This replaces DimReductionConfig and provides a cleaner interface.
    Supports both legacy processing_settings and unified preprocessing_config.

    Note: Either processing_settings or preprocessing_config should be provided.
    If both are provided, preprocessing_config takes precedence.
    """
    feature_name: str
    processing_settings: Optional[FeatureProcessingSettings] = None  # Legacy interface (optional)
    preprocessing_config: Optional['PreprocessingConfig'] = None  # For unified architecture
    preprocessing_states: Optional[Dict[str, Any]] = None  # For enabled/disabled states
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            "feature_name": self.feature_name
        }

        # Include legacy processing settings if available
        if self.processing_settings is not None:
            result["processing_settings"] = self.processing_settings.to_dict()

        # Include preprocessing config if available (unified architecture)
        if self.preprocessing_config is not None:
            # Convert PreprocessingConfig to dict
            preprocessing_dict = {}
            for attr_name in ['transform', 'variance_filter_pre', 'residualize', 'variance_filter_post',
                             'standardize', 'univariate', 'redundancy_prune', 'pca']:
                attr_value = getattr(self.preprocessing_config, attr_name, None)
                if attr_value is not None:
                    preprocessing_dict[attr_name] = attr_value

            if preprocessing_dict:
                result["preprocessing_config"] = preprocessing_dict

        # Include preprocessing states if available (enabled/disabled states)
        if self.preprocessing_states is not None:
            result["preprocessing_states"] = self.preprocessing_states

        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FeatureProcessingConfig':
        """Create from dictionary"""
        # Load legacy processing settings if available
        processing_settings = None
        if "processing_settings" in data:
            processing_settings = FeatureProcessingSettings.from_dict(data["processing_settings"])

        # Create config with feature name and optional processing settings
        config = cls(
            feature_name=data["feature_name"],
            processing_settings=processing_settings
        )

        # Load preprocessing config if available (unified architecture)
        if "preprocessing_config" in data:
            try:
                from imbiotype.processing.feature_processor import PreprocessingConfig
                preprocessing_data = data["preprocessing_config"]
                config.preprocessing_config = PreprocessingConfig(**preprocessing_data)
            except ImportError:
                # Unified architecture not available, create a simple object with the data
                class SimplePreprocessingConfig:
                    def __init__(self, **kwargs):
                        for key, value in kwargs.items():
                            setattr(self, key, value)

                config.preprocessing_config = SimplePreprocessingConfig(**preprocessing_data)

        # Load preprocessing states if available (enabled/disabled states)
        if "preprocessing_states" in data:
            config.preprocessing_states = data["preprocessing_states"]

        return config



