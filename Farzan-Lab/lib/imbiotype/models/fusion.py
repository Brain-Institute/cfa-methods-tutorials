"""
Data fusion model implementation with unified feature processing architecture
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
import difflib
from ..ml_functions import run_pca, run_cca
import statsmodels.formula.api as smf
from .feature_processing import FeatureProcessingSettings, FeatureProcessingConfig




@dataclass
class ModelSettings:
    """Core CCA model parameters"""
    # Covariate residualization
    x_covariate_feature_vector_name: Optional[str] = None
    y_covariate_feature_vector_name: Optional[str] = None

    # CCA variant (mutually exclusive)
    cca_type: str = "classic"  # "classic", "regularized", "sparse", "kernel", "deep", "partial"

    # Regularization (for regularized CCA)
    regularization_x: float = 0.1  # lambda_x from MATLAB
    regularization_y: float = 0.1  # lambda_y from MATLAB
    regularization_method: str = "covariance"  # "covariance" (MATLAB-style) or "L2" (CCA-Zoo style)

    # Sparsity (for sparse CCA)
    sparsity_penalty_x: float = 0.0
    sparsity_penalty_y: float = 0.0

    # Core parameters
    n_components: int = 10

    def to_dict(self) -> Dict:
        """Convert settings to dictionary"""
        return {
            "x_covariate_feature_vector_name": self.x_covariate_feature_vector_name,
            "y_covariate_feature_vector_name": self.y_covariate_feature_vector_name,
            "cca_type": self.cca_type,
            "regularization_x": self.regularization_x,
            "regularization_y": self.regularization_y,
            "regularization_method": self.regularization_method,
            "sparsity_penalty_x": self.sparsity_penalty_x,
            "sparsity_penalty_y": self.sparsity_penalty_y,
            "n_components": self.n_components
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'ModelSettings':
        """Create settings from dictionary"""
        # Handle backward compatibility for old use_matlab_style_regularization field
        if "regularization_method" in data:
            # New field takes precedence
            regularization_method = data["regularization_method"]
        elif "use_matlab_style_regularization" in data:
            # Migrate from old boolean field
            regularization_method = "covariance" if data["use_matlab_style_regularization"] else "L2"
        else:
            # Default value
            regularization_method = "covariance"

        return cls(
            x_covariate_feature_vector_name=data.get("x_covariate_feature_vector_name"),
            y_covariate_feature_vector_name=data.get("y_covariate_feature_vector_name"),
            cca_type=data.get("cca_type", "classic"),
            regularization_x=data.get("regularization_x", 0.1),
            regularization_y=data.get("regularization_y", 0.1),
            regularization_method=regularization_method,
            sparsity_penalty_x=data.get("sparsity_penalty_x", 0.0),
            sparsity_penalty_y=data.get("sparsity_penalty_y", 0.0),
            n_components=data.get("n_components", 10)
        )

@dataclass
class OptimizationSettings:
    """Hyperparameter optimization configuration"""
    # Cross-validation setup
    cv_method: str = "kfold"  # "kfold", "stratified", "leave_one_out"
    cv_folds: int = 5
    stratify_by: Optional[str] = None  # "site", "group", etc.
    random_state: Optional[int] = 42  # Random seed for reproducible CV splits

    # What to optimize
    optimize_regularization: bool = True
    optimize_n_components: bool = False  # Usually fixed in practice
    optimize_sparsity: bool = False
    optimize_feature_selection: bool = False

    # Parameter search spaces
    regularization_x_range: Tuple[float, float] = (0.001, 1.0)  # logspace(-3, 1) from MATLAB
    regularization_y_range: Tuple[float, float] = (0.001, 1.0)
    regularization_search_points: int = 20  # 20 points in MATLAB script (used if x/y values not specified)

    # Explicit lambda values (if provided, these override ranges and search_points)
    regularization_x_values: Optional[List[float]] = None  # Explicit lambda_x values
    regularization_y_values: Optional[List[float]] = None  # Explicit lambda_y values

    n_components_range: Tuple[int, int] = (1, 10)
    sparsity_range: Tuple[float, float] = (0.0, 1.0)

    # Optimization method
    search_method: str = "grid_search"  # "grid_search", "random_search"

    # Computational limits
    max_evaluations: int = 1000
    n_repeats: int = 1000  # training_nfit from MATLAB (for stability)

    # Objective function
    objective_metric: str = "correlation"  # "correlation", "explained_variance"

    # Validation logging (for MATLAB comparison)
    validation_logging_enabled: bool = False
    validation_log_dir: Optional[str] = None
    validation_log_repeats: int = 1  # How many repeats to log in detail
    validation_log_sample_size: int = 10  # How many data rows to save
    validation_debug_mode: bool = False  # Enable detailed debug logging

    # MATLAB CV indices import (for exact replication)
    matlab_cv_indices_dir: Optional[str] = None  # Directory containing MATLAB CV indices .mat files

    def to_dict(self) -> Dict:
        """Convert settings to dictionary"""
        return {
            "cv_method": self.cv_method,
            "cv_folds": self.cv_folds,
            "stratify_by": self.stratify_by,
            "optimize_regularization": self.optimize_regularization,
            "optimize_n_components": self.optimize_n_components,
            "optimize_sparsity": self.optimize_sparsity,
            "optimize_feature_selection": self.optimize_feature_selection,
            "regularization_x_range": self.regularization_x_range,
            "regularization_y_range": self.regularization_y_range,
            "regularization_search_points": self.regularization_search_points,
            "regularization_x_values": self.regularization_x_values,
            "regularization_y_values": self.regularization_y_values,
            "n_components_range": self.n_components_range,
            "sparsity_range": self.sparsity_range,
            "matlab_cv_indices_dir": self.matlab_cv_indices_dir,
            "search_method": self.search_method,
            "max_evaluations": self.max_evaluations,
            "n_repeats": self.n_repeats,
            "objective_metric": self.objective_metric,
            "validation_logging_enabled": self.validation_logging_enabled,
            "validation_log_dir": self.validation_log_dir,
            "validation_log_repeats": self.validation_log_repeats,
            "validation_log_sample_size": self.validation_log_sample_size,
            "validation_debug_mode": self.validation_debug_mode
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'OptimizationSettings':
        """Create settings from dictionary"""
        return cls(
            cv_method=data.get("cv_method", "kfold"),
            cv_folds=data.get("cv_folds", 5),
            stratify_by=data.get("stratify_by"),
            random_state=data.get("random_state", 42),
            optimize_regularization=data.get("optimize_regularization", True),
            optimize_n_components=data.get("optimize_n_components", False),
            optimize_sparsity=data.get("optimize_sparsity", False),
            optimize_feature_selection=data.get("optimize_feature_selection", False),
            regularization_x_range=tuple(data.get("regularization_x_range", (0.001, 1.0))),
            regularization_y_range=tuple(data.get("regularization_y_range", (0.001, 1.0))),
            regularization_search_points=data.get("regularization_search_points", 20),
            regularization_x_values=data.get("regularization_x_values"),
            regularization_y_values=data.get("regularization_y_values"),
            n_components_range=tuple(data.get("n_components_range", (1, 10))),
            sparsity_range=tuple(data.get("sparsity_range", (0.0, 1.0))),
            search_method=data.get("search_method", "grid_search"),
            max_evaluations=data.get("max_evaluations", 1000),
            n_repeats=data.get("n_repeats", 1000),
            objective_metric=data.get("objective_metric", "correlation"),
            validation_logging_enabled=data.get("validation_logging_enabled", False),
            validation_log_dir=data.get("validation_log_dir"),
            validation_log_repeats=data.get("validation_log_repeats", 1),
            validation_log_sample_size=data.get("validation_log_sample_size", 10),
            validation_debug_mode=data.get("validation_debug_mode", False),
            matlab_cv_indices_dir=data.get("matlab_cv_indices_dir")
        )

@dataclass
class StatisticalTestingSettings:
    """Permutation testing and significance assessment"""
    enabled: bool = False

    # Permutation setup
    test_type: str = "shuffle_y"  # "shuffle_x", "shuffle_y", "shuffle_both"
    n_permutations: int = 10000  # From MATLAB scripts
    respect_groups: bool = True  # Permute within groups (sites) as in MATLAB
    group_variable: Optional[str] = None  # e.g., "site" for stratified permutation
    random_state: Optional[int] = None  # For reproducible permutation testing

    # Statistical parameters
    alpha: float = 0.05
    multiple_comparisons_correction: str = "familywise"  # "bonferroni", "fdr", "familywise"

    # What to test
    test_statistic: str = "canonical_correlation"  # "canonical_correlation", "log_likelihood"
    test_individual_components: bool = True
    test_aggregate_statistic: bool = True

    def to_dict(self) -> Dict:
        """Convert settings to dictionary"""
        return {
            "enabled": self.enabled,
            "test_type": self.test_type,
            "n_permutations": self.n_permutations,
            "respect_groups": self.respect_groups,
            "group_variable": self.group_variable,
            "random_state": self.random_state,
            "alpha": self.alpha,
            "multiple_comparisons_correction": self.multiple_comparisons_correction,
            "test_statistic": self.test_statistic,
            "test_individual_components": self.test_individual_components,
            "test_aggregate_statistic": self.test_aggregate_statistic
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'StatisticalTestingSettings':
        """Create settings from dictionary"""
        return cls(
            enabled=data.get("enabled", False),
            test_type=data.get("test_type", "shuffle_y"),
            n_permutations=data.get("n_permutations", 10000),
            respect_groups=data.get("respect_groups", True),
            group_variable=data.get("group_variable"),
            random_state=data.get("random_state"),
            alpha=data.get("alpha", 0.05),
            multiple_comparisons_correction=data.get("multiple_comparisons_correction", "familywise"),
            test_statistic=data.get("test_statistic", "canonical_correlation"),
            test_individual_components=data.get("test_individual_components", True),
            test_aggregate_statistic=data.get("test_aggregate_statistic", True)
        )

@dataclass
class BootstrapSettings:
    """Bootstrap resampling for uncertainty quantification of CCA loadings"""
    enabled: bool = False

    # Bootstrap setup
    n_iterations: int = 10000  # From MATLAB: bootci(10000, ...)
    confidence_level: float = 0.95  # 95% confidence intervals
    random_state: Optional[int] = None  # For reproducible bootstrap

    # What to bootstrap
    compute_loading_ci: bool = True  # Confidence intervals for structural loadings

    # Significance testing for loadings
    compute_loading_pvalues: bool = True  # P-values for each loading
    loading_correction_method: str = "fdr"  # "fdr", "bonferroni", "none"
    loading_alpha: float = 0.05  # Significance threshold for loadings

    def to_dict(self) -> Dict:
        """Convert settings to dictionary"""
        return {
            "enabled": self.enabled,
            "n_iterations": self.n_iterations,
            "confidence_level": self.confidence_level,
            "random_state": self.random_state,
            "compute_loading_ci": self.compute_loading_ci,
            "compute_loading_pvalues": self.compute_loading_pvalues,
            "loading_correction_method": self.loading_correction_method,
            "loading_alpha": self.loading_alpha
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'BootstrapSettings':
        """Create settings from dictionary"""
        return cls(
            enabled=data.get("enabled", False),
            n_iterations=data.get("n_iterations", 10000),
            confidence_level=data.get("confidence_level", 0.95),
            random_state=data.get("random_state"),
            compute_loading_ci=data.get("compute_loading_ci", True),
            compute_loading_pvalues=data.get("compute_loading_pvalues", True),
            loading_correction_method=data.get("loading_correction_method", "fdr"),
            loading_alpha=data.get("loading_alpha", 0.05)
        )

@dataclass
class CCASettings:
    """Enhanced hierarchical settings for CCA analysis"""
    # Basic settings
    skip_cca: bool = False
    x_main_feature_vector_name: Optional[str] = None
    y_main_feature_vector_name: Optional[str] = None

    # Note: Input preprocessing is now handled by FeatureProcessingSettings in FusionConfig
    hyperparameter_optimization_enabled: bool = False
    model_settings: ModelSettings = field(default_factory=ModelSettings)
    optimization_settings: OptimizationSettings = field(default_factory=OptimizationSettings)
    statistical_testing: StatisticalTestingSettings = field(default_factory=StatisticalTestingSettings)
    bootstrap_settings: BootstrapSettings = field(default_factory=BootstrapSettings)

    # Advanced features (for future expansion)
    is_multiview: bool = False
    views: List[List[str]] = field(default_factory=list)

    # Legacy compatibility properties
    @property
    def use_rcca(self) -> bool:
        """Legacy compatibility: map to model_settings.cca_type"""
        return self.model_settings.cca_type == "regularized"

    @use_rcca.setter
    def use_rcca(self, value: bool):
        """Legacy compatibility: set model_settings.cca_type"""
        if value:
            self.model_settings.cca_type = "regularized"
        else:
            self.model_settings.cca_type = "classic"

    # Note: use_prepca removed - use FeatureProcessingSettings instead

    @property
    def regularization(self) -> float:
        """Legacy compatibility: map to model_settings.regularization_x"""
        return self.model_settings.regularization_x

    @regularization.setter
    def regularization(self, value: float):
        """Legacy compatibility: set both regularization parameters"""
        self.model_settings.regularization_x = value
        self.model_settings.regularization_y = value

    @property
    def use_matlab_style_regularization(self) -> bool:
        """Legacy compatibility: map to model_settings.regularization_method"""
        return self.model_settings.regularization_method == "covariance"

    @use_matlab_style_regularization.setter
    def use_matlab_style_regularization(self, value: bool):
        """Legacy compatibility: set model_settings.regularization_method"""
        self.model_settings.regularization_method = "covariance" if value else "L2"

    @property
    def c(self) -> float:
        """Legacy compatibility: map to optimization_settings.cv_folds when optimization enabled"""
        if self.hyperparameter_optimization_enabled:
            return float(self.optimization_settings.cv_folds)
        else:
            return 0.0

    @c.setter
    def c(self, value: float):
        """Legacy compatibility: set optimization enabled based on c value"""
        if value > 0:
            self.hyperparameter_optimization_enabled = True
            self.optimization_settings.cv_folds = int(value)
        else:
            self.hyperparameter_optimization_enabled = False

    @property
    def n_components(self) -> int:
        """Legacy compatibility: map to model_settings.n_components"""
        return self.model_settings.n_components

    @n_components.setter
    def n_components(self, value: int):
        """Legacy compatibility: set model_settings.n_components"""
        self.model_settings.n_components = value

    @property
    def x_covariate_feature_vector_name(self) -> Optional[str]:
        """Legacy compatibility: map to model_settings.x_covariate_feature_vector_name"""
        return self.model_settings.x_covariate_feature_vector_name

    @x_covariate_feature_vector_name.setter
    def x_covariate_feature_vector_name(self, value: Optional[str]):
        """Legacy compatibility: set model_settings.x_covariate_feature_vector_name"""
        self.model_settings.x_covariate_feature_vector_name = value

    @property
    def y_covariate_feature_vector_name(self) -> Optional[str]:
        """Legacy compatibility: map to model_settings.y_covariate_feature_vector_name"""
        return self.model_settings.y_covariate_feature_vector_name

    @y_covariate_feature_vector_name.setter
    def y_covariate_feature_vector_name(self, value: Optional[str]):
        """Legacy compatibility: set model_settings.y_covariate_feature_vector_name"""
        self.model_settings.y_covariate_feature_vector_name = value

    def to_dict(self) -> Dict:
        """Convert settings to dictionary for storage"""
        return {
            "skip_cca": self.skip_cca,
            "x_main_feature_vector_name": self.x_main_feature_vector_name,
            "y_main_feature_vector_name": self.y_main_feature_vector_name,
            "hyperparameter_optimization_enabled": self.hyperparameter_optimization_enabled,
            "model_settings": self.model_settings.to_dict(),
            "optimization_settings": self.optimization_settings.to_dict(),
            "statistical_testing": self.statistical_testing.to_dict(),
            "bootstrap_settings": self.bootstrap_settings.to_dict(),
            "is_multiview": self.is_multiview,
            "views": self.views
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'CCASettings':
        """Create settings from dictionary"""
        # Create settings with clean structure
        settings = cls(
            skip_cca=data.get("skip_cca", False),
            x_main_feature_vector_name=data.get("x_main_feature_vector_name", data.get("x_features")),
            y_main_feature_vector_name=data.get("y_main_feature_vector_name", data.get("y_features")),
            hyperparameter_optimization_enabled=data.get("hyperparameter_optimization_enabled", False),
            model_settings=ModelSettings.from_dict(data.get("model_settings", {})),
            optimization_settings=OptimizationSettings.from_dict(data.get("optimization_settings", {})),
            statistical_testing=StatisticalTestingSettings.from_dict(data.get("statistical_testing", {})),
            bootstrap_settings=BootstrapSettings.from_dict(data.get("bootstrap_settings", {})),
            is_multiview=data.get("is_multiview", False),
            views=data.get("views", [])
        )

        # Handle legacy fields if present
        if "use_rcca" in data:
            settings.model_settings.cca_type = "regularized" if data.get("use_rcca", False) else "classic"
        if "regularization" in data:
            settings.model_settings.regularization_x = data.get("regularization", 0.1)
            settings.model_settings.regularization_y = data.get("regularization", 0.1)
        if "n_components" in data:
            settings.model_settings.n_components = data.get("n_components", 10)
        if "x_covariate_feature_vector_name" in data:
            settings.model_settings.x_covariate_feature_vector_name = data.get("x_covariate_feature_vector_name")
        if "y_covariate_feature_vector_name" in data:
            settings.model_settings.y_covariate_feature_vector_name = data.get("y_covariate_feature_vector_name")

        # Handle c parameter (cross-validation)
        c_value = data.get("c", 0.0)
        if c_value > 0:
            settings.hyperparameter_optimization_enabled = True
            settings.optimization_settings.cv_folds = int(c_value)

        return settings



@dataclass
class FusionConfig:
    """
    Configuration for fusion analysis with unified feature processing.
    """
    name: str
    cca_settings: CCASettings
    feature_processing_configs: Optional[List[FeatureProcessingConfig]] = None

    def to_dict(self) -> Dict:
        """Convert config to dictionary for storage"""
        result = {
            "name": self.name,
            "cca_settings": self.cca_settings.to_dict()
        }

        # Include feature processing configs
        if self.feature_processing_configs is not None:
            result["feature_processing_configs"] = [
                config.to_dict() for config in self.feature_processing_configs
            ]

        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'FusionConfig':
        """Create config from dictionary"""

        # Load CCA settings
        cca_settings_data = data.get("cca_settings", {})
        cca_settings = CCASettings.from_dict(cca_settings_data)

        # Load feature processing configs
        feature_processing_configs = None
        feature_processing_data = data.get("feature_processing_configs")
        if feature_processing_data is not None:
            feature_processing_configs = [
                FeatureProcessingConfig.from_dict(config_data)
                for config_data in feature_processing_data
            ]

        return cls(
            name=data["name"],
            cca_settings=cca_settings,
            feature_processing_configs=feature_processing_configs
        )



class FusionModel:
    def validate_config(self, config: 'FusionConfig', feature_data: Dict[str, pd.DataFrame]) -> List[str]:
        """
        Validate fusion configuration against available data.

        Returns:
            List of validation errors (empty if valid)
        """
        errors: List[str] = []
        available_features = set(feature_data.keys())

        # Helper to suggest close feature name matches
        def suggest(name: str) -> str:
            if not available_features:
                return ""
            choices = list(available_features)
            close = difflib.get_close_matches(name or "", choices, n=1)
            return f" Did you mean '{close[0]}'?" if close else ""

        # Validate feature processing settings
        if config.feature_processing_configs:
            for processing_config in config.feature_processing_configs:
                if processing_config.feature_name not in available_features:
                    errors.append(
                        f"Processing feature '{processing_config.feature_name}' not found in available features: {sorted(list(available_features))}." + suggest(processing_config.feature_name)
                    )

        # Validate CCA settings
        cca = config.cca_settings
        if not cca.skip_cca:
            # X feature
            x_feature = cca.x_main_feature_vector_name
            if not x_feature:
                errors.append("CCA X feature name is required when CCA is enabled")
            elif x_feature not in available_features:
                errors.append(
                    f"CCA X feature '{x_feature}' not found in available features: {sorted(list(available_features))}." + suggest(x_feature)
                )

            # Y feature
            y_feature = cca.y_main_feature_vector_name
            if not y_feature:
                errors.append("CCA Y feature name is required when CCA is enabled")
            elif y_feature not in available_features:
                errors.append(
                    f"CCA Y feature '{y_feature}' not found in available features: {sorted(list(available_features))}." + suggest(y_feature)
                )

            # Covariates
            x_cov = cca.model_settings.x_covariate_feature_vector_name
            if x_cov and x_cov not in available_features:
                errors.append(
                    f"CCA X covariate feature '{x_cov}' not found in available features: {sorted(list(available_features))}." + suggest(x_cov)
                )
            y_cov = cca.model_settings.y_covariate_feature_vector_name
            if y_cov and y_cov not in available_features:
                errors.append(
                    f"CCA Y covariate feature '{y_cov}' not found in available features: {sorted(list(available_features))}." + suggest(y_cov)
                )

            # Multiview
            if getattr(cca, 'is_multiview', False):
                if not cca.views:
                    errors.append("Multiview CCA requires 'views' to be specified")
                elif len(cca.views) < 2:
                    errors.append("Multiview CCA requires at least 2 views")
                else:
                    for i, view in enumerate(cca.views):
                        if isinstance(view, str) and view not in available_features:
                            errors.append(
                                f"Multiview CCA view {i} '{view}' not found in available features: {sorted(list(available_features))}." + suggest(view)
                            )

        # Data compatibility: sample counts
        if feature_data:
            sample_counts = {name: int(len(df)) for name, df in feature_data.items()}
            unique_counts = set(sample_counts.values())
            if len(unique_counts) > 1:
                errors.append(
                    f"Feature vectors have different sample counts: {sample_counts}. Ensure all feature vectors have the same number of samples."
                )

        # Statistical testing
        st = cca.statistical_testing
        if st.enabled and st.respect_groups:
            group_var = st.group_variable
            if not group_var:
                errors.append("Group variable name required when respect_groups=True")
            # Optional: we could check presence across feature_data here

        return errors

    def validate_and_run(self, feature_data: Dict[str, pd.DataFrame],
                         config: 'FusionConfig', output_dir: str,
                         progress_callback=None) -> Dict:
        """Run fusion with comprehensive validation"""
        errors = self.validate_config(config, feature_data)
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            raise ValueError(error_msg)
        return self.run(feature_data, config, output_dir, progress_callback)

    """
    Implements data fusion algorithms including PCA and CCA
    """

    def _residualize_data(self, data_df: pd.DataFrame, feature_columns: List[str], covariate_columns: List[str]) -> pd.DataFrame:
        """
        Residualize feature columns against specified covariate columns using OLS.
        Assumes feature_columns and covariate_columns are present in data_df.
        """
        if not covariate_columns:
            return data_df[feature_columns].copy()

        residualized_features_df = pd.DataFrame(index=data_df.index)

        formula_rhs = " + ".join(covariate_columns)

        for feature_col in feature_columns:
            if feature_col in covariate_columns: # Skip if a feature is also a covariate
                residualized_features_df[feature_col] = data_df[feature_col]
                print(f"Warning: Feature column '{feature_col}' is also in covariate list. Skipping residualization for it.")
                continue
            try:
                formula = f"Q('{feature_col}') ~ {formula_rhs}" # Use Q('') for special characters in column names
                model = smf.ols(formula, data=data_df).fit()
                residualized_features_df[feature_col] = model.resid
            except Exception as e:
                print(f"Error residualizing {feature_col} with formula '{formula}': {e}")
                # Fallback: use original data for this feature if residualization fails
                residualized_features_df[feature_col] = data_df[feature_col]

        return residualized_features_df

    def run(self, feature_data: Dict[str, pd.DataFrame],
            config: 'FusionConfig', output_dir: str, progress_callback=None) -> Dict:
        """
        Run fusion analysis on feature data

        Args:
            feature_data: Dictionary of feature vectors as DataFrames
            config: Configuration for the fusion analysis
            output_dir: Directory to save results
            progress_callback: Optional callback function for progress updates (fn(progress_pct, message))

        Returns:
            Dictionary of results and metrics
        """
        results = {
            "metrics": {
                "explained_variance": 0.0,
                "correlation": 0.0
            }
        }

        # Initialize progress tracking
        if progress_callback:
            progress_callback(0, "Starting fusion analysis...")

        # Store raw feature data for independent pipelines
        raw_feature_data = {}
        for name, df in feature_data.items():
            if isinstance(df, pd.DataFrame):
                raw_feature_data[name] = df.copy()
            elif isinstance(df, np.ndarray):
                raw_feature_data[name] = pd.DataFrame(df)
            else:
                print(f"Warning: Feature '{name}' is not a DataFrame or array. Converting to DataFrame.")
                raw_feature_data[name] = pd.DataFrame(df)

        # Separate pipelines for feature processing and CCA
        processed_features = {}
        raw_features = {}

        # MATLAB-COMPATIBLE ARCHITECTURE:
        # When HPO is enabled, only apply outside-CV steps here.
        # Inside-CV steps (residualize, standardize, PCA) will be applied per-fold by CCA.
        # When HPO is disabled, apply all steps here.

        hpo_enabled = (not config.cca_settings.skip_cca and
                      config.cca_settings.hyperparameter_optimization_enabled)

        # Pipeline 1: Feature processing
        if config.feature_processing_configs:
            from ..processing.feature_processor import (
                PreprocessingConfig, build_processing_plan,
                apply_outside_cv_steps
            )

            if progress_callback:
                if hpo_enabled:
                    progress_callback(5, "Applying outside-CV preprocessing steps...")
                else:
                    progress_callback(5, "Applying all preprocessing steps...")

            # Determine mode based on HPO setting
            mode = "with_hpo" if hpo_enabled else "no_hpo"

            if hpo_enabled:
                print("[Fusion] MATLAB-compatible mode: Applying only outside-CV steps before CCA")
            else:
                print("[Fusion] No HPO mode: Applying all preprocessing steps before CCA")

            processed_data = {}
            for processing_config in config.feature_processing_configs:
                feature_name = processing_config.feature_name
                if feature_name not in raw_feature_data:
                    continue

                # Get the preprocessing config
                if hasattr(processing_config, 'preprocessing_config') and processing_config.preprocessing_config is not None:
                    prep_config = processing_config.preprocessing_config
                else:
                    # No preprocessing config - use raw data as DataFrame (preserve column names)
                    processed_data[feature_name] = raw_feature_data[feature_name]
                    continue

                # Build processing plan
                plan = build_processing_plan(prep_config, mode)

                # Apply outside-CV steps (in with_hpo mode, this is only transform/variance_filter_pre)
                # In no_hpo mode, this includes all steps
                # Note: Using generic variable names here - feature_name determines which CCA view this is
                feature_raw = raw_feature_data[feature_name]
                feature_processed = apply_outside_cv_steps(plan, feature_raw, fit=True)

                # Store as DataFrame to preserve column names
                # apply_outside_cv_steps returns a DataFrame with preserved column names
                processed_data[feature_name] = feature_processed

                print(f"[Fusion] Applied {mode} outside-CV steps to '{feature_name}': {feature_raw.shape} -> {feature_processed.shape}")

            # Store processed features
            # processed_data now contains DataFrames with preserved column names
            for feature_name, processed_df in processed_data.items():
                # apply_outside_cv_steps returns a DataFrame with column names preserved
                # Just store it directly
                processed_features[feature_name] = processed_df
                raw_features[feature_name] = processed_df.values if hasattr(processed_df, 'values') else processed_df

                # Log column preservation
                if hasattr(processed_df, 'columns'):
                    print(f"[Fusion] Preserved column names for '{feature_name}': {list(processed_df.columns)[:3]}... (n={len(processed_df.columns)})")
                else:
                    print(f"[Fusion] No column names for '{feature_name}' (not a DataFrame)")

            # Store unprocessed features as-is
            for name, df in raw_feature_data.items():
                if name not in processed_features:
                    processed_features[name] = df
                    raw_features[name] = df.values

            # Notify completion of feature processing
            if progress_callback:
                if hpo_enabled:
                    progress_callback(15, "Outside-CV preprocessing complete (inside-CV steps will be applied per-fold)")
                else:
                    progress_callback(15, "All preprocessing complete")
        else:
            # No feature_processing_configs: use input feature_data directly
            print("No feature_processing_configs provided. Raw data will be used.")
            for name, df in raw_feature_data.items():
                processed_features[name] = df
                raw_features[name] = df.values

            if progress_callback:
                progress_callback(15, "No feature processing needed")

        # Pipeline 2: CCA processing (independent, works on raw data)
        cca_results = None
        if not config.cca_settings.skip_cca:
            if config.cca_settings.hyperparameter_optimization_enabled:
                if progress_callback:
                    progress_callback(15, "Starting CCA hyperparameter optimization...")
            else:
                if progress_callback:
                    progress_callback(15, "Running Canonical Correlation Analysis...")

            # CCA works on processed feature data from FeatureProcessor
            # Use processed_features which contains the data after feature processing pipeline
            # Pass the progress callback through to CCA for detailed hyperparameter optimization progress
            # Create preprocessing_steps directory for intermediate CSV logging
            preprocessing_output_dir = os.path.join(output_dir, "preprocessing_steps")
            cca_results = self._run_cca(processed_features, config.cca_settings,
                                       progress_callback=progress_callback,
                                       preprocessing_output_dir=preprocessing_output_dir,
                                       feature_processing_configs=config.feature_processing_configs)
            # Ensure feature names are captured at analysis time for plotting/UI
            try:
                x_name = getattr(config.cca_settings, "x_main_feature_vector_name", None)
                y_name = getattr(config.cca_settings, "y_main_feature_vector_name", None)
                if isinstance(cca_results, dict):
                    if x_name and x_name in processed_features and not cca_results.get("x_feature_names"):
                        x_obj = processed_features.get(x_name)
                        x_cols = getattr(x_obj, "columns", None)
                        cca_results["x_feature_names"] = list(x_cols) if x_cols is not None else []
                        print(f"[Fusion] Captured x_feature_names (n={len(cca_results['x_feature_names'])}) for '{x_name}'")
                    if y_name and y_name in processed_features and not cca_results.get("y_feature_names"):
                        y_obj = processed_features.get(y_name)
                        y_cols = getattr(y_obj, "columns", None)
                        cca_results["y_feature_names"] = list(y_cols) if y_cols is not None else []
                        print(f"[Fusion] Captured y_feature_names (n={len(cca_results['y_feature_names'])}) for '{y_name}'")
            except Exception as e:
                print(f"Warning: failed to capture feature names for CCA results: {e}")

            results["cca"] = cca_results

            # Update metrics with correlation
            if "correlations" in cca_results and \
               cca_results["correlations"] is not None:
                correlations = cca_results["correlations"]
                # Handle both lists and numpy arrays
                if hasattr(correlations, 'size'):
                    has_correlations = correlations.size > 0
                else:
                    has_correlations = len(correlations) > 0

                if has_correlations:
                    results["metrics"]["correlation"] = np.mean(correlations)
        # Build standardized results and clustering input
        processing_info = {
            "feature_names": list(raw_feature_data.keys()),
            "n_samples": len(next(iter(raw_feature_data.values()))),
            "processed_features": processed_features,
        }
        standardized = self._create_standardized_results(
            pca_results=results.get("pca"),
            cca_results=cca_results,
            config=config,
            processing_info=processing_info,
        )

        # Save standardized JSON alongside npz
        std_path = os.path.join(output_dir, "standardized_results.json")
        try:
            with open(std_path, "w") as f:
                json.dump({
                    "metadata": standardized["metadata"],
                    "results": {
                        "pca": standardized["results"]["pca"],
                        "cca": standardized["results"]["cca"],
                    },
                }, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
        except Exception as e:
            print(f"Warning: failed to write standardized_results.json: {e}")

        # Create clustering input from both pipelines
        if progress_callback:
            progress_callback(95, "Preparing results...")

        # Create and save standardized clustering input
        clustering_input = self._create_clustering_input(
            raw_features=raw_features,
            processed_features=processed_features,
            pca_results=results.get("pca"),
            cca_results=cca_results,
            config=config
        )

        # Save results
        if progress_callback:
            progress_callback(98, "Saving results...")

        # Save comprehensive results
        results_path = os.path.join(output_dir, "results.npz")
        np.savez(results_path, **results)

        clustering_path = os.path.join(output_dir, "clustering_input.npz")
        np.savez(clustering_path, **clustering_input)

        # Save PCA components to CSV files if available
        self._save_pca_components_to_csv(results, output_dir, config)

        # Save CCA results to CSV files if available
        self._save_cca_results_to_csv(results, output_dir, config)

        # Final completion notification
        if progress_callback:
            progress_callback(100, "Fusion analysis complete")

        return results

    def _create_clustering_input(self, raw_features: Dict[str, np.ndarray],
                               processed_features: Dict[str, np.ndarray],
                               pca_results: Optional[Dict],
                               cca_results: Optional[Dict],
                               config: 'FusionConfig') -> Dict:
        """
        Create standardized clustering input combining results from separate feature processing and CCA pipelines

        Args:
            raw_features: Dictionary of raw feature arrays (pass-through features from feature processing pipeline)
            processed_features: Dictionary of feature-processed arrays (after residualization, standardization, PCA, etc.)
            pca_results: Optional PCA results dictionary
            cca_results: Optional CCA results dictionary
            config: Fusion configuration

        Returns:
            Dictionary containing standardized clustering input from both pipelines
        """
        clustering_input = {}

        # Add PCA results if available
        if pca_results:
            for feature_name, pca_result_data in pca_results.items():
                if 'transformed_data' in pca_result_data:
                    clustering_input[f'{feature_name}_pca'] = pca_result_data['transformed_data']

        # Add processed features from PCA pipeline (including pass-through and non-PCA features)
        if processed_features:
            for feature_name, data_df in processed_features.items():
                # Convert DataFrame to array if needed
                if isinstance(data_df, pd.DataFrame):
                    data_array = data_df.values
                else:
                    data_array = data_df

                # Only add if not already added as PCA result
                if f'{feature_name}_pca' not in clustering_input:
                    clustering_input[f'{feature_name}_processed'] = data_array

        # Add CCA results if available (independent of PCA)
        if cca_results and not config.cca_settings.skip_cca:
            if config.cca_settings.is_multiview:
                # For multiview, transformed_data is a list of arrays, one per view
                if 'transformed_data' in cca_results:
                    transformed_views = cca_results['transformed_data']
                    if isinstance(transformed_views, list):
                        # Add each view separately
                        for i, view_data in enumerate(transformed_views):
                            clustering_input[f'cca_view_{i}'] = view_data
                        # Also add concatenated version
                        clustering_input['cca_multiview_combined'] = np.concatenate(transformed_views, axis=1)
                    else:
                        clustering_input['cca_multiview'] = transformed_views
            else:
                # Two-view CCA: transformed_data is a list [x_transformed, y_transformed]
                if 'transformed_data' in cca_results:
                    transformed_data = cca_results['transformed_data']
                    if isinstance(transformed_data, list) and len(transformed_data) == 2:
                        clustering_input['cca_x_transformed'] = transformed_data[0]
                        clustering_input['cca_y_transformed'] = transformed_data[1]
                        # Add concatenated version for convenience
                        clustering_input['cca_xy_combined'] = np.concatenate(transformed_data, axis=1)
                    else:
                        clustering_input['cca_transformed'] = transformed_data

        # Add raw features if available (from pass-through)
        if raw_features:
            for feature_name, data_array in raw_features.items():
                clustering_input[f'{feature_name}_raw'] = data_array

        # If no results from either pipeline, fall back to processed features
        if not clustering_input and processed_features:
            print("Warning: No PCA or CCA results available. Using processed features as fallback.")
            for feature_name, data_df in processed_features.items():
                if isinstance(data_df, pd.DataFrame):
                    data_array = data_df.values
                else:
                    data_array = data_df
                clustering_input[f'{feature_name}_fallback'] = data_array

        if not clustering_input:
            raise ValueError("No suitable data found for clustering input from either PCA or CCA pipelines.")

        # Build nested structure for ClusteringModel compatibility
        # ClusteringModel.run() and GUI presenter expect 'feature_data', 'feature_names', 'feature_types'
        feature_data_list = []
        feature_names_list = []
        feature_types_list = []

        for key, array in clustering_input.items():
            # Only include 2D numeric arrays (exclude object dtype with strings)
            if isinstance(array, np.ndarray) and array.ndim == 2 and np.issubdtype(array.dtype, np.number):
                feature_data_list.append(array)
                feature_names_list.append(key)
                # Infer type from key suffix
                if '_pca' in key:
                    feature_types_list.append('pca')
                elif '_cca' in key or 'cca_' in key:
                    feature_types_list.append('cca')
                elif '_processed' in key:
                    feature_types_list.append('processed')
                elif '_raw' in key:
                    feature_types_list.append('raw')
                else:
                    feature_types_list.append('unknown')

        # Add nested structure (for ClusteringModel.run() and GUI)
        # Create object array explicitly to avoid numpy shape broadcasting issues
        feature_data_arr = np.empty(len(feature_data_list), dtype=object)
        for i, arr in enumerate(feature_data_list):
            feature_data_arr[i] = arr
        clustering_input['feature_data'] = feature_data_arr
        clustering_input['feature_names'] = np.array(feature_names_list)
        clustering_input['feature_types'] = np.array(feature_types_list)

        return clustering_input

    # Note: PCA processing is now handled by FeatureProcessor

    def _run_cca(self, feature_data: Dict[str, pd.DataFrame],
                 settings: CCASettings, progress_callback=None,
                 preprocessing_output_dir: Optional[str] = None,
                 feature_processing_configs=None) -> Dict:
        """Run Canonical Correlation Analysis using unified CCA implementation"""

        # Always use the unified ml_functions.cca implementation
        from ..ml_functions import run_cca
        print("Using unified CCA implementation...")

        try:
            cca_results = run_cca(
                feature_data=feature_data,
                settings=settings,
                progress_callback=progress_callback,
                preprocessing_output_dir=preprocessing_output_dir,
                feature_processing_configs=feature_processing_configs
            )

            # Convert results to legacy format for compatibility
            legacy_results = self._convert_enhanced_to_legacy_results(cca_results)
            return legacy_results

        except Exception as e:
            print(f"Error in unified CCA: {e}")
    def _create_standardized_results(self, pca_results: Dict, cca_results: Dict,
                                   config: 'FusionConfig', processing_info: Dict) -> Dict:
        """Create standardized results format"""
        standardized_results = {
            "metadata": {
                "config_name": config.name,
                "timestamp": datetime.now().isoformat(),
                "processing_info": processing_info,
                "data_info": {
                    "n_samples": processing_info.get("n_samples", 0),
                    "feature_vectors": list(processing_info.get("feature_names", [])),
                    "pca_applied": bool(pca_results),
                    "cca_applied": bool(cca_results)
                }
            },
            "results": {
                "pca": self._standardize_pca_results(pca_results) if pca_results else None,
                "cca": self._standardize_cca_results(cca_results) if cca_results else None
            },
            "clustering_input": self._create_standardized_clustering_input(
                pca_results, cca_results, config, processing_info
            )
        }
        # Mirror feature names into metadata.feature_info for UI ergonomics
        try:
            cca_std = standardized_results["results"].get("cca") or {}
            x_vec = getattr(config.cca_settings, "x_main_feature_vector_name", None)
            y_vec = getattr(config.cca_settings, "y_main_feature_vector_name", None)
            standardized_results["metadata"]["feature_info"] = {
                "x_vector_name": x_vec,
                "y_vector_name": y_vec,
                "x_feature_names": cca_std.get("x_feature_names", []),
                "y_feature_names": cca_std.get("y_feature_names", []),
            }
        except Exception as e:
            print(f"Warning: failed to populate feature_info metadata: {e}")

        return standardized_results

    def _standardize_pca_results(self, pca_results: Dict) -> Dict:
        """Standardize PCA results format"""
        standardized: Dict[str, Dict] = {}
        for feature_name, result in (pca_results or {}).items():
            transformed = result.get("transformed_data")
            shape = getattr(transformed, "shape", None)
            standardized[feature_name] = {
                "n_components": result.get("n_components", 0),
                "explained_variance_ratio": result.get("explained_variance_ratio", []),
                "cumulative_variance": result.get("cumulative_variance", []),
                "transformed_data_shape": shape,
                "feature_names": result.get("feature_names", [])
            }
        return standardized


    def _standardize_cca_results(self, cca_results: Dict) -> Dict:
        """Standardize CCA results format"""
        if not cca_results:
            return {}
        transformed = cca_results.get("transformed_data") or [None, None]
        x_shape = getattr(transformed[0], "shape", None) if isinstance(transformed, list) and len(transformed) > 0 else None
        y_shape = getattr(transformed[1], "shape", None) if isinstance(transformed, list) and len(transformed) > 1 else None
        return {
            "n_components": len(cca_results.get("correlations", [])),
            "canonical_correlations": cca_results.get("correlations", []),
            "x_feature_names": cca_results.get("x_feature_names", []),
            "y_feature_names": cca_results.get("y_feature_names", []),
            "transformed_data_shapes": {
                "x_transformed": x_shape,
                "y_transformed": y_shape,
            },
            "statistical_results": cca_results.get("statistical_results"),
            "optimization_results": cca_results.get("optimization_results"),
            "processing_info": cca_results.get("processing_info", {}),
        }

    def _create_standardized_clustering_input(self, pca_results: Dict, cca_results: Dict,
                                            config: 'FusionConfig', processing_info: Dict) -> Dict:
        """Create standardized clustering input with clear metadata"""
        clustering_input: Dict[str, Dict] = {
            "data_sources": {},
            "metadata": {
                "source_descriptions": {},
                "recommended_sources": [],
                "data_shapes": {},
                "processing_applied": {},
            },
        }
        # PCA sources
        if pca_results:
            clustering_input["data_sources"]["pca"] = {}
            for feature_name, result in pca_results.items():
                if "transformed_data" in result:
                    key = f"{feature_name}_pca"
                    clustering_input["data_sources"]["pca"][key] = result["transformed_data"]
                    clustering_input["metadata"]["source_descriptions"][key] = f"PCA-transformed {feature_name}"
                    clustering_input["metadata"]["data_shapes"][key] = result["transformed_data"].shape
                    clustering_input["metadata"]["processing_applied"][key] = "PCA dimensionality reduction"
        # CCA sources
        if cca_results and not config.cca_settings.skip_cca:
            clustering_input["data_sources"]["cca"] = {}
            transformed_data = cca_results.get("transformed_data")
            if config.cca_settings.is_multiview:
                if isinstance(transformed_data, list):
                    for i, view_data in enumerate(transformed_data):
                        key = f"cca_view_{i}"
                        clustering_input["data_sources"]["cca"][key] = view_data
                        clustering_input["metadata"]["source_descriptions"][key] = f"CCA canonical variate view {i}"
                        clustering_input["metadata"]["data_shapes"][key] = view_data.shape
                    if len(transformed_data) > 1:
                        combined = np.concatenate(transformed_data, axis=1)
                        key = "cca_multiview_combined"
                        clustering_input["data_sources"]["cca"][key] = combined
                        clustering_input["metadata"]["source_descriptions"][key] = "Combined CCA multiview canonical variates"
                        clustering_input["metadata"]["data_shapes"][key] = combined.shape
                        clustering_input["metadata"]["recommended_sources"].append(key)
            else:
                if isinstance(transformed_data, list) and len(transformed_data) == 2:
                    for variate, name in zip(transformed_data, ["x", "y"]):
                        key = f"cca_{name}_transformed"
                        clustering_input["data_sources"]["cca"][key] = variate
                        clustering_input["metadata"]["source_descriptions"][key] = f"CCA {name.upper()} canonical variates"
                        clustering_input["metadata"]["data_shapes"][key] = variate.shape
                    combined = np.concatenate(transformed_data, axis=1)
                    key = "cca_xy_combined"
                    clustering_input["data_sources"]["cca"][key] = combined
                    clustering_input["metadata"]["source_descriptions"][key] = "Combined CCA X+Y canonical variates"
                    clustering_input["metadata"]["data_shapes"][key] = combined.shape
                    clustering_input["metadata"]["recommended_sources"].append(key)
            clustering_input["metadata"]["processing_applied"]["cca"] = f"CCA ({config.cca_settings.model_settings.cca_type})"
        # Processed fallback
        if processing_info.get("processed_features"):
            clustering_input["data_sources"]["processed"] = {}
            for feature_name, data in processing_info["processed_features"].items():
                key = f"{feature_name}_processed"
                clustering_input["data_sources"]["processed"][key] = data
                clustering_input["metadata"]["source_descriptions"][key] = f"Processed {feature_name} (fallback)"
                clustering_input["metadata"]["data_shapes"][key] = data.shape
        return clustering_input

    def _run_multiview_cca(self, feature_data: Dict[str, pd.DataFrame],
                          settings: CCASettings, progress_callback=None) -> Dict:
        """Handle multiview CCA using the unified implementation"""

        print("Using unified multiview CCA implementation...")

        if not settings.views or not isinstance(settings.views, list):
            raise ValueError("Valid 'views' must be provided for multiview CCA in settings.views")

        # Extract views from feature_data
        views_data = []

        # Handle case where views are dataset keys vs column lists
        if isinstance(settings.views[0], str):
            # Views are dataset keys
            for view_key in settings.views:
                view_data = feature_data.get(view_key)
                if view_data is None:
                    raise ValueError(f"View key '{view_key}' not found in feature_data")
                views_data.append(view_data)
        else:
            # Views are lists of column names - need a source DataFrame
            if len(feature_data) != 1:
                raise ValueError("For column-based views, feature_data must contain exactly one DataFrame")

            main_df = list(feature_data.values())[0]
            for col_list in settings.views:
                view_data = main_df[col_list]
                views_data.append(view_data)

        # For multiview, we need to pass the first two views as x_data and y_data
        # and handle the rest through the multiview logic in the unified function
        if len(views_data) < 2:
            raise ValueError("Multiview CCA requires at least 2 views")

        # Use the unified CCA function with multiview settings
        try:
            results = run_cca(
                x_data=views_data[0],
                y_data=views_data[1],  # This will be handled by multiview logic
                settings=settings,
                progress_callback=progress_callback
            )
            return results

        except Exception as e:
            print(f"Error in multiview CCA: {e}")

    def _convert_enhanced_to_legacy_results(self, enhanced_results: Dict) -> Dict:
        """Convert enhanced CCA results to legacy format for compatibility"""
        cca_results = enhanced_results.get('cca_results', {})

        legacy_results = {
            'correlations': cca_results.get('correlations', []),
            'transformed_data': cca_results.get('transformed_data', []),
            'weights': cca_results.get('weights'),
            'structural_loadings': cca_results.get('structural_loadings'),  # Add structural loadings
            'x_feature_names': cca_results.get('x_feature_names', []),
            'y_feature_names': cca_results.get('y_feature_names', []),
            'statistical_results': enhanced_results.get('statistical_results'),
            'optimization_results': enhanced_results.get('optimization_results'),
            'processing_info': enhanced_results.get('processing_info', {})
        }

        return legacy_results

    def _save_pca_components_to_csv(self, results: Dict, output_dir: str, config: 'FusionConfig'):
        """
        Save PCA-transformed components to CSV files in a PCA subfolder.

        Args:
            results: Results dictionary containing CCA data
            output_dir: Directory to save PCA results
            config: Fusion configuration
        """
        try:
            # Create PCA subdirectory
            pca_dir = os.path.join(output_dir, "PCA")
            os.makedirs(pca_dir, exist_ok=True)

            # Extract CCA results
            cca_data = results.get('cca', {})
            if isinstance(cca_data, dict):
                # Get transformed data (PCA-transformed features)
                transformed_data = cca_data.get('transformed_data', None)

                if transformed_data is not None and isinstance(transformed_data, list) and len(transformed_data) >= 2:
                    # Get feature names from config
                    x_feature_name = config.cca_settings.x_main_feature_vector_name or "X"
                    y_feature_name = config.cca_settings.y_main_feature_vector_name or "Y"

                    # Save X (MSE) PCA components
                    x_pca_data = transformed_data[0]
                    if x_pca_data is not None and len(x_pca_data) > 0:
                        n_components = x_pca_data.shape[1]
                        x_columns = [f"PC{i+1}" for i in range(n_components)]
                        x_df = pd.DataFrame(x_pca_data, columns=x_columns)
                        x_csv_path = os.path.join(pca_dir, f"{x_feature_name}_pca_components.csv")
                        x_df.to_csv(x_csv_path, index=False)
                        print(f"Saved {x_feature_name} PCA components to: {x_csv_path}")

                    # Save Y (QIDS) PCA components
                    y_pca_data = transformed_data[1]
                    if y_pca_data is not None and len(y_pca_data) > 0:
                        n_components = y_pca_data.shape[1]
                        y_columns = [f"PC{i+1}" for i in range(n_components)]
                        y_df = pd.DataFrame(y_pca_data, columns=y_columns)
                        y_csv_path = os.path.join(pca_dir, f"{y_feature_name}_pca_components.csv")
                        y_df.to_csv(y_csv_path, index=False)
                        print(f"Saved {y_feature_name} PCA components to: {y_csv_path}")
                else:
                    print("No PCA-transformed data found in results")
            else:
                print("CCA data not found in results")

        except Exception as e:
            print(f"Warning: Failed to save PCA components to CSV: {e}")
            import traceback
            traceback.print_exc()

    def _save_cca_results_to_csv(self, results: Dict, output_dir: str, config: 'FusionConfig'):
        """
        Save CCA results to CSV files in a cca_steps subfolder.

        Saves:
        - Canonical correlations
        - CCA weights (transformation coefficients in PCA space)
        - Structural loadings (correlations with original features)
        - CCA variates (transformed data)

        Args:
            results: Results dictionary containing CCA data
            output_dir: Directory to save CCA results
            config: Fusion configuration
        """
        try:
            # Create cca_steps subdirectory
            cca_dir = os.path.join(output_dir, "cca_steps")
            os.makedirs(cca_dir, exist_ok=True)

            # Extract CCA results
            cca_data = results.get('cca', {})
            if not isinstance(cca_data, dict):
                print("CCA data not found in results")
                return

            # Get feature names from config
            x_feature_name = config.cca_settings.x_main_feature_vector_name or "X"
            y_feature_name = config.cca_settings.y_main_feature_vector_name or "Y"

            # 1. Save canonical correlations
            correlations = cca_data.get('correlations', None)
            if correlations is not None:
                corr_df = pd.DataFrame({
                    'Component': [f"CC{i+1}" for i in range(len(correlations))],
                    'Correlation': correlations
                })
                corr_path = os.path.join(cca_dir, "canonical_correlations.csv")
                corr_df.to_csv(corr_path, index=False)
                print(f"Saved canonical correlations to: {corr_path}")

            # 2. Save CCA weights (transformation coefficients in PCA space)
            weights = cca_data.get('weights', None)
            if weights is not None:
                if isinstance(weights, list) and len(weights) >= 2:
                    # X weights
                    x_weights = weights[0]
                    if x_weights is not None and len(x_weights) > 0:
                        x_feature_names = cca_data.get('x_feature_names',
                                                       [f"PC{i+1}" for i in range(x_weights.shape[0])])
                        x_weights_df = pd.DataFrame(
                            x_weights,
                            index=x_feature_names,
                            columns=[f"CC{i+1}" for i in range(x_weights.shape[1])]
                        )
                        x_weights_path = os.path.join(cca_dir, f"{x_feature_name}_cca_weights.csv")
                        x_weights_df.to_csv(x_weights_path)
                        print(f"Saved {x_feature_name} CCA weights to: {x_weights_path}")

                    # Y weights
                    y_weights = weights[1]
                    if y_weights is not None and len(y_weights) > 0:
                        y_feature_names = cca_data.get('y_feature_names',
                                                       [f"PC{i+1}" for i in range(y_weights.shape[0])])
                        y_weights_df = pd.DataFrame(
                            y_weights,
                            index=y_feature_names,
                            columns=[f"CC{i+1}" for i in range(y_weights.shape[1])]
                        )
                        y_weights_path = os.path.join(cca_dir, f"{y_feature_name}_cca_weights.csv")
                        y_weights_df.to_csv(y_weights_path)
                        print(f"Saved {y_feature_name} CCA weights to: {y_weights_path}")

            # 3. Save structural loadings (correlations with original features)
            structural_loadings = cca_data.get('structural_loadings', None)
            if structural_loadings is not None and isinstance(structural_loadings, dict):
                # X structural loadings
                x_loadings = structural_loadings.get('x_loadings', None)
                if x_loadings is not None and len(x_loadings) > 0:
                    x_orig_features = structural_loadings.get('x_feature_names',
                                                             [f"Feature_{i+1}" for i in range(x_loadings.shape[0])])
                    x_loadings_df = pd.DataFrame(
                        x_loadings,
                        index=x_orig_features,
                        columns=[f"CC{i+1}" for i in range(x_loadings.shape[1])]
                    )
                    x_loadings_path = os.path.join(cca_dir, f"{x_feature_name}_structural_loadings.csv")
                    x_loadings_df.to_csv(x_loadings_path)
                    print(f"Saved {x_feature_name} structural loadings to: {x_loadings_path}")

                    # Also save p-values if available
                    x_pvalues = structural_loadings.get('x_pvalues', None)
                    if x_pvalues is not None and len(x_pvalues) > 0:
                        x_pvalues_df = pd.DataFrame(
                            x_pvalues,
                            index=x_orig_features,
                            columns=[f"CC{i+1}" for i in range(x_pvalues.shape[1])]
                        )
                        x_pvalues_path = os.path.join(cca_dir, f"{x_feature_name}_structural_loadings_pvalues.csv")
                        x_pvalues_df.to_csv(x_pvalues_path)
                        print(f"Saved {x_feature_name} structural loadings p-values to: {x_pvalues_path}")

                # Y structural loadings
                y_loadings = structural_loadings.get('y_loadings', None)
                if y_loadings is not None and len(y_loadings) > 0:
                    y_orig_features = structural_loadings.get('y_feature_names',
                                                             [f"Feature_{i+1}" for i in range(y_loadings.shape[0])])
                    y_loadings_df = pd.DataFrame(
                        y_loadings,
                        index=y_orig_features,
                        columns=[f"CC{i+1}" for i in range(y_loadings.shape[1])]
                    )
                    y_loadings_path = os.path.join(cca_dir, f"{y_feature_name}_structural_loadings.csv")
                    y_loadings_df.to_csv(y_loadings_path)
                    print(f"Saved {y_feature_name} structural loadings to: {y_loadings_path}")

                    # Also save p-values if available
                    y_pvalues = structural_loadings.get('y_pvalues', None)
                    if y_pvalues is not None and len(y_pvalues) > 0:
                        y_pvalues_df = pd.DataFrame(
                            y_pvalues,
                            index=y_orig_features,
                            columns=[f"CC{i+1}" for i in range(y_pvalues.shape[1])]
                        )
                        y_pvalues_path = os.path.join(cca_dir, f"{y_feature_name}_structural_loadings_pvalues.csv")
                        y_pvalues_df.to_csv(y_pvalues_path)
                        print(f"Saved {y_feature_name} structural loadings p-values to: {y_pvalues_path}")

            # 4. Save CCA variates (transformed data in sample space)
            transformed_data = cca_data.get('transformed_data', None)
            if transformed_data is not None and isinstance(transformed_data, list) and len(transformed_data) >= 2:
                # X variates
                x_variates = transformed_data[0]
                if x_variates is not None and len(x_variates) > 0:
                    x_variates_df = pd.DataFrame(
                        x_variates,
                        columns=[f"CC{i+1}" for i in range(x_variates.shape[1])]
                    )
                    x_variates_path = os.path.join(cca_dir, f"{x_feature_name}_cca_variates.csv")
                    x_variates_df.to_csv(x_variates_path, index=False)
                    print(f"Saved {x_feature_name} CCA variates to: {x_variates_path}")

                # Y variates
                y_variates = transformed_data[1]
                if y_variates is not None and len(y_variates) > 0:
                    y_variates_df = pd.DataFrame(
                        y_variates,
                        columns=[f"CC{i+1}" for i in range(y_variates.shape[1])]
                    )
                    y_variates_path = os.path.join(cca_dir, f"{y_feature_name}_cca_variates.csv")
                    y_variates_df.to_csv(y_variates_path, index=False)
                    print(f"Saved {y_feature_name} CCA variates to: {y_variates_path}")

            print(f"\n✅ All CCA results saved to: {cca_dir}")

        except Exception as e:
            print(f"Warning: Failed to save CCA results to CSV: {e}")
            import traceback
            traceback.print_exc()

    def load_results(self, results_dir: str) -> Dict:
        """
        Load results from a previous fusion run

        Args:
            results_dir: Directory containing the results

        Returns:
            Dictionary of loaded results
        """
        results_path = os.path.join(results_dir, "results.npz")
        if not os.path.exists(results_path):
            raise FileNotFoundError(f"Results not found: {results_path}")

        loaded = np.load(results_path, allow_pickle=True)
        return dict(loaded)

    def print_results(self, results: Dict, view_type: str = ""):
        """Print fusion results in a formatted way."""
        if view_type:
            print(f"\n{'='*60}")
            print(f"{view_type.upper()} RESULTS")
            print(f"{'='*60}")

        # Print metrics
        if "metrics" in results:
            print(f"\n Metrics:")
            metrics = results["metrics"]
            # Handle numpy array that was loaded from file
            if isinstance(metrics, np.ndarray):
                try:
                    metrics = metrics.item()  # Convert single-element array to dict
                except (ValueError, AttributeError):
                    print("  Metrics data format not recognized")
                    metrics = {}

            if hasattr(metrics, 'items'):
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        print(f"  {metric}: {value:.3f}")
                    else:
                        print(f"  {metric}: {value}")
            else:
                print(f"  Metrics: {metrics}")

        # Print PCA results
        if "pca" in results:
            pca_data = results["pca"]
            if isinstance(pca_data, np.ndarray):
                pca_data = pca_data.item()
            print(f"\n{view_type} PCA Results:")
            if hasattr(pca_data, 'items'):
                for feature_name, feature_results in pca_data.items():
                    print(f"\nFeature Vector: {feature_name}")
                    if "explained_variance_ratio" in feature_results:
                        evr = feature_results["explained_variance_ratio"]
                        if isinstance(evr, np.ndarray):
                            evr = evr.item()
                        if isinstance(evr, (float, np.float64)):
                            print(f"Total explained variance: {float(evr):.3f}")
                            print("Number of components: 1")
                        else:
                            print(f"Total explained variance: {float(np.sum(evr)):.3f}")
                            print(f"Number of components: {len(evr)}")

        # Print CCA results
        if "cca" in results:
            cca_data = results["cca"]
            if isinstance(cca_data, np.ndarray):
                cca_data = cca_data.item()
            print(f"\n{view_type} CCA Results:")
            if "correlations" in cca_data:
                correlations = cca_data["correlations"]
                if isinstance(correlations, np.ndarray):
                    # Don't use .item() for multi-element arrays
                    if correlations.size == 1:
                        correlations = correlations.item()
                    # else keep as array for multi-component case

                if isinstance(correlations, (float, np.float64)):
                    print(f"Canonical correlation: {float(correlations):.3f}")
                else:
                    print(f"Canonical correlations: {[f'{float(x):.3f}' for x in correlations]}")

            # Print statistical testing results if available
            if "statistical_results" in cca_data and cca_data["statistical_results"]:
                self._print_statistical_results(cca_data["statistical_results"])

    def _print_statistical_results(self, stats_results: Dict):
        """Print statistical testing results in a formatted way."""
        print(f"\n Statistical Testing Results:")
        print(f"{'='*40}")

        if 'p_values' in stats_results:
            p_vals = stats_results['p_values']

            # Print individual component results
            if 'individual' in p_vals:
                individual = p_vals['individual']
                print(f"\n Individual Component Tests:")
                print(f"  Correction method: {p_vals.get('correction_method', 'none')}")
                print(f"  Alpha level: {p_vals.get('alpha', 0.05)}")

                for i, (p_uncorr, p_corr, sig) in enumerate(zip(
                    individual['p_uncorrected'],
                    individual['p_corrected'],
                    individual['significant'])):
                    status = "***" if sig else ""
                    print(f"  Component {i+1}: p={p_uncorr:.4f}, p_corr={p_corr:.4f} {status}")

            # Print aggregate test results
            if 'aggregate' in p_vals:
                aggregate = p_vals['aggregate']
                status = "***" if aggregate['significant'] else ""
                print(f"\n Aggregate Test:")
                print(f"  Log-likelihood test: p={aggregate['p_value']:.4f} {status}")

        if 'n_permutations' in stats_results:
            print(f"\n Permutations performed: {stats_results['n_permutations']}")

        print(f"*** = Significant at α = {stats_results.get('p_values', {}).get('alpha', 0.05)}")
        print(f"{'='*40}")