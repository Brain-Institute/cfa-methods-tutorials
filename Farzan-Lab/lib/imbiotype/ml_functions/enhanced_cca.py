"""
Enhanced CCA implementation with full hierarchical settings support
"""

import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from sklearn.preprocessing import StandardScaler, RobustScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, VarianceThreshold
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import make_scorer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
import warnings
from scipy import stats
import logging
import statsmodels.formula.api as smf

# Import ValidationLogger for MATLAB comparison
try:
    from .validation_logger import ValidationLogger
    VALIDATION_LOGGER_AVAILABLE = True
except ImportError:
    VALIDATION_LOGGER_AVAILABLE = False
    ValidationLogger = None

# Import CCA-Zoo models
try:
    from cca_zoo.linear import CCA, rCCA, SPLS, SCCA_IPLS, ElasticCCA, PartialCCA
    from cca_zoo.nonparametric import KCCA
    # Don't import deep models if torch is not available
    try:
        from cca_zoo.deep import DCCA
        DEEP_MODELS_AVAILABLE = True
    except ImportError:
        DEEP_MODELS_AVAILABLE = False
        DCCA = None
    from cca_zoo.model_selection import GridSearchCV as CCAGridSearchCV, permutation_test_score
    CCA_ZOO_AVAILABLE = True

    # CRITICAL FIX: Add missing _get_tags method to CCA-Zoo classes
    # This fixes the AttributeError: 'CCA' object has no attribute '_get_tags'
    def _get_tags_compatibility_fix(self):
        """
        Compatibility fix for CCA-Zoo 2.6.0 missing _get_tags method.

        The CCA-Zoo base class _validate_data method expects _get_tags() to return
        a dictionary-like object, but CCA-Zoo 2.6.0 uses sklearn's new Tags system.
        This method provides backward compatibility.
        """
        # Get the sklearn tags object
        tags_obj = self.__sklearn_tags__()

        # Convert to dictionary-like interface for backward compatibility
        tags_dict = {
            'multiview': True,  # CCA-Zoo models are multiview by default
            'requires_fit': getattr(tags_obj, 'requires_fit', True),
            'no_validation': getattr(tags_obj, 'no_validation', False),
        }

        # Return a dict-like object that supports .get() method
        return tags_dict

    # Apply the fix to all CCA-Zoo classes that might be used
    cca_classes_to_fix = [CCA, rCCA, SPLS, SCCA_IPLS, ElasticCCA, PartialCCA, KCCA]
    if DEEP_MODELS_AVAILABLE and DCCA:
        cca_classes_to_fix.append(DCCA)

    for cca_class in cca_classes_to_fix:
        if not hasattr(cca_class, '_get_tags'):
            cca_class._get_tags = _get_tags_compatibility_fix

except ImportError:
    print("Warning: CCA-Zoo not available. Some advanced CCA features will be unavailable.")
    CCA_ZOO_AVAILABLE = False
    DEEP_MODELS_AVAILABLE = False

# Import MATLAB adaptation modules if available
try:
    from ...models.fusion import InputPreprocessingSettings, ModelSettings, OptimizationSettings, StatisticalTestingSettings, CCASettings
except ImportError:
    # Handle relative import issues during development
    pass

# Import new feature processing architecture
try:
    # Try multiple import paths to handle different installation scenarios
    try:
        # Relative import (development mode)
        from ...processing.feature_processor import (
            PreprocessingConfig, ProcessingPlan, build_processing_plan,
            apply_outside_cv_steps, apply_inside_cv_steps,
            TransformStep, VarianceFilterStep, ResidualizeStep, StandardizeStep,
            UnivariateSelectStep, RedundancyPruneStep, PCAStep
        )
    except ImportError:
        try:
            # Absolute import (development mode)
            from lib.imbiotype.processing.feature_processor import (
                PreprocessingConfig, ProcessingPlan, build_processing_plan,
                apply_outside_cv_steps, apply_inside_cv_steps,
                TransformStep, VarianceFilterStep, ResidualizeStep, StandardizeStep,
                UnivariateSelectStep, RedundancyPruneStep, PCAStep
            )
        except ImportError:
            # Package import (installed mode)
            from imbiotype.processing.feature_processor import (
                PreprocessingConfig, ProcessingPlan, build_processing_plan,
                apply_outside_cv_steps, apply_inside_cv_steps,
                TransformStep, VarianceFilterStep, ResidualizeStep, StandardizeStep,
                UnivariateSelectStep, RedundancyPruneStep, PCAStep
            )
    NEW_FEATURE_PROCESSOR_AVAILABLE = True
except ImportError:
    NEW_FEATURE_PROCESSOR_AVAILABLE = False
    # Create dummy classes for type hints
    class PreprocessingConfig: pass
    class ProcessingPlan: pass
    def build_processing_plan(*args, **kwargs): return ProcessingPlan()
    def apply_outside_cv_steps(*args, **kwargs): return None
    def apply_inside_cv_steps(*args, **kwargs): return None
    print("Warning: New feature processor not available. Using legacy preprocessing.")

logger = logging.getLogger(__name__)


def convert_legacy_settings_to_preprocessing_config(feature_processing_configs) -> Dict[str, Any]:
    """
    Convert legacy FeatureProcessingSettings to new PreprocessingConfig format.

    Args:
        feature_processing_configs: List of FeatureProcessingConfig objects

    Returns:
        Dictionary mapping feature names to PreprocessingConfig objects
    """
    if not NEW_FEATURE_PROCESSOR_AVAILABLE:
        return {}

    configs = {}

    for config in feature_processing_configs:
        # Check if preprocessing_config is already provided (new unified architecture)
        if hasattr(config, 'preprocessing_config') and config.preprocessing_config is not None:
            # Use the provided preprocessing config directly
            configs[config.feature_name] = config.preprocessing_config
            continue

        # Otherwise, convert from legacy processing_settings (if available)
        if config.processing_settings is None:
            # No settings provided - create empty config
            configs[config.feature_name] = PreprocessingConfig()
            continue

        settings = config.processing_settings

        # Build preprocessing config from legacy settings
        preprocessing_config = PreprocessingConfig()

        # Transform
        if hasattr(settings, 'transform_method') and settings.transform_method != "none":
            preprocessing_config.transform = {"kind": settings.transform_method}

        # Variance filtering (basic implementation)
        if hasattr(settings, 'feature_selection_enabled') and settings.feature_selection_enabled:
            if hasattr(settings, 'feature_selection_method') and settings.feature_selection_method == "variance":
                threshold = getattr(settings, 'variance_threshold', 0.01)
                preprocessing_config.variance_filter_pre = {"threshold": threshold}

        # Standardization
        if hasattr(settings, 'standardize') and settings.standardize:
            method = getattr(settings, 'standardization_method', 'zscore')
            if method == 'zscore':
                preprocessing_config.standardize = {"with_mean": True, "with_std": True}
            elif method == 'robust':
                # For now, map to standard scaling - could be enhanced later
                preprocessing_config.standardize = {"with_mean": True, "with_std": True}

        # PCA
        if hasattr(settings, 'pca_enabled') and settings.pca_enabled:
            if hasattr(settings, 'pca_method'):
                if settings.pca_method == "explained_variance":
                    n_components = getattr(settings, 'pca_explained_variance', 0.95)
                elif settings.pca_method == "num_components":
                    n_components = getattr(settings, 'pca_num_components', 3)
                else:
                    n_components = 0.95  # default

                preprocessing_config.pca = {
                    "n_components": n_components,
                    "whiten": False
                }

        configs[config.feature_name] = preprocessing_config

    return configs


class MATLABStyleRCCA:
    """
    MATLAB-style regularized CCA implementation that exactly matches rcc_matlab.m

    This implementation applies regularization by adding lambda to the diagonal
    of covariance matrices, exactly as MATLAB's rcc_matlab function does.
    """

    def __init__(self, n_components=2, regularization_x=0.1, regularization_y=0.1):
        self.n_components = n_components
        self.regularization_x = regularization_x
        self.regularization_y = regularization_y

    def fit(self, views):
        """Fit the MATLAB-style rCCA model."""
        X, Y = views[0], views[1]

        # Convert to numpy arrays and ensure they're 2D
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        n_samples_x, n_features_x = X.shape
        n_samples_y, n_features_y = Y.shape

        if n_samples_x != n_samples_y:
            raise ValueError(f"X and Y must have same number of samples: {n_samples_x} vs {n_samples_y}")

        # Center the data
        X_centered = X - np.mean(X, axis=0)
        Y_centered = Y - np.mean(Y, axis=0)

        # Covariance matrices with L2-norm penalties (MATLAB style)
        # Cxx = cov(X) + diag(lambda1*ones(ncolX,1))
        # Cyy = cov(Y) + diag(lambda2*ones(ncolY,1))
        Cxx = np.cov(X_centered.T) + np.diag(self.regularization_x * np.ones(n_features_x))
        Cyy = np.cov(Y_centered.T) + np.diag(self.regularization_y * np.ones(n_features_y))

        # Cross-covariance matrix
        # Cxy = ((X-mean(X))'*(Y-mean(Y)))/(nrowX-1)
        Cxy = (X_centered.T @ Y_centered) / (n_samples_x - 1)

        # Make covariance matrices symmetric (MATLAB style)
        Cxx = (Cxx + Cxx.T) / 2
        Cyy = (Cyy + Cyy.T) / 2

        # Cholesky factorization
        try:
            Cxx_chol = np.linalg.cholesky(Cxx)
            Cyy_chol = np.linalg.cholesky(Cyy)
        except np.linalg.LinAlgError:
            # If Cholesky fails, add more regularization
            eps = 1e-6
            Cxx += np.eye(n_features_x) * eps
            Cyy += np.eye(n_features_y) * eps
            Cxx_chol = np.linalg.cholesky(Cxx)
            Cyy_chol = np.linalg.cholesky(Cyy)

        # Matrix inverse of Cholesky factors
        # NumPy cholesky returns lower triangular L where L @ L.T = A
        # MATLAB chol returns upper triangular R where R.T @ R = A
        # So MATLAB's R = L.T, and inv(R) = inv(L.T) = inv(L).T
        Cxx_chol_inv = np.linalg.inv(Cxx_chol)
        Cyy_chol_inv = np.linalg.inv(Cyy_chol)

        # Compute the matrix for SVD
        # MATLAB: covmat = Cxxfacinv' * Cxy * Cyyfacinv = inv(R)' * Cxy * inv(S)
        # Since inv(R)' = inv(L) and inv(S) = inv(L_yy).T:
        covmat = Cxx_chol_inv @ Cxy @ Cyy_chol_inv.T

        # Singular value decomposition
        a, d, b_T = np.linalg.svd(covmat, full_matrices=False)
        b = b_T.T

        # Canonical coefficients
        # MATLAB: A = Cxxfacinv * a = inv(R) * a = inv(L).T * a
        # MATLAB: B = Cyyfacinv * b = inv(S) * b = inv(L_yy).T * b
        A = Cxx_chol_inv.T @ a
        B = Cyy_chol_inv.T @ b

        # Canonical correlations
        r = d

        # Store results
        self.x_weights_ = A[:, :self.n_components]
        self.y_weights_ = B[:, :self.n_components]
        self.correlations_ = r[:self.n_components]
        self.x_mean_ = np.mean(X, axis=0)
        self.y_mean_ = np.mean(Y, axis=0)

        return self

    def transform(self, views):
        """Transform data using fitted model."""
        X, Y = views[0], views[1]

        # Convert to numpy arrays
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)

        # Center using training means
        X_centered = X - self.x_mean_
        Y_centered = Y - self.y_mean_

        # Handle NaN values (MATLAB style)
        X_centered = np.nan_to_num(X_centered, nan=0.0)
        Y_centered = np.nan_to_num(Y_centered, nan=0.0)

        # Canonical variates
        U = X_centered @ self.x_weights_
        V = Y_centered @ self.y_weights_

        return [U, V]


class CovariateResidualizer(BaseEstimator, TransformerMixin):
    """
    Transformer that residualizes features against covariates using OLS.
    Properly handles train/test splits for cross-validation.
    """
    
    def __init__(self, covariate_columns: Optional[List[str]] = None):
        self.covariate_columns = covariate_columns or []
        self.residualization_models_ = {}
        
    def fit(self, X, y=None):
        """
        Fit residualization models on training data.
        
        Args:
            X: DataFrame with both features and covariates
            y: Not used
        """
        if not self.covariate_columns:
            return self
            
        # Get feature columns (everything except covariates)
        feature_columns = [col for col in X.columns if col not in self.covariate_columns]
        
        if not feature_columns:
            return self
            
        # Fit OLS models for each feature against covariates
        for feature_col in feature_columns:
            try:
                # Create formula for statsmodels
                formula_rhs = " + ".join([f"Q('{cov}')" for cov in self.covariate_columns])
                formula = f"Q('{feature_col}') ~ {formula_rhs}"
                
                # Fit OLS model
                model = smf.ols(formula, data=X).fit()
                self.residualization_models_[feature_col] = model
                
            except Exception as e:
                # Residualization failed - this is acceptable when confounds have structural issues
                # Store None to indicate no residualization for this feature
                self.residualization_models_[feature_col] = None
                
        return self
    
    def transform(self, X):
        """
        Apply residualization to data.
        
        Args:
            X: DataFrame with both features and covariates
            
        Returns:
            DataFrame with residualized features (covariates removed)
        """
        if not self.covariate_columns:
            # No covariates, return all columns as features
            return X
            
        # Get feature columns
        feature_columns = [col for col in X.columns if col not in self.covariate_columns]
        
        if not feature_columns:
            raise ValueError("No feature columns found after removing covariates")
            
        residualized_data = pd.DataFrame(index=X.index)
        
        for feature_col in feature_columns:
            model = self.residualization_models_.get(feature_col)

            if model is None:
                # No residualization model available, use original data
                residualized_data[feature_col] = X[feature_col]
            else:
                try:
                    # Calculate residuals using the fitted model
                    predicted = model.predict(X)
                    residual = X[feature_col] - predicted

                    # Guard against NaN/Inf in residuals (e.g., unseen categorical levels)
                    residual = pd.to_numeric(residual, errors='coerce')
                    original = pd.to_numeric(X[feature_col], errors='coerce')
                    # Where residual is not finite, fall back to original values
                    residual_safe = residual.mask(~np.isfinite(residual), other=original)
                    # If still non-finite (both were bad), fill with 0 as a last resort
                    residual_safe = residual_safe.fillna(0.0)
                    residualized_data[feature_col] = residual_safe
                except Exception as e:
                    # Residualization failed - use original feature values
                    residualized_data[feature_col] = pd.to_numeric(X[feature_col], errors='coerce').fillna(0.0)

        return residualized_data

class CCAEstimator(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible CCA estimator that wraps CCA-Zoo models.
    """
    
    def __init__(self,
                 cca_type: str = "classic",
                 n_components: int = 2,
                 regularization_x: float = 0.1,
                 regularization_y: float = 0.1,
                 sparsity_penalty_x: float = 0.0,
                 sparsity_penalty_y: float = 0.0,
                 use_matlab_style: bool = False):
        self.cca_type = cca_type
        self.n_components = n_components
        self.regularization_x = regularization_x
        self.regularization_y = regularization_y
        self.sparsity_penalty_x = sparsity_penalty_x
        self.sparsity_penalty_y = sparsity_penalty_y
        self.use_matlab_style = use_matlab_style
        
    def _create_cca_model(self):
        """Create the appropriate CCA model based on settings."""
        if self.cca_type == "classic":
            if not CCA_ZOO_AVAILABLE:
                raise ImportError("CCA-Zoo is required for classic CCA functionality")
            return CCA(latent_dimensions=self.n_components)
        elif self.cca_type == "regularized":
            if self.use_matlab_style:
                # Use MATLAB-style rCCA implementation for exact MATLAB matching
                # This doesn't require CCA-Zoo
                return MATLABStyleRCCA(
                    n_components=self.n_components,
                    regularization_x=self.regularization_x,
                    regularization_y=self.regularization_y
                )
            else:
                # Use CCA-Zoo's rCCA implementation (requires CCA-Zoo)
                if not CCA_ZOO_AVAILABLE:
                    raise ImportError("CCA-Zoo is required for L2 regularization functionality")
                return rCCA(
                    latent_dimensions=self.n_components,
                    c=[self.regularization_x, self.regularization_y]
                )
        elif self.cca_type == "sparse":
            if not CCA_ZOO_AVAILABLE:
                raise ImportError("CCA-Zoo is required for sparse CCA functionality")
            return SPLS(
                latent_dimensions=self.n_components,
                c=[self.sparsity_penalty_x, self.sparsity_penalty_y]
            )
        elif self.cca_type == "elastic":
            if not CCA_ZOO_AVAILABLE:
                raise ImportError("CCA-Zoo is required for elastic CCA functionality")
            return ElasticCCA(
                latent_dimensions=self.n_components,
                l1_ratio=[self.sparsity_penalty_x, self.sparsity_penalty_y],
                alpha=[self.regularization_x, self.regularization_y]
            )
        elif self.cca_type == "kernel":
            if not CCA_ZOO_AVAILABLE:
                raise ImportError("CCA-Zoo is required for kernel CCA functionality")
            return KCCA(latent_dimensions=self.n_components)
        elif self.cca_type == "partial":
            if not CCA_ZOO_AVAILABLE:
                raise ImportError("CCA-Zoo is required for partial CCA functionality")
            return PartialCCA(latent_dimensions=self.n_components)
        elif self.cca_type == "deep":
            if not DEEP_MODELS_AVAILABLE:
                raise ImportError("Deep CCA models require torch/lightning to be installed")
            return DCCA(latent_dimensions=self.n_components)
        else:
            raise ValueError(f"Unsupported CCA type: {self.cca_type}")
    
    def fit(self, X, y):
        """
        Fit CCA model.
        
        Args:
            X: X data (features) - DataFrame or array
            y: Y data (features) - DataFrame or array
        """
        # Filter out categorical columns if input is DataFrame
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
            X_array = X_numeric.values
        else:
            X_array = X.values if hasattr(X, 'values') else X
            
        if isinstance(y, pd.DataFrame):
            y_numeric = y.select_dtypes(include=[np.number])
            y_array = y_numeric.values
        else:
            y_array = y.values if hasattr(y, 'values') else y
        
        # Final safety check: ensure arrays are finite before passing to cca_zoo
        X_finite = np.isfinite(X_array).all()
        y_finite = np.isfinite(y_array).all()
        if not X_finite or not y_finite:
            print(f"Warning: Non-finite values detected before CCA fit. X_finite={X_finite}, y_finite={y_finite}")
            print(f"X shape: {X_array.shape}, Y shape: {y_array.shape}")
            print(f"X NaN count: {np.isnan(X_array).sum()}, X Inf count: {np.isinf(X_array).sum()}")
            print(f"Y NaN count: {np.isnan(y_array).sum()}, Y Inf count: {np.isinf(y_array).sum()}")
            # Replace non-finite with 0 as last resort
            X_array = np.nan_to_num(X_array, nan=0.0, posinf=0.0, neginf=0.0)
            y_array = np.nan_to_num(y_array, nan=0.0, posinf=0.0, neginf=0.0)
            print("Replaced non-finite values with 0.0")

        # Create and fit CCA model
        self.cca_model_ = self._create_cca_model()
        self.cca_model_.fit([X_array, y_array])

        return self
    
    def transform(self, X, y):
        """Transform data using fitted CCA model."""
        # Filter out categorical columns if input is DataFrame
        if isinstance(X, pd.DataFrame):
            X_numeric = X.select_dtypes(include=[np.number])
            X_array = X_numeric.values
        else:
            X_array = X.values if hasattr(X, 'values') else X
            
        if isinstance(y, pd.DataFrame):
            y_numeric = y.select_dtypes(include=[np.number])
            y_array = y_numeric.values
        else:
            y_array = y.values if hasattr(y, 'values') else y
        
        return self.cca_model_.transform([X_array, y_array])
    
    def score(self, X, y):
        """Return the canonical correlation score."""
        try:
            # Filter out categorical columns if input is DataFrame
            if isinstance(X, pd.DataFrame):
                X_numeric = X.select_dtypes(include=[np.number])
                X_array = X_numeric.values
            else:
                X_array = X.values if hasattr(X, 'values') else X
                
            if isinstance(y, pd.DataFrame):
                y_numeric = y.select_dtypes(include=[np.number])
                y_array = y_numeric.values
            else:
                y_array = y.values if hasattr(y, 'values') else y
            
            # Ensure we have 2D arrays
            if X_array.ndim == 1:
                X_array = X_array.reshape(-1, 1)
            if y_array.ndim == 1:
                y_array = y_array.reshape(-1, 1)
            
            # Transform the data
            transformed = self.cca_model_.transform([X_array, y_array])
            
            # MATLAB-style validation score: use ONLY first component correlation
            # This matches MATLAB's: val_r = corr(val_variate_mse(:,1), val_variate_qids(:,1))

            if transformed[0].shape[1] > 0 and transformed[1].shape[1] > 0:
                # Get first canonical variates (MATLAB uses (:,1))
                x_variate = transformed[0][:, 0]  # First component only
                y_variate = transformed[1][:, 0]  # First component only

                # Check for constant variates (use tolerance for floating-point robustness)
                if np.var(x_variate) < 1e-12 or np.var(y_variate) < 1e-12:
                    return 0.0

                # Calculate correlation with NaN handling (MATLAB-style)
                corr_matrix = np.corrcoef(x_variate, y_variate)
                if corr_matrix.shape == (2, 2):  # Ensure proper matrix shape
                    corr = corr_matrix[0, 1]
                else:
                    return 0.0

                # Handle NaN/inf cases
                if np.isnan(corr) or np.isinf(corr):
                    return 0.0
                else:
                    # Return RAW correlation (not absolute value) to match MATLAB
                    return float(corr)
            else:
                return 0.0
            
        except Exception as e:
            print(f"Warning: Could not calculate CCA score: {e}")
            return 0.0

    def evaluate_test_set(self, X_test, Y_test) -> np.ndarray:
        """
        Evaluate test set using fitted CCA weights.

        Args:
            X_test: Test X data (n_samples x n_features_x)
            Y_test: Test Y data (n_samples x n_features_y)

        Returns:
            Array of canonical correlations on test data (n_components,)
        """
        # Convert to arrays
        if isinstance(X_test, pd.DataFrame):
            X_array = X_test.select_dtypes(include=[np.number]).values
        else:
            X_array = X_test.values if hasattr(X_test, 'values') else X_test

        if isinstance(Y_test, pd.DataFrame):
            Y_array = Y_test.select_dtypes(include=[np.number]).values
        else:
            Y_array = Y_test.values if hasattr(Y_test, 'values') else Y_test

        # Ensure 2D
        if X_array.ndim == 1:
            X_array = X_array.reshape(-1, 1)
        if Y_array.ndim == 1:
            Y_array = Y_array.reshape(-1, 1)

        # Transform test data
        transformed = self.cca_model_.transform([X_array, Y_array])

        # Compute correlation for each component
        n_components = min(transformed[0].shape[1], transformed[1].shape[1])
        test_correlations = np.zeros(n_components)

        for i in range(n_components):
            x_variate = transformed[0][:, i]
            y_variate = transformed[1][:, i]

            # Check for constant variates (use tolerance for floating-point robustness)
            if np.var(x_variate) < 1e-12 or np.var(y_variate) < 1e-12:
                test_correlations[i] = 0.0
                continue

            corr_matrix = np.corrcoef(x_variate, y_variate)
            # Validate shape (should be 2x2 for two variates)
            if corr_matrix.shape != (2, 2):
                test_correlations[i] = 0.0
                continue
            corr = corr_matrix[0, 1]
            if np.isnan(corr) or np.isinf(corr):
                test_correlations[i] = 0.0
            else:
                test_correlations[i] = corr

        return test_correlations

    def get_fitted_model(self) -> dict:
        """
        Return fitted model parameters for later use.

        Returns:
            Dict with weights, means, and other model parameters
        """
        model_info = {
            'cca_type': self.cca_type,
            'n_components': self.n_components,
            'regularization_x': self.regularization_x,
            'regularization_y': self.regularization_y
        }

        if hasattr(self, 'cca_model_'):
            # For MATLABStyleRCCA
            if hasattr(self.cca_model_, 'x_weights_'):
                model_info['x_weights'] = self.cca_model_.x_weights_
                model_info['y_weights'] = self.cca_model_.y_weights_
            if hasattr(self.cca_model_, 'x_mean_'):
                model_info['x_mean'] = self.cca_model_.x_mean_
                model_info['y_mean'] = self.cca_model_.y_mean_
            if hasattr(self.cca_model_, 'correlations_'):
                model_info['train_correlations'] = self.cca_model_.correlations_

            # For CCA-Zoo models
            if hasattr(self.cca_model_, 'weights'):
                model_info['weights'] = self.cca_model_.weights

        return model_info


class EnhancedCCAPipeline:
    """
    Complete pipeline for enhanced CCA with proper cross-validation handling.
    """
    
    def __init__(self, settings: 'CCASettings', feature_processing_configs=None):
        self.settings = settings
        self.feature_processing_configs = feature_processing_configs or []
        self.preprocessing_steps = []
        self.pipeline_ = None
        self.best_params_ = None
        self.cv_results_ = None

        # Convert legacy settings to new preprocessing configs
        if NEW_FEATURE_PROCESSOR_AVAILABLE and self.feature_processing_configs:
            self.preprocessing_configs = convert_legacy_settings_to_preprocessing_config(self.feature_processing_configs)
        else:
            self.preprocessing_configs = {}

    def _create_processing_plans(self, x_feature_name: str, y_feature_name: str,
                                x_covariate_columns: List[str], y_covariate_columns: List[str]) -> Tuple[ProcessingPlan, ProcessingPlan]:
        """Create processing plans for X and Y views."""
        if not NEW_FEATURE_PROCESSOR_AVAILABLE:
            # Return empty plans if new processor not available
            return ProcessingPlan(), ProcessingPlan()

        # Determine mode based on hyperparameter optimization setting
        mode = "with_hpo" if self.settings.hyperparameter_optimization_enabled else "no_hpo"

        # Get preprocessing configs for X and Y
        x_config = self.preprocessing_configs.get(x_feature_name, PreprocessingConfig())
        y_config = self.preprocessing_configs.get(y_feature_name, PreprocessingConfig())

        # Add residualization for confounds if present (only if not already configured)
        if x_covariate_columns and x_config.residualize is None:
            x_config.residualize = {"confounds": x_covariate_columns, "add_intercept": True}
        if y_covariate_columns and y_config.residualize is None:
            y_config.residualize = {"confounds": y_covariate_columns, "add_intercept": True}

        # Ensure standardization is enabled (required for CCA)
        if x_config.standardize is None:
            x_config.standardize = {"with_mean": True, "with_std": True}
        if y_config.standardize is None:
            y_config.standardize = {"with_mean": True, "with_std": True}

        # Build processing plans
        x_plan = build_processing_plan(x_config, mode)
        y_plan = build_processing_plan(y_config, mode)

        return x_plan, y_plan

    def _build_preprocessing_pipeline(self, x_covariate_columns: List[str], y_covariate_columns: List[str]):
        """Build preprocessing pipeline components."""
        steps = []
        
        # Add covariate residualization for X data
        if x_covariate_columns:
            steps.append(('x_residualizer', CovariateResidualizer(x_covariate_columns)))
            
        # Note: Input preprocessing is now handled by FeatureProcessor before CCA
        # Always add standardization for CCA (required for proper correlation analysis)
        steps.append(('x_scaler', StandardScaler()))
        
        return steps
    
    def _create_pipeline(self, x_covariate_columns: List[str] = None, y_covariate_columns: List[str] = None):
        """Create the complete CCA pipeline."""
        # For now, create a simple pipeline with residualization and CCA
        # Note: This is a simplified version - full preprocessing pipeline would be more complex
        
        pipeline_steps = []
        
        # Add CCA estimator
        cca_estimator = CCAEstimator(
            cca_type=self.settings.model_settings.cca_type,
            n_components=self.settings.model_settings.n_components,
            regularization_x=self.settings.model_settings.regularization_x,
            regularization_y=self.settings.model_settings.regularization_y,
            sparsity_penalty_x=self.settings.model_settings.sparsity_penalty_x,
            sparsity_penalty_y=self.settings.model_settings.sparsity_penalty_y,
            use_matlab_style=(self.settings.model_settings.regularization_method == "covariance")
        )
        
        pipeline_steps.append(('cca', cca_estimator))
        
        # Note: For proper implementation, we'd need a custom pipeline that handles
        # X and Y data separately through their respective preprocessing steps
        # This is a simplified version for demonstration
        
        return pipeline_steps
    
    def fit(self,
            x_data: pd.DataFrame,
            y_data: pd.DataFrame,
            x_covariates: Optional[pd.DataFrame] = None,
            y_covariates: Optional[pd.DataFrame] = None,
            progress_callback=None,
            preprocessing_output_dir: Optional[str] = None):
        """
        Fit the enhanced CCA pipeline with new preprocessing architecture.

        IMPORTANT ASSUMPTION:
        - When called from fusion.py, x_data and y_data have ALREADY had outside-CV steps applied
          (transform, variance_filter_pre in with_hpo mode; all steps in no_hpo mode)
        - This method only applies inside-CV steps (residualize, standardize, etc.) per-fold during HPO
        - This prevents double application of outside-CV steps and matches MATLAB's architecture

        Args:
            x_data: X feature data (may already have outside-CV preprocessing applied)
            y_data: Y feature data (may already have outside-CV preprocessing applied)
            x_covariates: Optional X covariates for residualization
            y_covariates: Optional Y covariates for residualization
            progress_callback: Optional callback for progress updates
            preprocessing_output_dir: Optional directory to save intermediate preprocessing CSVs
        """
        # Combine X data with covariates if present
        if x_covariates is not None:
            x_combined = pd.concat([x_data, x_covariates], axis=1)
            x_covariate_columns = x_covariates.columns.tolist()
        else:
            x_combined = x_data
            x_covariate_columns = []

        # Similarly for Y data
        if y_covariates is not None:
            y_combined = pd.concat([y_data, y_covariates], axis=1)
            y_covariate_columns = y_covariates.columns.tolist()
        else:
            y_combined = y_data
            y_covariate_columns = []

        # Store information for later use
        self.x_covariate_columns_ = x_covariate_columns
        self.y_covariate_columns_ = y_covariate_columns

        # STEP 1: Create processing plans using new architecture
        if NEW_FEATURE_PROCESSOR_AVAILABLE:
            # Use new architecture when available
            print("Using new preprocessing architecture...")

            # Get feature names (use defaults if configs are empty)
            if self.preprocessing_configs:
                x_feature_name = list(self.preprocessing_configs.keys())[0]
                y_feature_name = list(self.preprocessing_configs.keys())[1] if len(self.preprocessing_configs) > 1 else "y_features"
            else:
                x_feature_name = "x_features"
                y_feature_name = "y_features"

            print("Creating processing plans with new architecture...")
            x_plan, y_plan = self._create_processing_plans(x_feature_name, y_feature_name,
                                                          x_covariate_columns, y_covariate_columns)

            # SIMPLIFIED APPROACH: Assume outside-CV steps already applied by fusion.py
            # We only need to handle inside-CV steps here (applied per-fold during HPO)

            # For structural loadings, we need data after residualize/standardize but before PCA
            # Create simplified plans that exclude dimensionality reduction steps
            print("Preparing data for structural loadings...")
            x_plan_no_dimred = ProcessingPlan(
                outside_cv_steps=[],  # Already applied by fusion.py
                inside_cv_steps=[s for s in x_plan.inside_cv_steps
                                if s.name not in ['pca', 'univariate', 'redundancy_prune']]
            )
            y_plan_no_dimred = ProcessingPlan(
                outside_cv_steps=[],  # Already applied by fusion.py
                inside_cv_steps=[s for s in y_plan.inside_cv_steps
                                if s.name not in ['pca', 'univariate', 'redundancy_prune']]
            )

            # Extract only feature columns (exclude covariates) for structural loadings
            # This prevents standardize from trying to standardize non-numeric covariate columns
            x_feature_columns_temp = [col for col in x_combined.columns if col not in x_covariate_columns]
            y_feature_columns_temp = [col for col in y_combined.columns if col not in y_covariate_columns]
            x_features_only = x_combined[x_feature_columns_temp] if x_feature_columns_temp else x_combined
            y_features_only = y_combined[y_feature_columns_temp] if y_feature_columns_temp else y_combined

            # Apply only residualize/standardize steps (no dimensionality reduction)
            # Note: Since we removed covariates, residualize will be skipped (no confounds found)
            x_for_structural = apply_inside_cv_steps(x_plan_no_dimred, x_features_only, fit=True, other_view=y_features_only,
                                                     output_dir=preprocessing_output_dir, feature_name=x_feature_name)
            y_for_structural = apply_inside_cv_steps(y_plan_no_dimred, y_features_only, fit=True, other_view=x_features_only,
                                                     output_dir=preprocessing_output_dir, feature_name=y_feature_name)

            # Store data for structural loadings
            self.x_data_before_pca_ = x_for_structural
            self.y_data_before_pca_ = y_for_structural

            # Use the input data directly (outside-CV steps already applied by fusion.py)
            # No need to re-apply outside-CV steps here!
            x_processed = x_combined
            y_processed = y_combined

            print("[Enhanced CCA] Using data with outside-CV steps already applied by fusion.py")
            print(f"[Enhanced CCA] X data shape: {x_processed.shape}, Y data shape: {y_processed.shape}")

            # Store processing plans for later use (for inside-CV steps during HPO)
            self.x_processing_plan_ = x_plan
            self.y_processing_plan_ = y_plan

        else:
            # Fallback to legacy preprocessing
            print("Using legacy preprocessing (new architecture not available)...")
            x_processed, y_processed = self._apply_feature_selection_and_transforms(x_combined, y_combined)

        # Extract feature columns (excluding covariates) for CCA
        x_feature_columns = [col for col in x_processed.columns if col not in x_covariate_columns]
        y_feature_columns = [col for col in y_processed.columns if col not in y_covariate_columns]

        x_features_only = x_processed[x_feature_columns] if x_feature_columns else x_processed
        y_features_only = y_processed[y_feature_columns] if y_feature_columns else y_processed

        # Store feature information for later use
        self.x_feature_columns_ = x_feature_columns
        self.y_feature_columns_ = y_feature_columns

        # Ensure we have features to work with
        if x_features_only.empty:
            print("Warning: No X features available after preprocessing. Using all available columns.")
            x_features_only = x_processed
            self.x_feature_columns_ = x_processed.columns.tolist()
        if y_features_only.empty:
            print("Warning: No Y features available after preprocessing. Using all available columns.")
            y_features_only = y_processed
            self.y_feature_columns_ = y_processed.columns.tolist()

        # CRITICAL: Store original preprocessed data (after residualization/standardization, before PCA)
        # This is needed for computing structural loadings later
        self.x_original_preprocessed_ = x_features_only.copy()
        self.y_original_preprocessed_ = y_features_only.copy()
        
        # Create and store CCA estimator
        self.cca_estimator_ = CCAEstimator(
            cca_type=self.settings.model_settings.cca_type,
            n_components=self.settings.model_settings.n_components,
            regularization_x=self.settings.model_settings.regularization_x,
            regularization_y=self.settings.model_settings.regularization_y,
            sparsity_penalty_x=self.settings.model_settings.sparsity_penalty_x,
            sparsity_penalty_y=self.settings.model_settings.sparsity_penalty_y,
            use_matlab_style=(self.settings.model_settings.regularization_method == "covariance")
        )

        # STEP 2: Handle hyperparameter optimization if enabled
        if self.settings.hyperparameter_optimization_enabled:
            print("Starting hyperparameter optimization...")
            if NEW_FEATURE_PROCESSOR_AVAILABLE and hasattr(self, 'x_processing_plan_'):
                # Use new architecture for HPO
                self._optimize_hyperparameters_new_architecture(
                    x_combined, y_combined, x_features_only, y_features_only,
                    progress_callback, preprocessing_output_dir
                )
            else:
                # Fallback to legacy HPO
                self._optimize_hyperparameters_fixed_features(
                    x_combined, y_combined, x_features_only, y_features_only,
                    x_covariate_columns, y_covariate_columns, progress_callback
                )
        else:
            # STEP 3: Fit the model directly (no HPO)
            if NEW_FEATURE_PROCESSOR_AVAILABLE and hasattr(self, 'x_processing_plan_'):
                # For new architecture, inside-CV steps are already applied in no_hpo mode
                self.cca_estimator_.fit(x_features_only, y_features_only)
            else:
                # Legacy approach with fold-specific preprocessing
                x_final, y_final, _ = self._apply_fold_specific_preprocessing(
                    x_features_only, y_features_only, x_covariate_columns, y_covariate_columns,
                    x_combined, y_combined
                )
                self.cca_estimator_.fit(x_final, y_final)
        
        return self

    def _optimize_hyperparameters_new_architecture(self, x_combined: pd.DataFrame, y_combined: pd.DataFrame,
                                                   x_features_only: pd.DataFrame, y_features_only: pd.DataFrame,
                                                   progress_callback=None, preprocessing_output_dir: Optional[str] = None):
        """
        Hyperparameter optimization using the new preprocessing architecture.

        This method implements MATLAB-style nested cross-validation with multiple repeats:
        - n_repeats iterations (e.g., 1000 in MATLAB)
        - cv_folds per repeat (e.g., 10-fold CV)
        - Tests all lambda combinations per fold
        - Selects lambda that maximizes median correlation across all repeats and folds

        Args:
            x_combined: X data with covariates
            y_combined: Y data with covariates
            x_features_only: X features only (after outside-CV preprocessing)
            y_features_only: Y features only (after outside-CV preprocessing)
            progress_callback: Optional callback for progress updates
            preprocessing_output_dir: Optional directory to save intermediate preprocessing CSVs
        """
        if not NEW_FEATURE_PROCESSOR_AVAILABLE:
            raise ValueError("New preprocessing architecture not available")

        # Get processing plans
        x_plan = self.x_processing_plan_
        y_plan = self.y_processing_plan_

        # Get optimization settings
        opt_settings = self.settings.optimization_settings
        n_repeats = opt_settings.n_repeats
        cv_folds = opt_settings.cv_folds
        random_state = opt_settings.random_state if opt_settings.random_state is not None else 42

        # Initialize ValidationLogger
        logger = None
        if VALIDATION_LOGGER_AVAILABLE and opt_settings.validation_logging_enabled:
            logger = ValidationLogger(
                log_dir=opt_settings.validation_log_dir,
                enabled=True,
                log_repeats=opt_settings.validation_log_repeats,
                sample_size=opt_settings.validation_log_sample_size,
                debug_mode=opt_settings.validation_debug_mode
            )
            print(f"Validation logging enabled: {opt_settings.validation_log_dir}")

        # Get stratification variable if specified
        stratify_values = None
        if opt_settings.stratify_by:
            if opt_settings.stratify_by in x_combined.columns:
                stratify_values = x_combined[opt_settings.stratify_by].values
            elif opt_settings.stratify_by in y_combined.columns:
                stratify_values = y_combined[opt_settings.stratify_by].values
            else:
                print(f"Warning: Stratification variable '{opt_settings.stratify_by}' not found. Using unstratified CV.")

        # Hyperparameter grid
        param_grid = self._create_hyperparameter_grid()
        n_combinations = len(param_grid)

        # Storage for all scores: shape (n_repeats, cv_folds, n_combinations)
        all_scores = np.full((n_repeats, cv_folds, n_combinations), np.nan)

        print(f"\n{'='*80}")
        print(f"MATLAB-STYLE HYPERPARAMETER OPTIMIZATION")
        print(f"{'='*80}")
        print(f"Repeats: {n_repeats}")
        print(f"CV folds: {cv_folds}")
        print(f"Lambda combinations: {n_combinations}")
        print(f"Total evaluations: {n_repeats * cv_folds * n_combinations:,}")
        print(f"Stratify by: {opt_settings.stratify_by if opt_settings.stratify_by else 'None'}")
        print(f"{'='*80}\n")

        # REPEAT LOOP (MATLAB: for trainingi = 1:training_nfit)
        for repeat_idx in range(n_repeats):
            if repeat_idx % 10 == 0 or repeat_idx < 5:
                print(f"\n{'='*80}")
                print(f"REPEAT {repeat_idx + 1}/{n_repeats}")
                print(f"{'='*80}")

            # Create CV splitter for this repeat (different random seed each time)
            # Option to load MATLAB CV indices for exact replication
            matlab_cv_dir = getattr(self.settings.optimization_settings, 'matlab_cv_indices_dir', None)

            if matlab_cv_dir is not None:
                # Load MATLAB CV indices for exact replication
                import scipy.io
                cv_filename = f"cv_indices_repeat_{repeat_idx + 1:04d}.mat"
                cv_filepath = os.path.join(matlab_cv_dir, cv_filename)

                if os.path.exists(cv_filepath):
                    matlab_cv = scipy.io.loadmat(cv_filepath)
                    cv_splits = []

                    for fold_i in range(cv_folds):
                        # MATLAB is 1-indexed, Python is 0-indexed
                        raw_train = matlab_cv['cv_indices'][0, fold_i]['train']
                        raw_test = matlab_cv['cv_indices'][0, fold_i]['test']

                        # Handle different MATLAB struct export formats
                        if raw_train.shape == (1, 1):
                            # Nested: [[array]]
                            train_idx = raw_train[0, 0].flatten() - 1
                            test_idx = raw_test[0, 0].flatten() - 1
                        else:
                            # Direct: [array] or array
                            train_idx = raw_train.flatten() - 1
                            test_idx = raw_test.flatten() - 1

                        cv_splits.append((train_idx, test_idx))

                    if repeat_idx == 0:
                        print(f"✓ Loaded MATLAB CV indices from: {cv_filepath}")
                        print(f"  Fold 1: {len(cv_splits[0][0])} train, {len(cv_splits[0][1])} test")
                else:
                    print(f"⚠️  MATLAB CV indices not found: {cv_filepath}")
                    print(f"  Falling back to Python StratifiedKFold")
                    matlab_cv_dir = None  # Fall back to Python CV

            if matlab_cv_dir is None:
                # Use Python's StratifiedKFold (may differ from MATLAB)
                if stratify_values is not None:
                    kf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state + repeat_idx)
                    cv_splits = list(kf.split(x_combined, stratify_values))
                else:
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state + repeat_idx)
                    cv_splits = list(kf.split(x_combined))

            # FOLD LOOP (MATLAB: for foldi = 1:cv.NumTestSets)
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                if repeat_idx < 5 or (repeat_idx % 100 == 0 and fold_idx == 0):
                    print(f"\n--- Repeat {repeat_idx + 1}/{n_repeats}, Fold {fold_idx + 1}/{cv_folds} ---")

                # Log CV split (Tier 2 - only for first N repeats)
                if logger:
                    logger.log_cv_split(repeat_idx, fold_idx, train_idx, val_idx, stratify_values)

                # Split data for this fold - use x_combined/y_combined which INCLUDE covariates
                # Covariates are needed for residualization inside CV folds (MATLAB-compatible)
                x_train_fold = x_combined.iloc[train_idx]
                y_train_fold = y_combined.iloc[train_idx]
                x_val_fold = x_combined.iloc[val_idx]
                y_val_fold = y_combined.iloc[val_idx]

                # Apply inside-CV steps (fit on train, transform both train and val)
                x_feature_name = self.settings.x_main_feature_vector_name
                y_feature_name = self.settings.y_main_feature_vector_name

                x_train_processed = apply_inside_cv_steps(x_plan, x_train_fold, fit=True, other_view=y_train_fold,
                                                         output_dir=preprocessing_output_dir, feature_name=x_feature_name)
                y_train_processed = apply_inside_cv_steps(y_plan, y_train_fold, fit=True, other_view=x_train_fold,
                                                         output_dir=preprocessing_output_dir, feature_name=y_feature_name)

                x_val_processed = apply_inside_cv_steps(x_plan, x_val_fold, fit=False, other_view=y_val_fold)
                y_val_processed = apply_inside_cv_steps(y_plan, y_val_fold, fit=False, other_view=x_val_fold)

                # Remove covariate columns after residualization (keep only feature columns)
                x_train_processed = x_train_processed[[c for c in x_train_processed.columns if c not in self.x_covariate_columns_]]
                y_train_processed = y_train_processed[[c for c in y_train_processed.columns if c not in self.y_covariate_columns_]]
                x_val_processed = x_val_processed[[c for c in x_val_processed.columns if c not in self.x_covariate_columns_]]
                y_val_processed = y_val_processed[[c for c in y_val_processed.columns if c not in self.y_covariate_columns_]]

                # Log preprocessed data (Tier 2 - only for first N repeats)
                if logger:
                    logger.log_preprocessing(repeat_idx, fold_idx, "after_inside_cv_steps",
                                           x_train_processed, y_train_processed,
                                           x_val_processed, y_val_processed)

                # LAMBDA COMBINATION LOOP (MATLAB: for each lambda combination)
                for param_idx, params in enumerate(param_grid):
                    # Only print for first few repeats or periodically
                    verbose = (repeat_idx < 3) or (repeat_idx % 100 == 0 and fold_idx == 0 and param_idx < 5)

                    if verbose:
                        print(f"  Testing λx={params['regularization_x']:.4f}, λy={params['regularization_y']:.4f}", end="")

                    try:
                        # Create CCA estimator with these parameters
                        cca_estimator = CCAEstimator(
                            cca_type=self.settings.model_settings.cca_type,
                            n_components=self.settings.model_settings.n_components,
                            regularization_x=params.get('regularization_x', self.settings.model_settings.regularization_x),
                            regularization_y=params.get('regularization_y', self.settings.model_settings.regularization_y),
                            use_matlab_style=(self.settings.model_settings.regularization_method == "covariance")
                        )

                        # Fit and evaluate
                        cca_estimator.fit(x_train_processed, y_train_processed)
                        score = cca_estimator.score(x_val_processed, y_val_processed)

                        # Store score
                        all_scores[repeat_idx, fold_idx, param_idx] = score

                        if verbose:
                            print(f" → {score:.4f}")

                        # Log lambda score (Tier 1 + Tier 2)
                        if logger:
                            logger.log_lambda_score(
                                repeat_idx, fold_idx,
                                params['regularization_x'],
                                params['regularization_y'],
                                score, param_idx, n_combinations
                            )

                    except Exception as e:
                        if verbose:
                            print(f" → FAILED: {e}")
                        # Leave as NaN in all_scores
                        continue

                # Find best lambda for this fold
                fold_scores = all_scores[repeat_idx, fold_idx, :]
                valid_scores = fold_scores[~np.isnan(fold_scores)]
                if len(valid_scores) > 0:
                    best_fold_idx = np.nanargmax(fold_scores)
                    best_fold_params = param_grid[best_fold_idx]
                    best_fold_score = fold_scores[best_fold_idx]

                    if logger:
                        logger.log_fold_best_lambda(
                            repeat_idx, fold_idx,
                            best_fold_params['regularization_x'],
                            best_fold_params['regularization_y'],
                            best_fold_score
                        )

            # Find optimal lambda for this repeat (median across folds)
            repeat_scores = all_scores[repeat_idx, :, :]  # Shape: (cv_folds, n_combinations)
            repeat_median_scores = np.nanmedian(repeat_scores, axis=0)  # Shape: (n_combinations,)

            if not np.all(np.isnan(repeat_median_scores)):
                best_repeat_idx = np.nanargmax(repeat_median_scores)
                best_repeat_params = param_grid[best_repeat_idx]
                best_repeat_score = repeat_median_scores[best_repeat_idx]

                if repeat_idx < 5 or repeat_idx % 100 == 0:
                    print(f"\nRepeat {repeat_idx + 1} optimal: λx={best_repeat_params['regularization_x']:.4f}, "
                          f"λy={best_repeat_params['regularization_y']:.4f}, median_score={best_repeat_score:.4f}")

                if logger:
                    logger.log_repeat_summary(
                        repeat_idx,
                        best_repeat_params['regularization_x'],
                        best_repeat_params['regularization_y'],
                        best_repeat_score
                    )

        # MATLAB SELECTION CRITERION: max(median(val_r_results, [1 2]))
        # Median across both repeats (axis 0) and folds (axis 1)
        print(f"\n{'='*80}")
        print(f"FINAL LAMBDA SELECTION (MATLAB-style)")
        print(f"{'='*80}")

        overall_median_scores = np.nanmedian(all_scores, axis=(0, 1))  # Shape: (n_combinations,)

        if np.all(np.isnan(overall_median_scores)):
            raise ValueError("All lambda combinations failed. Cannot select optimal parameters.")

        best_idx = np.nanargmax(overall_median_scores)
        best_params = param_grid[best_idx]
        best_score = overall_median_scores[best_idx]

        print(f"Optimal lambda_x: {best_params['regularization_x']:.6f}")
        print(f"Optimal lambda_y: {best_params['regularization_y']:.6f}")
        print(f"Median correlation: {best_score:.6f}")
        print(f"Selection method: max(median(scores across {n_repeats} repeats and {cv_folds} folds))")
        print(f"{'='*80}\n")

        # Save validation logs
        if logger:
            logger.save_final_summary(
                best_params['regularization_x'],
                best_params['regularization_y'],
                best_score,
                selection_method=f"max_median_across_{n_repeats}_repeats_and_{cv_folds}_folds"
            )

        # Store results
        self.best_params_ = best_params
        self.cv_results_ = {
            'all_scores': all_scores,
            'median_scores': overall_median_scores,
            'best_score': best_score,
            'best_idx': best_idx,
            'param_grid': param_grid
        }

        # Refit inside-CV steps on full training data with optimal parameters
        # NOTE: Must use x_combined/y_combined (includes covariates) for residualization,
        # then remove covariate columns after - same as CV loop does
        print("Refitting on full dataset with optimal parameters...")
        x_feature_name = self.settings.x_main_feature_vector_name
        y_feature_name = self.settings.y_main_feature_vector_name
        x_final = apply_inside_cv_steps(x_plan, x_combined, fit=True, other_view=y_combined,
                                       output_dir=preprocessing_output_dir, feature_name=x_feature_name)
        y_final = apply_inside_cv_steps(y_plan, y_combined, fit=True, other_view=x_combined,
                                       output_dir=preprocessing_output_dir, feature_name=y_feature_name)

        # Remove covariate columns after residualization (keep only feature columns)
        x_final = x_final[[c for c in x_final.columns if c not in self.x_covariate_columns_]]
        y_final = y_final[[c for c in y_final.columns if c not in self.y_covariate_columns_]]

        # Update CCA estimator with best parameters and fit on full data
        self.cca_estimator_.set_params(**best_params)
        self.cca_estimator_.fit(x_final, y_final)

        print(f"Hyperparameter optimization completed. Best median score: {best_score:.4f}")

    def _create_hyperparameter_grid(self):
        """Create hyperparameter grid for optimization."""
        opt_settings = self.settings.optimization_settings

        # Check if explicit lambda values are provided (preferred for exact MATLAB matching)
        if opt_settings.regularization_x_values is not None and opt_settings.regularization_y_values is not None:
            # Use explicit values provided by user
            x_reg_values = np.array(opt_settings.regularization_x_values)
            y_reg_values = np.array(opt_settings.regularization_y_values)
            print(f"Using explicit lambda values: {len(x_reg_values)} x {len(y_reg_values)} = {len(x_reg_values) * len(y_reg_values)} combinations")
        else:
            # Generate logarithmically spaced points from ranges
            x_reg_range = opt_settings.regularization_x_range
            y_reg_range = opt_settings.regularization_y_range
            search_points = opt_settings.regularization_search_points

            x_reg_values = np.logspace(np.log10(x_reg_range[0]), np.log10(x_reg_range[1]), search_points)
            y_reg_values = np.logspace(np.log10(y_reg_range[0]), np.log10(y_reg_range[1]), search_points)
            print(f"Generated lambda grid: {search_points} x {search_points} = {search_points * search_points} combinations")

        # Create grid from all combinations
        param_grid = []
        for x_reg in x_reg_values:
            for y_reg in y_reg_values:
                param_grid.append({
                    'regularization_x': x_reg,
                    'regularization_y': y_reg
                })

        print(f"Created hyperparameter grid with {len(param_grid)} combinations ({len(x_reg_values)}×{len(y_reg_values)})")
        return param_grid

    def _apply_input_preprocessing(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy method - input preprocessing is now handled by FeatureProcessor.
        This method now only applies standardization which is required for CCA.
        """
        # Apply standardization (required for CCA)
        x_data, y_data = self._standardize_features(x_data, y_data)
        return x_data, y_data
    
    def _apply_data_transformation(self, data: pd.DataFrame, transform_type: str) -> pd.DataFrame:
        """Apply data transformation."""
        if transform_type == "log":
            return pd.DataFrame(np.log1p(data), index=data.index, columns=data.columns)
        elif transform_type == "sqrt":
            return pd.DataFrame(np.sqrt(np.abs(data)), index=data.index, columns=data.columns)
        elif transform_type == "zscore":
            return pd.DataFrame(StandardScaler().fit_transform(data), index=data.index, columns=data.columns)
        elif transform_type == "robust_scale":
            return pd.DataFrame(RobustScaler().fit_transform(data), index=data.index, columns=data.columns)
        else:
            return data
    
    def _apply_prepca(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Apply pre-PCA dimensionality reduction."""
        # Legacy method - PCA is now handled by FeatureProcessor
        variance_threshold = 0.95  # Default value since input_preprocessing removed
        
        # Apply PCA to X data
        x_pca = PCA(n_components=variance_threshold)
        x_transformed = x_pca.fit_transform(x_data)
        x_df = pd.DataFrame(x_transformed, index=x_data.index, 
                           columns=[f"X_PC{i+1}" for i in range(x_transformed.shape[1])])
        
        # Apply PCA to Y data
        y_pca = PCA(n_components=variance_threshold)
        y_transformed = y_pca.fit_transform(y_data)
        y_df = pd.DataFrame(y_transformed, index=y_data.index,
                           columns=[f"Y_PC{i+1}" for i in range(y_transformed.shape[1])])
        
        # Store PCA models
        self.x_pca_ = x_pca
        self.y_pca_ = y_pca
        
        return x_df, y_df
    
    def _apply_feature_selection(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy method - feature selection is now handled by FeatureProcessor.
        This method now just returns the data as-is.
        """
        # Feature selection is now handled by FeatureProcessor before CCA
        return x_data, y_data
    
    def _remove_correlated_features(self, data: pd.DataFrame, threshold: float) -> pd.DataFrame:
        """Remove highly correlated features."""
        corr_matrix = data.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
        return data.drop(columns=to_drop)
    
    def _standardize_features(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Standardize numeric features for CCA.
        Categorical features are left unchanged.
        """
        from sklearn.preprocessing import StandardScaler
        
        # Identify numeric columns
        x_numeric_cols = x_data.select_dtypes(include=[np.number]).columns
        y_numeric_cols = y_data.select_dtypes(include=[np.number]).columns
        
        x_standardized = x_data.copy()
        y_standardized = y_data.copy()
        
        # Standardize only numeric columns
        if len(x_numeric_cols) > 0:
            x_scaler = StandardScaler()
            x_standardized[x_numeric_cols] = x_scaler.fit_transform(x_data[x_numeric_cols])
        
        if len(y_numeric_cols) > 0:
            y_scaler = StandardScaler()
            y_standardized[y_numeric_cols] = y_scaler.fit_transform(y_data[y_numeric_cols])
        
        return x_standardized, y_standardized
    
    def _optimize_hyperparameters_fixed_features(self, 
                                                x_combined: pd.DataFrame, 
                                                y_combined: pd.DataFrame,
                                                x_features_only: pd.DataFrame, 
                                                y_features_only: pd.DataFrame,
                                                x_covariate_columns: List[str],
                                                y_covariate_columns: List[str],
                                                progress_callback=None):
        """
        Optimize hyperparameters using cross-validation with FIXED feature selection.
        Feature selection has already been applied - now we just vary regularization parameters.
        """
        # Create parameter grid
        param_grid = self._create_parameter_grid()
        param_combinations = self._expand_parameter_grid(param_grid)
        
        # Create cross-validation strategy
        cv_strategy = self._create_cv_strategy(x_features_only)
        n_splits = cv_strategy.get_n_splits(x_features_only)
        
        print(f"Starting hyperparameter optimization: {len(param_combinations)} parameter combinations, {n_splits}-fold CV")

        # Track best parameters and problematic combinations
        best_score = -np.inf
        best_params = None
        warning_count = 0
        problematic_params = []
        
        # Iterate through parameter combinations
        for param_idx, params in enumerate(param_combinations):
            cv_scores = []
            
            # Cross-validation with FIXED features
            for fold_idx, (train_idx, val_idx) in enumerate(cv_strategy.split(x_features_only)):
                # Get train/validation splits of FEATURE-SELECTED data
                x_train_fold = x_features_only.iloc[train_idx]
                y_train_fold = y_features_only.iloc[train_idx]
                x_val_fold = x_features_only.iloc[val_idx]
                y_val_fold = y_features_only.iloc[val_idx]
                
                # Get corresponding covariate data if needed
                x_train_with_covs = x_combined.iloc[train_idx] if x_covariate_columns else x_train_fold
                y_train_with_covs = y_combined.iloc[train_idx] if y_covariate_columns else y_train_fold
                x_val_with_covs = x_combined.iloc[val_idx] if x_covariate_columns else x_val_fold
                y_val_with_covs = y_combined.iloc[val_idx] if y_covariate_columns else y_val_fold
                
                try:
                    # Apply fold-specific preprocessing (residualization + standardization)
                    # Training data
                    x_train_processed, y_train_processed, train_transformers = self._apply_fold_specific_preprocessing(
                        x_train_fold, y_train_fold, x_covariate_columns, y_covariate_columns,
                        x_train_with_covs, y_train_with_covs
                    )

                    # Validation data (use training fold's fitted transformers)
                    x_val_processed, y_val_processed, _ = self._apply_fold_specific_preprocessing(
                        x_val_fold, y_val_fold, x_covariate_columns, y_covariate_columns,
                        x_val_with_covs, y_val_with_covs, train_transformers
                    )

                    # Validate processed data before CCA (only critical issues)
                    if x_train_processed.empty or y_train_processed.empty:
                        continue  # Skip silently - this is expected in some edge cases

                    if x_val_processed.empty or y_val_processed.empty:
                        continue  # Skip silently - this is expected in some edge cases

                    # Basic sanity check: ensure we have some numeric data
                    x_train_numeric = x_train_processed.select_dtypes(include=[np.number])
                    y_train_numeric = y_train_processed.select_dtypes(include=[np.number])

                    if x_train_numeric.shape[1] == 0 or y_train_numeric.shape[1] == 0:
                        continue  # Skip silently - no numeric features

                    # MATLAB-style sample size check: very permissive
                    # MATLAB's rCCA works with regularization even when samples < features
                    if len(x_train_processed) < 2 or len(x_val_processed) < 2:
                        continue  # Skip silently - need at least 2 samples

                    # Create and fit CCA model with current parameters
                    # Apply MATLAB-style regularization adjustment for numerical stability
                    adjusted_params = params.copy()
                    reg_x = adjusted_params.get('regularization_x', 0.1)
                    reg_y = adjusted_params.get('regularization_y', 0.1)

                    # MATLAB's rCCA is more robust to very small regularization values
                    # Add small epsilon for numerical stability if needed
                    min_reg = 1e-6
                    if reg_x < min_reg:
                        adjusted_params['regularization_x'] = min_reg
                    if reg_y < min_reg:
                        adjusted_params['regularization_y'] = min_reg

                    # Add MATLAB-style parameter to adjusted params
                    adjusted_params['use_matlab_style'] = (self.settings.model_settings.regularization_method == "covariance")
                    fold_estimator = CCAEstimator(**adjusted_params)

                    # Capture warnings during CCA fit to track problematic parameter combinations
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        fold_estimator.fit(x_train_processed, y_train_processed)

                        # Log any warnings with parameter details
                        if w:
                            reg_x = adjusted_params.get('regularization_x', 'N/A')
                            reg_y = adjusted_params.get('regularization_y', 'N/A')
                            warning_count += len(w)
                            for warning in w:
                                if "invalid value encountered in sqrt" in str(warning.message):
                                    print(f"⚠️  sqrt warning: fold {fold_idx}, λ_x={reg_x:.6f}, λ_y={reg_y:.6f}")
                                    problematic_params.append((reg_x, reg_y, "sqrt"))
                                elif "RuntimeWarning" in str(warning.category.__name__):
                                    print(f"⚠️  {warning.category.__name__}: fold {fold_idx}, λ_x={reg_x:.6f}, λ_y={reg_y:.6f} - {warning.message}")
                                    problematic_params.append((reg_x, reg_y, str(warning.category.__name__)))

                    # Score on validation data
                    score = fold_estimator.score(x_val_processed, y_val_processed)

                    # Validate score
                    if np.isfinite(score):
                        cv_scores.append(score)
                    # Skip non-finite scores silently

                except Exception as e:
                    # Log what's causing high lambda failures for debugging
                    if reg_x > 1.0 or reg_y > 1.0:
                        print(f"   ⚠️  High λ failure: fold {fold_idx}, λ_x={reg_x:.6f}, λ_y={reg_y:.6f} - {str(e)[:50]}...")
                    continue
            
            # Calculate mean CV score for this parameter combination
            # MATLAB-style: accept parameter combinations if at least 2 folds succeed
            min_successful_folds = max(2, n_splits // 2)  # At least 2 folds, or half the folds
            median_score = None  # Initialize to avoid UnboundLocalError

            if len(cv_scores) >= min_successful_folds:
                # Use MEDIAN like MATLAB: max(median(val_r_results,[1 2]))
                median_score = np.median(cv_scores)
                print(f" → score={median_score:.4f} ({len(cv_scores)}/{n_splits} folds)")

                # Update best parameters if this is the best score so far
                if median_score > best_score:
                    best_score = median_score
                    best_params = params.copy()
                    print(f"   ✨ New best! λ_x={reg_x:.6f}, λ_y={reg_y:.6f}, score={best_score:.4f}")
            elif len(cv_scores) > 0:
                print(f" → insufficient folds ({len(cv_scores)}/{n_splits}, need ≥{min_successful_folds})")
            else:
                print(" → skipped (no valid scores)")

            # Progress reporting
            if progress_callback:
                progress = 20 + int((param_idx + 1) / len(param_combinations) * 60)  # 20-80% of total
                λ_x = params.get('regularization_x', 'N/A')
                λ_y = params.get('regularization_y', 'N/A')

                # Handle case where median_score might be None
                score_text = f"{median_score:.4f}" if median_score is not None else "N/A"
                best_score_text = f"{best_score:.4f}" if best_score > -np.inf else "N/A"

                # Format best parameters safely
                best_x = best_params.get('regularization_x', 'N/A') if best_params else 'N/A'
                best_y = best_params.get('regularization_y', 'N/A') if best_params else 'N/A'
                best_x_text = f"{best_x:.3f}" if isinstance(best_x, (int, float)) else str(best_x)
                best_y_text = f"{best_y:.3f}" if isinstance(best_y, (int, float)) else str(best_y)

                progress_callback(progress,
                    f"Parameter {param_idx+1}/{len(param_combinations)} (λ_x={λ_x:.3f}, λ_y={λ_y:.3f}): "
                    f"score={score_text}, best={best_score_text} "
                    f"(λ_x={best_x_text}, λ_y={best_y_text})")
        
        # Validate that we found at least one valid parameter combination
        if best_params is None:
            print("Warning: No valid parameter combinations found during hyperparameter optimization.")
            print("Falling back to default regularization parameters.")
            # Use safe default parameters
            best_params = {
                'cca_type': self.settings.model_settings.cca_type,
                'n_components': self.settings.model_settings.n_components,
                'regularization_x': 0.1,
                'regularization_y': 0.1
            }
            best_score = 0.0

        # Store best parameters and fit final model
        self.best_params_ = best_params
        print(f"Hyperparameter optimization complete. Best parameters: "
              f"λ_x={best_params['regularization_x']:.3f}, λ_y={best_params['regularization_y']:.3f}, "
              f"score: {best_score:.4f}")

        # Summary of warnings
        if warning_count > 0:
            print(f"\n📊 Warning Summary: {warning_count} warnings encountered during optimization")
            if problematic_params:
                # Group by warning type
                sqrt_warnings = [(x, y) for x, y, t in problematic_params if t == "sqrt"]
                other_warnings = [(x, y, t) for x, y, t in problematic_params if t != "sqrt"]

                if sqrt_warnings:
                    print(f"   • sqrt warnings: {len(sqrt_warnings)} occurrences")
                    # Show range of problematic lambda values
                    x_vals = [x for x, y in sqrt_warnings]
                    y_vals = [y for x, y in sqrt_warnings]
                    print(f"     λ_x range: {min(x_vals):.6f} to {max(x_vals):.6f}")
                    print(f"     λ_y range: {min(y_vals):.6f} to {max(y_vals):.6f}")

                if other_warnings:
                    print(f"   • Other warnings: {len(other_warnings)} occurrences")
            print("   ℹ️  These warnings are often normal for extreme lambda values and don't prevent optimization")

        if progress_callback:
            progress_callback(80, f"Hyperparameter optimization complete. Best parameters: "
                             f"λ_x={best_params['regularization_x']:.3f}, λ_y={best_params['regularization_y']:.3f}, "
                             f"score: {best_score:.4f}")
        
        # Fit final model with best parameters on full dataset
        x_final_processed, y_final_processed, final_transformers = self._apply_fold_specific_preprocessing(
            x_features_only, y_features_only, x_covariate_columns, y_covariate_columns,
            x_combined, y_combined
        )
        
        # Store information needed for consistent inference
        self.x_feature_columns_ = x_features_only.columns.tolist()
        self.y_feature_columns_ = y_features_only.columns.tolist()
        self.x_covariate_columns_ = x_covariate_columns
        self.y_covariate_columns_ = y_covariate_columns
        self.final_transformers_ = final_transformers
        
        # Ensure MATLAB-style parameter is included in final model
        final_params = best_params.copy()
        final_params['use_matlab_style'] = (self.settings.model_settings.regularization_method == "covariance")

        self.cca_estimator_ = CCAEstimator(**final_params)
        self.cca_estimator_.fit(x_final_processed, y_final_processed)
    
    def _create_parameter_grid(self) -> Dict:
        """Create parameter grid for hyperparameter optimization."""
        param_grid = {}
        
        # CRITICAL FIX: Always include essential CCA parameters
        param_grid['cca_type'] = [self.settings.model_settings.cca_type]
        param_grid['n_components'] = [self.settings.model_settings.n_components]
        
        if self.settings.optimization_settings.optimize_regularization:
            reg_x_min, reg_x_max = self.settings.optimization_settings.regularization_x_range
            reg_y_min, reg_y_max = self.settings.optimization_settings.regularization_y_range
            n_points = self.settings.optimization_settings.regularization_search_points
            
            param_grid['regularization_x'] = np.logspace(np.log10(reg_x_min), np.log10(reg_x_max), n_points)
            param_grid['regularization_y'] = np.logspace(np.log10(reg_y_min), np.log10(reg_y_max), n_points)
        else:
            # Use fixed regularization values if not optimizing
            param_grid['regularization_x'] = [self.settings.model_settings.regularization_x]
            param_grid['regularization_y'] = [self.settings.model_settings.regularization_y]
        
        if self.settings.optimization_settings.optimize_n_components:
            n_comp_min, n_comp_max = self.settings.optimization_settings.n_components_range
            param_grid['n_components'] = list(range(n_comp_min, n_comp_max + 1))
        
        if self.settings.optimization_settings.optimize_sparsity:
            sparse_min, sparse_max = self.settings.optimization_settings.sparsity_range
            param_grid['sparsity_penalty_x'] = np.linspace(sparse_min, sparse_max, 10)
            param_grid['sparsity_penalty_y'] = np.linspace(sparse_min, sparse_max, 10)
        else:
            # Use fixed sparsity values if not optimizing
            param_grid['sparsity_penalty_x'] = [self.settings.model_settings.sparsity_penalty_x]
            param_grid['sparsity_penalty_y'] = [self.settings.model_settings.sparsity_penalty_y]
        
        return param_grid
    
    def _expand_parameter_grid(self, param_grid: Dict) -> List[Dict]:
        """Expand parameter grid into list of parameter combinations."""
        from itertools import product
        
        if not param_grid:
            return [{}]
        
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _create_cv_strategy(self, x_data: pd.DataFrame):
        """Create cross-validation strategy."""
        cv_method = self.settings.optimization_settings.cv_method
        cv_folds = self.settings.optimization_settings.cv_folds

        if cv_method == "kfold":
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_method == "stratified":
            # Would need stratification variable
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        elif cv_method == "leave_one_out":
            return LeaveOneOut()
        else:
            return KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    def _compute_structural_loadings(self, original_data: pd.DataFrame,
                                     variates: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute structural loadings by correlating original features with CCA variates.

        Structural loadings are the correlations between each original feature and each
        CCA variate. This provides interpretable loadings in the original variable space,
        which is essential for visualization (e.g., topographic plots).

        Args:
            original_data: Original preprocessed features (N × p) before PCA
            variates: CCA variates (N × k) where k is number of components

        Returns:
            Tuple of (loadings, p_values):
                - loadings: (p × k) array of correlations
                - p_values: (p × k) array of p-values for correlations
        """
        from scipy.stats import pearsonr

        n_features = original_data.shape[1]
        n_components = variates.shape[1]

        loadings = np.zeros((n_features, n_components))
        p_values = np.zeros((n_features, n_components))

        for feat_idx in range(n_features):
            feature_values = original_data.iloc[:, feat_idx].values

            for comp_idx in range(n_components):
                variate_values = variates[:, comp_idx]

                # Compute Pearson correlation
                corr, pval = pearsonr(feature_values, variate_values)
                loadings[feat_idx, comp_idx] = corr
                p_values[feat_idx, comp_idx] = pval

        return loadings, p_values

    def _compute_bootstrap_loading_ci(self, original_data: pd.DataFrame,
                                      variates: np.ndarray,
                                      n_iterations: int = 10000,
                                      confidence_level: float = 0.95,
                                      random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute bootstrap confidence intervals for structural loadings.

        This implements the MATLAB methodology from rCCA_runall.m lines 803-812:
        - Bootstrap resampling with replacement
        - Compute correlation for each bootstrap sample
        - Extract confidence interval bounds

        OPTIMIZED VERSION: Uses vectorized operations for 100-1000x speedup.

        Args:
            original_data: Original preprocessed features (N × p) before PCA
            variates: CCA variates (N × k) where k is number of components
            n_iterations: Number of bootstrap iterations (default: 10000, matching MATLAB)
            confidence_level: Confidence level for intervals (default: 0.95)
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (ci_lower, ci_upper):
                - ci_lower: (p × k) array of lower CI bounds
                - ci_upper: (p × k) array of upper CI bounds
        """
        if random_state is not None:
            np.random.seed(random_state)

        n_samples = original_data.shape[0]
        n_features = original_data.shape[1]
        n_components = variates.shape[1]

        # Convert to numpy array for faster indexing
        features_array = original_data.values  # (n_samples, n_features)

        # Pre-allocate array for all bootstrap correlations
        # Shape: (n_features, n_components, n_iterations)
        all_bootstrap_corrs = np.zeros((n_features, n_components, n_iterations))

        # Vectorized correlation computation
        # For each bootstrap iteration, compute all feature-component correlations at once
        print(f"   Computing {n_iterations} bootstrap samples for {n_features} features × {n_components} components...")

        for boot_iter in range(n_iterations):
            # Progress indicator every 1000 iterations
            if boot_iter > 0 and boot_iter % 1000 == 0:
                print(f"   Progress: {boot_iter}/{n_iterations} iterations ({100*boot_iter/n_iterations:.1f}%)")

            # Generate bootstrap indices once per iteration
            boot_indices = np.random.choice(n_samples, size=n_samples, replace=True)

            # Resample features and variates
            boot_features = features_array[boot_indices, :]  # (n_samples, n_features)
            boot_variates = variates[boot_indices, :]  # (n_samples, n_components)

            # Compute correlations for all feature-component pairs at once
            # Using vectorized correlation formula: corr(X, Y) = cov(X, Y) / (std(X) * std(Y))

            # Center the data
            boot_features_centered = boot_features - boot_features.mean(axis=0, keepdims=True)
            boot_variates_centered = boot_variates - boot_variates.mean(axis=0, keepdims=True)

            # Compute standard deviations
            features_std = boot_features_centered.std(axis=0, keepdims=True)  # (1, n_features)
            variates_std = boot_variates_centered.std(axis=0, keepdims=True)  # (1, n_components)

            # Compute correlations: (n_features, n_components)
            # corr = (X^T @ Y) / (n * std(X) * std(Y))
            correlations = (boot_features_centered.T @ boot_variates_centered) / (n_samples - 1)
            correlations = correlations / (features_std.T @ variates_std)

            # Store correlations for this bootstrap iteration
            all_bootstrap_corrs[:, :, boot_iter] = correlations

        print(f"   ✓ Bootstrap sampling complete!")

        # Compute confidence interval bounds using percentile method
        # MATLAB bootci uses percentile method by default
        alpha = 1 - confidence_level
        ci_lower = np.percentile(all_bootstrap_corrs, 100 * alpha / 2, axis=2)  # (n_features, n_components)
        ci_upper = np.percentile(all_bootstrap_corrs, 100 * (1 - alpha / 2), axis=2)  # (n_features, n_components)

        return ci_lower, ci_upper

    def get_results(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Dict:
        """
        Extract results from fitted model.

        IMPORTANT ASSUMPTION:
        - x_data and y_data have ALREADY had outside-CV steps applied (same as in fit())
        - We only need to apply inside-CV steps here using the fitted transformers
        """
        if not hasattr(self, 'cca_estimator_'):
            raise ValueError("Model must be fitted before extracting results")

        # CRITICAL FIX: Apply the SAME feature selection and preprocessing pipeline
        # that was used during training

        # Check if we're using the new preprocessing architecture
        if hasattr(self, 'x_processing_plan_') and hasattr(self, 'y_processing_plan_'):
            # New preprocessing architecture: apply only inside-CV steps
            # Outside-CV steps were already applied by fusion.py before calling this method

            # Apply inside-CV steps (residualize, standardize, PCA in with_hpo mode)
            # Use fit=False to apply the transformers fitted during training
            x_processed = apply_inside_cv_steps(self.x_processing_plan_, x_data, fit=False, other_view=y_data)
            y_processed = apply_inside_cv_steps(self.y_processing_plan_, y_data, fit=False, other_view=x_data)

            # Extract feature columns (excluding covariates)
            x_covariate_columns = getattr(self, 'x_covariate_columns_', [])
            y_covariate_columns = getattr(self, 'y_covariate_columns_', [])

            x_feature_columns = [col for col in x_processed.columns if col not in x_covariate_columns]
            y_feature_columns = [col for col in y_processed.columns if col not in y_covariate_columns]

            x_processed = x_processed[x_feature_columns] if x_feature_columns else x_processed
            y_processed = y_processed[y_feature_columns] if y_feature_columns else y_processed
        else:
            # Legacy preprocessing architecture
            # Step 1: Apply feature selection and transforms (same as training)
            if hasattr(self, 'x_feature_columns_') and hasattr(self, 'y_feature_columns_'):
                # Use the exact same features that were selected during training
                x_features_selected = x_data[self.x_feature_columns_]
                y_features_selected = y_data[self.y_feature_columns_]
            else:
                # Fallback: apply feature selection fresh (not ideal but prevents crashes)
                x_features_selected, y_features_selected = self._apply_feature_selection_and_transforms(x_data, y_data)

            # Step 2: Apply the same fold-specific preprocessing transformers that were fitted during training
            if hasattr(self, 'final_transformers_'):
                # Use the same transformers that were fitted on the full training data
                x_processed, y_processed, _ = self._apply_fold_specific_preprocessing(
                    x_features_selected, y_features_selected,
                    self.x_covariate_columns_, self.y_covariate_columns_,
                    x_data, y_data, self.final_transformers_
                )
            else:
                # Fallback: fit fresh transformers (not ideal)
                x_processed, y_processed, _ = self._apply_fold_specific_preprocessing(
                    x_features_selected, y_features_selected,
                    self.x_covariate_columns_, self.y_covariate_columns_,
                    x_data, y_data
                )
        
        # Transform data using the SAME preprocessing that was used during training
        transformed = self.cca_estimator_.transform(x_processed, y_processed)

        # Use canonical correlations from the fitted model (SVD singular values)
        # This is more accurate than re-computing from variates
        cca_model = self.cca_estimator_.cca_model_
        if hasattr(cca_model, 'correlations_'):
            # MATLAB-style rCCA stores correlations from SVD
            correlations = np.array(cca_model.correlations_)
        else:
            # Fallback: compute correlations from variates
            correlations = []
            for i in range(min(transformed[0].shape[1], transformed[1].shape[1])):
                x_variate = transformed[0][:, i]
                y_variate = transformed[1][:, i]

                # Check for constant variates (zero variance)
                if np.var(x_variate) == 0 or np.var(y_variate) == 0:
                    correlations.append(0.0)
                    continue

                # Calculate correlation with NaN handling
                corr_matrix = np.corrcoef(x_variate, y_variate)
                corr = corr_matrix[0, 1]

                # Handle NaN/inf cases
                if np.isnan(corr) or np.isinf(corr):
                    correlations.append(0.0)
                else:
                    correlations.append(abs(corr))
            correlations = np.array(correlations)

        # CCA correlations are already sorted by SVD, no need to re-sort
        
        # Extract weights properly from different CCA model types
        weights = None
        cca_model = self.cca_estimator_.cca_model_

        if hasattr(cca_model, 'x_weights_') and hasattr(cca_model, 'y_weights_'):
            # MATLAB-style rCCA or similar models
            x_weights = getattr(cca_model, 'x_weights_', None)
            y_weights = getattr(cca_model, 'y_weights_', None)
            if x_weights is not None and y_weights is not None:
                # Weights are already in correct order from SVD (no sorting needed)
                weights = {
                    'x_weights': x_weights,
                    'y_weights': y_weights
                }
        elif hasattr(cca_model, 'weights_'):
            # CCA-Zoo style models
            weights = getattr(cca_model, 'weights_', None)

        # Compute structural loadings if we have the original preprocessed data (before PCA)
        structural_loadings = None
        if hasattr(self, 'x_data_before_pca_') and hasattr(self, 'y_data_before_pca_'):
            # Compute structural loadings for X and Y using data before PCA
            x_loadings, x_pvals = self._compute_structural_loadings(
                self.x_data_before_pca_, transformed[0]
            )
            y_loadings, y_pvals = self._compute_structural_loadings(
                self.y_data_before_pca_, transformed[1]
            )

            structural_loadings = {
                'x_loadings': x_loadings,
                'x_pvalues': x_pvals,
                'x_feature_names': self.x_data_before_pca_.columns.tolist(),
                'y_loadings': y_loadings,
                'y_pvalues': y_pvals,
                'y_feature_names': self.y_data_before_pca_.columns.tolist()
            }

            # Compute bootstrap confidence intervals if enabled
            print(f"[Bootstrap] Settings: enabled={self.settings.bootstrap_settings.enabled}, compute_ci={self.settings.bootstrap_settings.compute_loading_ci}")
            if self.settings.bootstrap_settings.enabled and self.settings.bootstrap_settings.compute_loading_ci:
                print(f"[Bootstrap] Computing confidence intervals ({self.settings.bootstrap_settings.n_iterations} iterations)...")

                x_ci_lower, x_ci_upper = self._compute_bootstrap_loading_ci(
                    self.x_data_before_pca_, transformed[0],
                    n_iterations=self.settings.bootstrap_settings.n_iterations,
                    confidence_level=self.settings.bootstrap_settings.confidence_level,
                    random_state=self.settings.bootstrap_settings.random_state
                )
                y_ci_lower, y_ci_upper = self._compute_bootstrap_loading_ci(
                    self.y_data_before_pca_, transformed[1],
                    n_iterations=self.settings.bootstrap_settings.n_iterations,
                    confidence_level=self.settings.bootstrap_settings.confidence_level,
                    random_state=self.settings.bootstrap_settings.random_state
                )

                # Add bootstrap CI to structural loadings
                structural_loadings['x_ci_lower'] = x_ci_lower
                structural_loadings['x_ci_upper'] = x_ci_upper
                structural_loadings['y_ci_lower'] = y_ci_lower
                structural_loadings['y_ci_upper'] = y_ci_upper

                # Compute FDR-corrected significance if enabled
                if self.settings.bootstrap_settings.compute_loading_pvalues:
                    from statsmodels.stats.multitest import multipletests

                    # Apply FDR correction to p-values for each component
                    n_components = x_pvals.shape[1]
                    x_significant = np.zeros_like(x_pvals, dtype=bool)
                    y_significant = np.zeros_like(y_pvals, dtype=bool)

                    for comp_idx in range(n_components):
                        # X loadings FDR correction
                        if self.settings.bootstrap_settings.loading_correction_method == "fdr":
                            _, x_pvals_corrected, _, _ = multipletests(
                                x_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='fdr_bh'
                            )
                            x_significant[:, comp_idx] = x_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        elif self.settings.bootstrap_settings.loading_correction_method == "bonferroni":
                            _, x_pvals_corrected, _, _ = multipletests(
                                x_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='bonferroni'
                            )
                            x_significant[:, comp_idx] = x_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        else:  # no correction
                            x_significant[:, comp_idx] = x_pvals[:, comp_idx] < self.settings.bootstrap_settings.loading_alpha

                        # Y loadings FDR correction
                        if self.settings.bootstrap_settings.loading_correction_method == "fdr":
                            _, y_pvals_corrected, _, _ = multipletests(
                                y_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='fdr_bh'
                            )
                            y_significant[:, comp_idx] = y_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        elif self.settings.bootstrap_settings.loading_correction_method == "bonferroni":
                            _, y_pvals_corrected, _, _ = multipletests(
                                y_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='bonferroni'
                            )
                            y_significant[:, comp_idx] = y_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        else:  # no correction
                            y_significant[:, comp_idx] = y_pvals[:, comp_idx] < self.settings.bootstrap_settings.loading_alpha

                    structural_loadings['x_significant'] = x_significant
                    structural_loadings['y_significant'] = y_significant

                print("✓ Bootstrap confidence intervals computed")
        elif hasattr(self, 'x_original_preprocessed_') and hasattr(self, 'y_original_preprocessed_'):
            # Fallback to legacy approach (for backward compatibility)
            x_loadings, x_pvals = self._compute_structural_loadings(
                self.x_original_preprocessed_, transformed[0]
            )
            y_loadings, y_pvals = self._compute_structural_loadings(
                self.y_original_preprocessed_, transformed[1]
            )

            structural_loadings = {
                'x_loadings': x_loadings,
                'x_pvalues': x_pvals,
                'x_feature_names': self.x_original_preprocessed_.columns.tolist(),
                'y_loadings': y_loadings,
                'y_pvalues': y_pvals,
                'y_feature_names': self.y_original_preprocessed_.columns.tolist()
            }

            # Compute bootstrap confidence intervals if enabled (legacy path)
            if self.settings.bootstrap_settings.enabled and self.settings.bootstrap_settings.compute_loading_ci:
                print(f"Computing bootstrap confidence intervals ({self.settings.bootstrap_settings.n_iterations} iterations)...")

                x_ci_lower, x_ci_upper = self._compute_bootstrap_loading_ci(
                    self.x_original_preprocessed_, transformed[0],
                    n_iterations=self.settings.bootstrap_settings.n_iterations,
                    confidence_level=self.settings.bootstrap_settings.confidence_level,
                    random_state=self.settings.bootstrap_settings.random_state
                )
                y_ci_lower, y_ci_upper = self._compute_bootstrap_loading_ci(
                    self.y_original_preprocessed_, transformed[1],
                    n_iterations=self.settings.bootstrap_settings.n_iterations,
                    confidence_level=self.settings.bootstrap_settings.confidence_level,
                    random_state=self.settings.bootstrap_settings.random_state
                )

                structural_loadings['x_ci_lower'] = x_ci_lower
                structural_loadings['x_ci_upper'] = x_ci_upper
                structural_loadings['y_ci_lower'] = y_ci_lower
                structural_loadings['y_ci_upper'] = y_ci_upper

                # Compute FDR-corrected significance if enabled
                if self.settings.bootstrap_settings.compute_loading_pvalues:
                    from statsmodels.stats.multitest import multipletests

                    n_components = x_pvals.shape[1]
                    x_significant = np.zeros_like(x_pvals, dtype=bool)
                    y_significant = np.zeros_like(y_pvals, dtype=bool)

                    for comp_idx in range(n_components):
                        if self.settings.bootstrap_settings.loading_correction_method == "fdr":
                            _, x_pvals_corrected, _, _ = multipletests(
                                x_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='fdr_bh'
                            )
                            x_significant[:, comp_idx] = x_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        elif self.settings.bootstrap_settings.loading_correction_method == "bonferroni":
                            _, x_pvals_corrected, _, _ = multipletests(
                                x_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='bonferroni'
                            )
                            x_significant[:, comp_idx] = x_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        else:
                            x_significant[:, comp_idx] = x_pvals[:, comp_idx] < self.settings.bootstrap_settings.loading_alpha

                        if self.settings.bootstrap_settings.loading_correction_method == "fdr":
                            _, y_pvals_corrected, _, _ = multipletests(
                                y_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='fdr_bh'
                            )
                            y_significant[:, comp_idx] = y_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        elif self.settings.bootstrap_settings.loading_correction_method == "bonferroni":
                            _, y_pvals_corrected, _, _ = multipletests(
                                y_pvals[:, comp_idx],
                                alpha=self.settings.bootstrap_settings.loading_alpha,
                                method='bonferroni'
                            )
                            y_significant[:, comp_idx] = y_pvals_corrected < self.settings.bootstrap_settings.loading_alpha
                        else:
                            y_significant[:, comp_idx] = y_pvals[:, comp_idx] < self.settings.bootstrap_settings.loading_alpha

                    structural_loadings['x_significant'] = x_significant
                    structural_loadings['y_significant'] = y_significant

                print("✓ Bootstrap confidence intervals computed")

        results = {
            'correlations': correlations,
            'transformed_data': transformed,
            'weights': weights,
            'structural_loadings': structural_loadings,
            'x_feature_names': getattr(self, 'x_feature_columns_', x_processed.columns.tolist()),
            'y_feature_names': getattr(self, 'y_feature_columns_', y_processed.columns.tolist()),
            'n_components': len(correlations)
        }

        return results

    def _apply_feature_selection_and_transforms(self, x_data: pd.DataFrame, y_data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Legacy method - feature selection is now handled by FeatureProcessor.
        This method now just returns the data as-is since preprocessing is done upstream.
        """
        # Feature selection and transformations are now handled by FeatureProcessor
        return x_data, y_data
        
        x_processed = x_data.copy()
        y_processed = y_data.copy()
        
        # CRITICAL FIX: Separate features from covariates before feature selection
        # Get covariate column names if they exist
        x_covariate_columns = getattr(self, 'x_covariate_columns_', [])
        y_covariate_columns = getattr(self, 'y_covariate_columns_', [])
        
        # Separate actual features from covariates
        x_feature_columns = [col for col in x_data.columns if col not in x_covariate_columns]
        y_feature_columns = [col for col in y_data.columns if col not in y_covariate_columns]
        
        x_features_only = x_data[x_feature_columns] if x_feature_columns else x_data
        y_features_only = y_data[y_feature_columns] if y_feature_columns else y_data
        x_covariates_only = x_data[x_covariate_columns] if x_covariate_columns else pd.DataFrame(index=x_data.index)
        y_covariates_only = y_data[y_covariate_columns] if y_covariate_columns else pd.DataFrame(index=y_data.index)
        
        # Note: Data transformations and feature selection are now handled by FeatureProcessor
        # Use features as-is since preprocessing is done upstream
        x_features_selected = x_features_only
        y_features_selected = y_features_only
        self.selected_x_features_ = x_features_selected.columns.tolist()
        self.selected_y_features_ = y_features_selected.columns.tolist()
        
        # Recombine selected features with covariates
        if not x_covariates_only.empty:
            x_processed = pd.concat([x_features_selected, x_covariates_only], axis=1)
        else:
            x_processed = x_features_selected
            
        if not y_covariates_only.empty:
            y_processed = pd.concat([y_features_selected, y_covariates_only], axis=1)
        else:
            y_processed = y_features_selected
        
        return x_processed, y_processed
    
    def _apply_fold_specific_preprocessing(self, x_features: pd.DataFrame, y_features: pd.DataFrame, 
                                          x_covariate_columns: List[str], y_covariate_columns: List[str],
                                          x_combined: Optional[pd.DataFrame] = None, 
                                          y_combined: Optional[pd.DataFrame] = None,
                                          fitted_transformers: Optional[Dict] = None) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Apply preprocessing that should vary by CV fold: residualization and standardization.
        Feature selection should have already been applied.
        
        Args:
            x_features: X features (post-feature-selection)
            y_features: Y features (post-feature-selection) 
            x_covariate_columns: Names of X covariate columns
            y_covariate_columns: Names of Y covariate columns
            x_combined: Original X data with covariates (needed for residualization)
            y_combined: Original Y data with covariates (needed for residualization)
            fitted_transformers: Pre-fitted transformers (for validation data)
        
        Returns:
            Tuple of (x_processed, y_processed, fitted_transformers)
        """
        x_processed = x_features.copy()
        y_processed = y_features.copy()
        
        # Initialize transformers dict if not provided
        if fitted_transformers is None:
            fitted_transformers = {}
        
        # Apply residualization if covariates are present
        if x_covariate_columns and x_combined is not None:
            # Combine features with covariates for residualization
            x_for_residual = pd.concat([x_features, x_combined[x_covariate_columns]], axis=1)
            
            if 'x_residualizer' in fitted_transformers:
                # Use pre-fitted transformer (for validation data)
                x_processed = fitted_transformers['x_residualizer'].transform(x_for_residual)
            else:
                # Fit new transformer (for training data)
                residualizer = CovariateResidualizer(x_covariate_columns)
                residualizer.fit(x_for_residual)
                x_processed = residualizer.transform(x_for_residual)
                fitted_transformers['x_residualizer'] = residualizer
        
        if y_covariate_columns and y_combined is not None:
            y_for_residual = pd.concat([y_features, y_combined[y_covariate_columns]], axis=1)
            
            if 'y_residualizer' in fitted_transformers:
                # Use pre-fitted transformer (for validation data)
                y_processed = fitted_transformers['y_residualizer'].transform(y_for_residual)
            else:
                # Fit new transformer (for training data)
                residualizer = CovariateResidualizer(y_covariate_columns)
                residualizer.fit(y_for_residual)
                y_processed = residualizer.transform(y_for_residual)
                fitted_transformers['y_residualizer'] = residualizer
        
        # Note: Pre-PCA is now handled by FeatureProcessor before CCA
        # Skip pre-PCA step as it's done upstream
        
        # Apply standardization (always required for CCA)
        # Note: input_preprocessing removed, standardization always applied
        if 'standardizers' in fitted_transformers:
                # Validation: drop zero-variance and non-finite columns discovered during training
                zero_var_x = fitted_transformers.get('zero_var_x', [])
                zero_var_y = fitted_transformers.get('zero_var_y', [])
                nonfinite_drop_x = fitted_transformers.get('nonfinite_drop_x', [])
                nonfinite_drop_y = fitted_transformers.get('nonfinite_drop_y', [])
                cols_to_drop_x = [c for c in (zero_var_x + nonfinite_drop_x) if c in x_processed.columns]
                cols_to_drop_y = [c for c in (zero_var_y + nonfinite_drop_y) if c in y_processed.columns]
                if cols_to_drop_x:
                    x_processed = x_processed.drop(columns=cols_to_drop_x)
                if cols_to_drop_y:
                    y_processed = y_processed.drop(columns=cols_to_drop_y)

                # Use pre-fitted standardizers
                x_scaler, y_scaler = fitted_transformers['standardizers']
                x_numeric_cols = x_processed.select_dtypes(include=[np.number]).columns
                y_numeric_cols = y_processed.select_dtypes(include=[np.number]).columns

                if len(x_numeric_cols) > 0:
                    x_processed[x_numeric_cols] = x_scaler.transform(x_processed[x_numeric_cols])
                if len(y_numeric_cols) > 0:
                    y_processed[y_numeric_cols] = y_scaler.transform(y_processed[y_numeric_cols])
        else:
                # Training: detect and drop zero-variance columns to avoid NaN/Inf during scaling
                x_numeric_cols = x_processed.select_dtypes(include=[np.number]).columns.tolist()
                y_numeric_cols = y_processed.select_dtypes(include=[np.number]).columns.tolist()

                # Compute std with ddof=0 and mark zero or NaN std columns
                zero_var_x = []
                zero_var_y = []
                if x_numeric_cols:
                    x_std = x_processed[x_numeric_cols].std(ddof=0)
                    zero_var_x = [c for c in x_numeric_cols if (pd.isna(x_std[c]) or float(x_std[c]) == 0.0)]
                    if zero_var_x:
                        x_processed = x_processed.drop(columns=zero_var_x)
                if y_numeric_cols:
                    y_std = y_processed[y_numeric_cols].std(ddof=0)
                    zero_var_y = [c for c in y_numeric_cols if (pd.isna(y_std[c]) or float(y_std[c]) == 0.0)]
                    if zero_var_y:
                        y_processed = y_processed.drop(columns=zero_var_y)

                # Store zero-variance columns to ensure consistent validation processing
                fitted_transformers['zero_var_x'] = zero_var_x
                fitted_transformers['zero_var_y'] = zero_var_y

                # Recompute numeric columns after dropping
                x_numeric_cols = x_processed.select_dtypes(include=[np.number]).columns
                y_numeric_cols = y_processed.select_dtypes(include=[np.number]).columns

                x_scaler = StandardScaler()
                y_scaler = StandardScaler()

                if len(x_numeric_cols) > 0:
                    x_processed[x_numeric_cols] = x_scaler.fit_transform(x_processed[x_numeric_cols])
                if len(y_numeric_cols) > 0:
                    y_processed[y_numeric_cols] = y_scaler.fit_transform(y_processed[y_numeric_cols])

                fitted_transformers['standardizers'] = (x_scaler, y_scaler)

        # Final guards: drop any columns or rows with non-finite values
        # Columns (training): detect and store; (validation): apply stored
        if 'nonfinite_drop_x' in fitted_transformers:
            nf_x_cols = [c for c in fitted_transformers.get('nonfinite_drop_x', []) if c in x_processed.columns]
            nf_y_cols = [c for c in fitted_transformers.get('nonfinite_drop_y', []) if c in y_processed.columns]
            if nf_x_cols:
                x_processed = x_processed.drop(columns=nf_x_cols)
            if nf_y_cols:
                y_processed = y_processed.drop(columns=nf_y_cols)
        else:
            x_num = x_processed.select_dtypes(include=[np.number])
            y_num = y_processed.select_dtypes(include=[np.number])
            nf_x_cols = [c for c in x_num.columns if not np.isfinite(x_num[c].to_numpy()).all()]
            nf_y_cols = [c for c in y_num.columns if not np.isfinite(y_num[c].to_numpy()).all()]
            if nf_x_cols:
                x_processed = x_processed.drop(columns=nf_x_cols)
            if nf_y_cols:
                y_processed = y_processed.drop(columns=nf_y_cols)
            fitted_transformers['nonfinite_drop_x'] = nf_x_cols
            fitted_transformers['nonfinite_drop_y'] = nf_y_cols

        # Rows: compute row-wise finite masks and drop misbehaving rows in lockstep
        x_num = x_processed.select_dtypes(include=[np.number])
        y_num = y_processed.select_dtypes(include=[np.number])
        x_finite_mask = np.isfinite(x_num.to_numpy()).all(axis=1) if not x_num.empty else np.ones(len(x_processed), dtype=bool)
        y_finite_mask = np.isfinite(y_num.to_numpy()).all(axis=1) if not y_num.empty else np.ones(len(y_processed), dtype=bool)
        keep_mask = x_finite_mask & y_finite_mask
        rows_dropped = len(x_processed) - keep_mask.sum()
        if rows_dropped > 0:
            # Align indices and keep only good rows
            x_processed = x_processed.loc[keep_mask]
            y_processed = y_processed.loc[keep_mask]
            print(f"Fold preprocessing: dropped {rows_dropped} rows with non-finite values")

        # Log fold preprocessing summary
        zero_var_x = fitted_transformers.get('zero_var_x', [])
        zero_var_y = fitted_transformers.get('zero_var_y', [])
        nf_x_cols = fitted_transformers.get('nonfinite_drop_x', [])
        nf_y_cols = fitted_transformers.get('nonfinite_drop_y', [])
        if zero_var_x or zero_var_y or nf_x_cols or nf_y_cols or rows_dropped > 0:
            print(f"Fold preprocessing summary: zero_var_x={len(zero_var_x)}, zero_var_y={len(zero_var_y)}, "
                  f"nonfinite_x={len(nf_x_cols)}, nonfinite_y={len(nf_y_cols)}, rows_dropped={rows_dropped}")
            print(f"Final shapes: X={x_processed.shape}, Y={y_processed.shape}")

        return x_processed, y_processed, fitted_transformers

def _run_multiview_cca(x_data: pd.DataFrame,
                      y_data: pd.DataFrame,
                      settings: 'CCASettings',
                      progress_callback=None) -> Dict[str, Any]:
    """
    Handle multiview CCA using the legacy implementation for compatibility.

    This function provides a bridge to multiview functionality
    while using the modern CCASettings configuration.
    """
    from cca_zoo.linear import MCCA
    from sklearn.impute import SimpleImputer

    try:
        # Extract views from settings
        if not settings.views or not isinstance(settings.views, list):
            raise ValueError("Valid 'views' must be provided for multiview CCA in settings.views")

        # For multiview, x_data should contain multiple views
        # This is a simplified implementation - in practice, you'd need more sophisticated view handling
        views_data = []

        # Handle case where views are feature names vs dataset keys
        if isinstance(settings.views[0], str):
            # Views are dataset keys - x_data should be a dict-like structure
            # For now, assume x_data and y_data are the views
            views_data = [x_data.values, y_data.values]
        else:
            # Views are lists of column names
            for view_cols in settings.views:
                if isinstance(view_cols, list):
                    # Extract columns from x_data
                    view_data = x_data[view_cols].values
                    views_data.append(view_data)

        # Process views with imputation and standardization
        processed_views = []
        imputer = SimpleImputer(strategy='mean')

        for view_data in views_data:
            # Handle missing values
            view_imputed = imputer.fit_transform(view_data)
            # Standardize
            view_mean = np.mean(view_imputed, axis=0)
            view_std = np.std(view_imputed, axis=0)
            view_scaled = (view_imputed - view_mean) / (view_std + 1e-10)
            processed_views.append(view_scaled)

        # Determine number of components
        max_possible_components = min(view.shape[1] for view in processed_views)
        n_components = settings.model_settings.n_components
        if n_components > max_possible_components:
            print(f"Warning: Reducing n_components from {n_components} to {max_possible_components} for multiview CCA")
            n_components = max_possible_components

        # Initialize and fit MCCA
        model = MCCA(latent_dimensions=n_components)
        model.fit(processed_views)
        transformed_data = model.transform(processed_views)

        return {
            'correlations': model.score(processed_views),
            'weights': model.weights_,
            'transformed_data': transformed_data,
            'view_names': settings.views,
            'n_components': n_components,
            'processing_info': {
                'multiview_cca_used': True,
                'n_views': len(processed_views),
                'n_samples': processed_views[0].shape[0]
            }
        }

    except Exception as e:
        print(f"Error in multiview CCA: {e}")
        raise

def run_cca(x_data: pd.DataFrame,
           y_data: pd.DataFrame,
           settings: 'CCASettings',
           x_covariates: Optional[pd.DataFrame] = None,
           y_covariates: Optional[pd.DataFrame] = None,
           stratification_variable: Optional[pd.DataFrame] = None,
           progress_callback: Optional[Callable] = None,
           preprocessing_output_dir: Optional[str] = None,
           feature_processing_configs=None) -> Dict[str, Any]:
    """
    Run Canonical Correlation Analysis with modern CCASettings configuration.

    This unified function replaces both legacy run_cca() and run_enhanced_cca().
    All configuration is done through the CCASettings object - no legacy parameters.

    Args:
        x_data: X feature matrix (pandas DataFrame)
        y_data: Y feature matrix (pandas DataFrame)
        settings: CCA configuration object (CCASettings)
        x_covariates: Optional X covariates for residualization
        y_covariates: Optional Y covariates for residualization
        stratification_variable: Optional variable for stratified CV/permutation testing
        progress_callback: Optional callback function for progress updates
        preprocessing_output_dir: Optional directory to save intermediate preprocessing CSVs
        feature_processing_configs: Optional list of FeatureProcessingConfig objects

    Returns:
        Dictionary containing CCA results and metadata
    """
    try:
        # Handle multiview CCA case
        if settings.is_multiview:
            return _run_multiview_cca(x_data, y_data, settings, progress_callback)

        # Handle legacy numpy array inputs by converting to DataFrames
        if isinstance(x_data, np.ndarray):
            x_data = pd.DataFrame(x_data, columns=[f'x_feature_{i}' for i in range(x_data.shape[1])])
        if isinstance(y_data, np.ndarray):
            y_data = pd.DataFrame(y_data, columns=[f'y_feature_{i}' for i in range(y_data.shape[1])])

        # CRITICAL: Input validation
        if x_data.empty or y_data.empty:
            raise ValueError("Input data cannot be empty")

        if len(x_data) != len(y_data):
            raise ValueError(f"X data ({len(x_data)} samples) and Y data ({len(y_data)} samples) must have the same number of samples")
        
        # Check for sufficient samples
        n_samples = len(x_data)
        n_components = settings.model_settings.n_components
        if n_samples < 10:
            raise ValueError(f"Need at least 10 samples for CCA, got {n_samples}")
        
        # Validate n_components
        max_components = min(n_samples - 1, x_data.shape[1], y_data.shape[1])
        if n_components > max_components:
            print(f"Warning: Reducing n_components from {n_components} to {max_components} based on data constraints")
            settings.model_settings.n_components = max_components
        
        # Check for numeric data
        x_numeric_cols = x_data.select_dtypes(include=[np.number]).columns
        y_numeric_cols = y_data.select_dtypes(include=[np.number]).columns
        
        if len(x_numeric_cols) == 0:
            raise ValueError("X data contains no numeric columns")
        if len(y_numeric_cols) == 0:
            raise ValueError("Y data contains no numeric columns")
        
        # Validate regularization ranges
        if settings.optimization_settings.optimize_regularization:
            reg_x_range = settings.optimization_settings.regularization_x_range
            reg_y_range = settings.optimization_settings.regularization_y_range
            
            if reg_x_range[0] <= 0 or reg_x_range[1] <= 0:
                raise ValueError("Regularization parameters must be positive")
            if reg_x_range[0] >= reg_x_range[1]:
                raise ValueError("Regularization range minimum must be less than maximum")
        
        # Validate group variable for statistical testing
        if (settings.statistical_testing.enabled and 
            settings.statistical_testing.respect_groups and
            settings.statistical_testing.group_variable and
            stratification_variable is not None):
            
            if settings.statistical_testing.group_variable not in stratification_variable.columns:
                print(f"Warning: Group variable '{settings.statistical_testing.group_variable}' not found. Using simple permutation.")
                settings.statistical_testing.respect_groups = False
            else:
                groups = stratification_variable[settings.statistical_testing.group_variable].values
                unique_groups, group_counts = np.unique(groups, return_counts=True)
                min_group_size = np.min(group_counts)
                if min_group_size < 2:
                    print(f"Warning: Some groups have only 1 sample. Consider using simple permutation.")
                if len(unique_groups) < 2:
                    print("Warning: Only one group found. Using simple permutation.")
                    settings.statistical_testing.respect_groups = False
        
        # Create and fit pipeline
        pipeline = EnhancedCCAPipeline(settings, feature_processing_configs=feature_processing_configs)
        pipeline.fit(
            x_data=x_data,
            y_data=y_data,
            x_covariates=x_covariates,
            y_covariates=y_covariates,
            progress_callback=progress_callback,
            preprocessing_output_dir=preprocessing_output_dir
        )
        
        # Extract results - CRITICAL FIX: Pass combined data with covariates
        x_combined_for_results = x_data.copy()
        y_combined_for_results = y_data.copy()
        
        if x_covariates is not None:
            x_combined_for_results = pd.concat([x_data, x_covariates], axis=1)
        if y_covariates is not None:
            y_combined_for_results = pd.concat([y_data, y_covariates], axis=1)
            
        cca_results = pipeline.get_results(x_combined_for_results, y_combined_for_results)
        
        # Run statistical testing if enabled
        statistical_results = None
        if settings.statistical_testing.enabled:
            if progress_callback:
                progress_callback(90, "Running statistical significance testing...")
            
            # Get stratification variable if specified
            groups = None
            if (settings.statistical_testing.respect_groups and 
                settings.statistical_testing.group_variable and
                stratification_variable is not None):
                if settings.statistical_testing.group_variable in stratification_variable.columns:
                    groups = stratification_variable[settings.statistical_testing.group_variable].values
                    unique_groups = np.unique(groups)
                else:
                    print(f"Warning: Group variable '{settings.statistical_testing.group_variable}' not found in stratification data")
            
            statistical_results = _run_statistical_testing(
                pipeline, x_combined_for_results, y_combined_for_results, settings, x_covariates, y_covariates, groups
            )
            
            if progress_callback:
                progress_callback(95, "Statistical testing complete")
        
        if progress_callback:
            progress_callback(98, "Preparing results...")
        
        # Compile final results
        results = {
            'cca_results': cca_results,
            'optimization_results': {
                'best_parameters': pipeline.best_params_,
                'hyperparameter_optimization_enabled': settings.hyperparameter_optimization_enabled
            },
            'statistical_results': statistical_results,
            'settings_used': settings.to_dict(),
            'processing_info': {
                'enhanced_cca_used': True,
                'preprocessing_applied': True,  # Now handled by FeatureProcessor
                'hyperparameter_optimization': settings.hyperparameter_optimization_enabled,
                'statistical_testing': settings.statistical_testing.enabled,
                'n_samples': len(x_data),
                'n_features_x': x_data.shape[1],
                'n_features_y': y_data.shape[1]
            }
        }
        
        return results
        
    except Exception as e:
        print(f"Error in enhanced CCA: {e}")
        import traceback
        traceback.print_exc()
        raise

# Legacy compatibility function for old parameter style
def run_enhanced_cca(*args, **kwargs):
    """
    DEPRECATED: Legacy compatibility function.

    This function is deprecated. Use run_cca() with CCASettings instead.
    This wrapper is provided for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "run_enhanced_cca() is deprecated. Use run_cca() with CCASettings instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return run_cca(*args, **kwargs)

def _run_statistical_testing(fitted_pipeline: EnhancedCCAPipeline,
                            x_data: pd.DataFrame, 
                            y_data: pd.DataFrame,
                            settings: 'CCASettings',
                            x_covariates: Optional[pd.DataFrame] = None,
                            y_covariates: Optional[pd.DataFrame] = None,
                            groups: Optional[np.ndarray] = None) -> Dict:
    """
    Run efficient permutation testing using the fitted pipeline's optimal parameters.
    
    This follows the R script methodology:
    1. Use the already-optimized hyperparameters from the fitted pipeline
    2. For each permutation, just shuffle Y data and fit with fixed parameters
    3. Extract canonical correlations and compare to observed
    """
    if not CCA_ZOO_AVAILABLE:
        return {"error": "CCA-Zoo not available for statistical testing"}
    
    print("Running permutation testing...")
    if groups is not None:
        print(f"Using site-aware permutation with {len(np.unique(groups))} sites")
    else:
        print("Using simple (non-site-aware) permutation")

    # Get observed results from the fitted pipeline
    observed_results = fitted_pipeline.get_results(x_data, y_data)
    observed_correlations = observed_results['correlations']
    n_components = len(observed_correlations)
    n_permutations = settings.statistical_testing.n_permutations
    
    # Get optimal parameters from the fitted pipeline (no re-optimization needed!)
    if hasattr(fitted_pipeline, 'best_params_') and fitted_pipeline.best_params_ is not None:
        optimal_params = fitted_pipeline.best_params_.copy()
        optimal_params['cca_type'] = fitted_pipeline.settings.model_settings.cca_type
        # Ensure n_components is included (may not be in best_params_ from HPO)
        optimal_params['n_components'] = n_components
    else:
        # Use the model settings directly
        optimal_params = {
            'cca_type': settings.model_settings.cca_type,
            'n_components': n_components,
            'regularization_x': settings.model_settings.regularization_x,
            'regularization_y': settings.model_settings.regularization_y,
            'sparsity_penalty_x': settings.model_settings.sparsity_penalty_x,
            'sparsity_penalty_y': settings.model_settings.sparsity_penalty_y
        }
    
    # Initialize storage for permutation results
    perm_correlations = np.zeros((n_permutations, n_components))
    # Store cumulative log-likelihood for each component (MATLAB style)
    perm_log_likelihood = np.zeros((n_permutations, n_components))
    
    # CRITICAL FIX: x_data and y_data already contain covariates (combined in run_enhanced_cca)
    # So we don't need to concatenate again
    x_combined = x_data.copy()
    y_combined = y_data.copy()

    # Apply the same preprocessing as the fitted pipeline (once, outside the loop)
    # Check if we're using the new preprocessing architecture
    if hasattr(fitted_pipeline, 'x_processing_plan_') and hasattr(fitted_pipeline, 'y_processing_plan_'):
        # New preprocessing architecture: apply only inside-CV steps
        # Outside-CV steps were already applied by fusion.py before the original fit()
        # x_combined and y_combined already have outside-CV steps applied

        # Apply inside-CV steps (residualize, standardize, PCA in with_hpo mode)
        # Use fit=False to apply the transformers fitted during original training
        x_processed = apply_inside_cv_steps(fitted_pipeline.x_processing_plan_, x_combined, fit=False, other_view=y_combined)
        y_processed = apply_inside_cv_steps(fitted_pipeline.y_processing_plan_, y_combined, fit=False, other_view=x_combined)

        # Extract feature columns (excluding covariates)
        x_covariate_columns = getattr(fitted_pipeline, 'x_covariate_columns_', [])
        y_covariate_columns = getattr(fitted_pipeline, 'y_covariate_columns_', [])

        x_feature_columns = [col for col in x_processed.columns if col not in x_covariate_columns]
        y_feature_columns = [col for col in y_processed.columns if col not in y_covariate_columns]

        x_processed = x_processed[x_feature_columns] if x_feature_columns else x_processed
        y_processed = y_processed[y_feature_columns] if y_feature_columns else y_processed
    else:
        # Legacy preprocessing architecture
        # STEP 1: Apply feature selection and transforms (same as training)
        if hasattr(fitted_pipeline, 'x_feature_columns_') and hasattr(fitted_pipeline, 'y_feature_columns_'):
            # Use the same feature selection as was applied during training
            x_features_only = x_combined[fitted_pipeline.x_feature_columns_]
            y_features_only = y_combined[fitted_pipeline.y_feature_columns_]
        else:
            # Fallback: apply feature selection fresh (not ideal but prevents crashes)
            x_features_only, y_features_only = fitted_pipeline._apply_feature_selection_and_transforms(x_combined, y_combined)

        # STEP 2: Apply fold-specific preprocessing (residualization + standardization)
        x_processed, y_processed, fitted_transformers = fitted_pipeline._apply_fold_specific_preprocessing(
            x_features_only, y_features_only,
            fitted_pipeline.x_covariate_columns_, fitted_pipeline.y_covariate_columns_,
            x_combined, y_combined
        )
    
    # Run permutations efficiently on the PROCESSED data
    failed_permutations = 0
    for perm_i in range(n_permutations):
        # Create permuted Y FEATURES only (after preprocessing)
        # This way we don't mess up the covariate residualization
        perm_random_state = settings.statistical_testing.random_state
        if perm_random_state is not None:
            perm_random_state = perm_random_state + perm_i  # Unique seed per permutation
        
        y_processed_perm = _create_permuted_data(y_processed, groups, settings.statistical_testing.test_type, perm_random_state)
        
        # Fit CCA with FIXED optimal parameters (no optimization!)
        try:
            # Ensure covariance regularization parameter is included for permutation testing
            perm_params = optimal_params.copy()
            perm_params['use_matlab_style'] = (fitted_pipeline.settings.model_settings.regularization_method == "covariance")

            perm_estimator = CCAEstimator(**perm_params)
            perm_estimator.fit(x_processed, y_processed_perm)

            # Get canonical correlations from the fitted model (SVD singular values)
            # This matches MATLAB's approach
            perm_cca_model = perm_estimator.cca_model_
            if hasattr(perm_cca_model, 'correlations_'):
                perm_corr = list(perm_cca_model.correlations_)
            else:
                # Fallback: compute from variates
                transformed = perm_estimator.transform(x_processed, y_processed_perm)
                perm_corr = []
                for i in range(min(transformed[0].shape[1], transformed[1].shape[1])):
                    x_variate = transformed[0][:, i]
                    y_variate = transformed[1][:, i]

                    if np.var(x_variate) == 0 or np.var(y_variate) == 0:
                        perm_corr.append(0.0)
                        continue

                    corr_matrix = np.corrcoef(x_variate, y_variate)
                    corr = corr_matrix[0, 1]

                    if np.isnan(corr) or np.isinf(corr):
                        perm_corr.append(0.0)
                    else:
                        perm_corr.append(abs(corr))
            
            # Store correlations and cumulative log-likelihood (MATLAB style)
            perm_correlations[perm_i, :len(perm_corr)] = perm_corr
            perm_lW = _compute_log_likelihood(np.array(perm_corr))
            perm_log_likelihood[perm_i, :len(perm_lW)] = perm_lW

        except Exception as e:
            print(f"Warning: Permutation {perm_i+1} failed: {e}")
            failed_permutations += 1
            # Fill with zeros for failed permutation
            perm_correlations[perm_i, :] = 0.0
            perm_log_likelihood[perm_i, :] = 0.0
            
            # Skip if too many permutations fail
            if failed_permutations > n_permutations * 0.5:
                print(f"Error: More than 50% of permutations failed ({failed_permutations}/{perm_i+1}). Stopping statistical testing.")
                return {
                    "error": f"Statistical testing failed: {failed_permutations} out of {perm_i+1} permutations failed",
                    "n_successful_permutations": perm_i + 1 - failed_permutations
                }
        
        if (perm_i + 1) % 10 == 0:
            print(f"  Completed {perm_i + 1}/{n_permutations} permutations")
    
    if failed_permutations > 0:
        print(f"Warning: {failed_permutations} out of {n_permutations} permutations failed")
    
    # Calculate p-values
    try:
        observed_log_likelihood = _compute_log_likelihood(observed_correlations)
        p_values = _calculate_permutation_pvalues(
            observed_correlations, observed_log_likelihood,
            perm_correlations, perm_log_likelihood, settings.statistical_testing
        )

        return {
            'p_values': p_values,
            'observed_correlations': observed_correlations,
            'observed_log_likelihood': observed_log_likelihood,  # Cumulative lW for each component
            'permutation_correlations': perm_correlations,
            'permutation_log_likelihood': perm_log_likelihood,  # Cumulative lW for each permutation
            'n_permutations': n_permutations,
            'n_successful_permutations': n_permutations - failed_permutations,
            'n_failed_permutations': failed_permutations
        }
        
    except Exception as e:
        print(f"Error calculating p-values: {e}")
        return {
            "error": f"Failed to calculate p-values: {e}",
            "n_successful_permutations": n_permutations - failed_permutations
        }


def _create_permuted_data(data: pd.DataFrame, groups: Optional[np.ndarray], test_type: str, random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Create permuted version of data respecting group structure.
    
    Args:
        data: Original data to permute
        groups: Group labels for site-aware permutation
        test_type: Type of permutation ('shuffle_x', 'shuffle_y', 'shuffle_both')
        random_state: Random seed for reproducibility
    
    Returns:
        Permuted data
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    permuted_data = data.copy()
    
    if groups is not None:
        # Site-aware permutation - permute within each group (site)
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            group_mask = groups == group
            group_indices = np.where(group_mask)[0]
            
            if len(group_indices) > 1:  # Only permute if more than 1 subject in group
                # Permute indices within this group
                permuted_group_indices = np.random.permutation(group_indices)
                
                # Apply permutation to data
                permuted_data.iloc[group_indices] = permuted_data.iloc[permuted_group_indices].values
            # If only 1 subject in group, leave unchanged (no permutation possible)
    else:
        # Simple permutation across all samples
        n_samples = len(permuted_data)
        permuted_indices = np.random.permutation(n_samples)
        permuted_data = permuted_data.iloc[permuted_indices].reset_index(drop=True)
    
    return permuted_data


def _compute_log_likelihood(correlations: np.ndarray) -> np.ndarray:
    """
    Compute cumulative log-likelihood statistic from canonical correlations (Wilks' Lambda).

    This implements MATLAB's PALM toolbox calculation:
    lW = -fliplr(cumsum(fliplr(log(1-cc.^2))))

    Returns an array of cumulative log-likelihoods, one per component.
    For component i, this is the log-likelihood of component i and all subsequent components.

    Args:
        correlations: Array of canonical correlations

    Returns:
        Array of cumulative log-likelihoods (Wilks' Lambda values)
    """
    if len(correlations) == 0:
        return np.array([])

    # Ensure correlations are valid (between 0 and 1)
    correlations = np.clip(correlations, 1e-10, 1 - 1e-10)

    # Calculate log terms: log(1 - r^2)
    log_terms = np.log(1 - correlations**2)

    # Compute cumulative sum from right to left (like MATLAB's fliplr(cumsum(fliplr(...))))
    # This gives us Wilks' Lambda for each component
    cumulative_lW = -np.cumsum(log_terms[::-1])[::-1]

    return cumulative_lW


def _calculate_permutation_pvalues(observed_correlations: np.ndarray,
                                 observed_log_likelihood: np.ndarray,
                                 perm_correlations: np.ndarray,
                                 perm_log_likelihood: np.ndarray,
                                 testing_settings) -> Dict:
    """
    Calculate p-values from permutation results with multiple comparisons correction.

    Implements MATLAB's methodology:
    - Tests canonical correlations directly for individual components
    - Uses (count + 1) / (n_permutations + 1) for unbiased p-value estimate
    - Strict greater-than comparison (>)
    - Applies cummax for FWER correction or Bonferroni

    Args:
        observed_correlations: Observed canonical correlations
        observed_log_likelihood: Observed cumulative log-likelihood (array, one per component)
        perm_correlations: Permutation correlations (n_perm x n_components)
        perm_log_likelihood: Permutation cumulative log-likelihoods (n_perm x n_components)
        testing_settings: Statistical testing settings

    Returns:
        Dictionary with p-values and significance results
    """
    n_components = len(observed_correlations)
    n_permutations = perm_correlations.shape[0]

    # Calculate individual component p-values using MATLAB's approach:
    # - Use canonical correlations directly (not Wilks' Lambda)
    # - Strict greater-than comparison (>)
    # - Add +1 to numerator and denominator for unbiased estimate
    individual_p_uncorrected = np.zeros(n_components)

    for comp_i in range(n_components):
        # MATLAB: perm_p(i) = (sum(perm_r(:,i) > real_r(i)) + 1) / (n_perm + 1)
        count = np.sum(perm_correlations[:, comp_i] > observed_correlations[comp_i])
        individual_p_uncorrected[comp_i] = (count + 1) / (n_permutations + 1)

    # For aggregate test, use Wilks' Lambda (log-likelihood) - tests all components jointly
    # This is different from individual component tests
    aggregate_count = np.sum(perm_log_likelihood[:, 0] >= observed_log_likelihood[0])
    aggregate_p_uncorrected = (aggregate_count + 1) / (n_permutations + 1) if n_components > 0 else 1.0

    # Apply multiple comparisons correction
    if testing_settings.multiple_comparisons_correction == "familywise":
        # Familywise error rate correction (MATLAB's cummax)
        individual_p_corrected = _apply_fwer_correction(individual_p_uncorrected)
    elif testing_settings.multiple_comparisons_correction == "bonferroni":
        # Bonferroni correction
        individual_p_corrected = np.minimum(individual_p_uncorrected * n_components, 1.0)
    else:
        # No correction
        individual_p_corrected = individual_p_uncorrected

    # Determine significance
    alpha = testing_settings.alpha
    individual_significant = individual_p_corrected < alpha
    aggregate_significant = aggregate_p_uncorrected < alpha

    return {
        'individual': {
            'p_uncorrected': individual_p_uncorrected,
            'p_corrected': individual_p_corrected,
            'significant': individual_significant
        },
        'aggregate': {
            'p_value': aggregate_p_uncorrected,
            'significant': aggregate_significant
        },
        'alpha': alpha,
        'correction_method': testing_settings.multiple_comparisons_correction
    }


def _apply_fwer_correction(p_values: np.ndarray) -> np.ndarray:
    """
    Apply familywise error rate correction using cumulative maximum (closure method).

    This implements MATLAB's cummax function: pfwer = cummax(punc)
    This is the closure method for FWER control, which ensures that if component i
    is significant, all earlier components are also significant.

    Args:
        p_values: Uncorrected p-values

    Returns:
        FWER-corrected p-values using cumulative maximum
    """
    # Apply cumulative maximum (MATLAB's cummax)
    # This ensures monotonicity: p_corrected[i] >= p_corrected[i-1]
    return np.maximum.accumulate(p_values)