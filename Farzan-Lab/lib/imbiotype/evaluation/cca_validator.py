"""
CCAValidator - Handles CCA validation within nested CV loop.
"""

from typing import Dict, List, Optional, Any
import numpy as np
from dataclasses import dataclass

from ..ml_functions.enhanced_cca import CCAEstimator
from ..processing.feature_processor import FeatureProcessor


@dataclass
class CCAFoldResult:
    """Results from a single CV fold."""
    train_correlations: np.ndarray  # (n_components,)
    test_correlations: np.ndarray   # (n_components,)
    fitted_model: Dict[str, Any]
    best_lambda_x: Optional[float] = None
    best_lambda_y: Optional[float] = None


class CCAValidator:
    """
    Handles CCA validation within nested CV loop.

    Responsible for:
    - Fitting CCA on train data
    - Evaluating on test data
    - Managing inside-CV preprocessing steps
    - Aggregating results across folds/repeats
    """

    def __init__(self, feature_processing_configs: Optional[List] = None):
        """
        Initialize CCA validator.

        Args:
            feature_processing_configs: List of FeatureProcessingConfig for inside-CV steps
        """
        self.feature_processing_configs = feature_processing_configs or []

    def validate_fold(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray,
        cca_type: str = "regularized",
        n_components: int = 2,
        regularization_x: float = 0.1,
        regularization_y: float = 0.1,
        use_matlab_style: bool = True,
        run_hpo: bool = False,
        reduced_hpo_repeats: int = 10
    ) -> CCAFoldResult:
        """
        Run CCA on train, evaluate on test for one fold.

        Args:
            X_train: Training X features (n_train x n_features_x)
            Y_train: Training Y features (n_train x n_features_y)
            X_test: Test X features (n_test x n_features_x)
            Y_test: Test Y features (n_test x n_features_y)
            cca_type: Type of CCA ('regularized', 'classic', 'sparse')
            n_components: Number of canonical components
            regularization_x: Regularization for X (if run_hpo=False)
            regularization_y: Regularization for Y (if run_hpo=False)
            use_matlab_style: Use MATLAB-compatible rCCA
            run_hpo: Whether to run hyperparameter optimization
            reduced_hpo_repeats: Number of HPO repeats (reduced for efficiency)

        Returns:
            CCAFoldResult with train/test correlations and fitted model
        """
        # Apply inside-CV preprocessing if configured
        X_train_proc, Y_train_proc = X_train, Y_train
        X_test_proc, Y_test_proc = X_test, Y_test

        if self.feature_processing_configs:
            # Use fit/transform pattern: fit on train, transform both train and test
            # This prevents data leakage by fitting only on training data
            X_train_proc, X_test_proc = self._apply_preprocessing(
                X_train, X_test, self.feature_processing_configs, view='x'
            )
            Y_train_proc, Y_test_proc = self._apply_preprocessing(
                Y_train, Y_test, self.feature_processing_configs, view='y'
            )

        best_lambda_x = regularization_x
        best_lambda_y = regularization_y

        if run_hpo:
            # Run reduced HPO to find optimal regularization
            best_lambda_x, best_lambda_y = self._run_hpo(
                X_train_proc, Y_train_proc,
                cca_type=cca_type,
                n_components=n_components,
                use_matlab_style=use_matlab_style,
                n_repeats=reduced_hpo_repeats
            )

        # Create and fit CCA estimator
        estimator = CCAEstimator(
            cca_type=cca_type,
            n_components=n_components,
            regularization_x=best_lambda_x,
            regularization_y=best_lambda_y,
            use_matlab_style=use_matlab_style
        )

        estimator.fit(X_train_proc, Y_train_proc)

        # Get train correlations for ALL components (not just first)
        # This ensures shape consistency with test_correlations
        train_correlations = self._get_all_train_correlations(
            estimator, X_train_proc, Y_train_proc, n_components
        )

        # Get test correlations (already returns all components)
        test_correlations = estimator.evaluate_test_set(X_test_proc, Y_test_proc)

        # Get fitted model
        fitted_model = estimator.get_fitted_model()

        return CCAFoldResult(
            train_correlations=train_correlations,
            test_correlations=test_correlations,
            fitted_model=fitted_model,
            best_lambda_x=best_lambda_x,
            best_lambda_y=best_lambda_y
        )

    def _run_hpo(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cca_type: str,
        n_components: int,
        use_matlab_style: bool,
        n_repeats: int = 10
    ) -> tuple:
        """
        Run hyperparameter optimization to find best regularization.

        Uses reduced number of repeats for efficiency within nested CV.
        Applies inside-CV preprocessing per fold to prevent data leakage.

        Returns:
            (best_lambda_x, best_lambda_y)
        """
        from sklearn.model_selection import KFold

        # Lambda grid (same as in enhanced_cca.py)
        lambda_grid = [0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        best_score = -np.inf
        best_lambda_x = 0.1
        best_lambda_y = 0.1

        # Simplified grid search: same lambda for both views
        for lambda_val in lambda_grid:
            scores = []

            for repeat in range(n_repeats):
                kfold = KFold(n_splits=5, shuffle=True, random_state=42 + repeat)

                for train_idx, val_idx in kfold.split(X):
                    X_train = X[train_idx]
                    Y_train = Y[train_idx]
                    X_val = X[val_idx]
                    Y_val = Y[val_idx]

                    # Apply inside-CV preprocessing if configured
                    # This ensures HPO is optimizing for preprocessed data
                    if self.feature_processing_configs:
                        X_train, X_val = self._apply_preprocessing(
                            X_train, X_val, self.feature_processing_configs, view='x'
                        )
                        Y_train, Y_val = self._apply_preprocessing(
                            Y_train, Y_val, self.feature_processing_configs, view='y'
                        )

                    estimator = CCAEstimator(
                        cca_type=cca_type,
                        n_components=n_components,
                        regularization_x=lambda_val,
                        regularization_y=lambda_val,
                        use_matlab_style=use_matlab_style
                    )

                    try:
                        estimator.fit(X_train, Y_train)
                        score = estimator.score(X_val, Y_val)
                        if not np.isnan(score):
                            scores.append(score)
                    except Exception:
                        continue

            if scores:
                mean_score = np.mean(scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_lambda_x = lambda_val
                    best_lambda_y = lambda_val

        return best_lambda_x, best_lambda_y

    def _apply_preprocessing(
        self,
        train_data: np.ndarray,
        test_data: np.ndarray,
        configs: List,
        view: str = 'x'
    ) -> tuple:
        """
        Apply preprocessing using fit/transform pattern.

        Fits on train data, transforms both train and test to prevent data leakage.

        Args:
            train_data: Training data (n_train x n_features)
            test_data: Test data (n_test x n_features)
            configs: List of FeatureProcessingConfig objects
            view: Which view this is ('x' or 'y')

        Returns:
            Tuple of (train_processed, test_processed)
        """
        import pandas as pd
        from ..processing.feature_processor import (
            PreprocessingConfig, build_processing_plan,
            apply_inside_cv_steps
        )

        # Convert to DataFrame for processing
        train_df = pd.DataFrame(train_data)
        test_df = pd.DataFrame(test_data)

        # Find config for this view
        target_config = None
        for cfg in configs:
            if hasattr(cfg, 'feature_name'):
                # Match by view name in feature name
                if (view == 'x' and 'bio' in cfg.feature_name.lower()) or \
                   (view == 'y' and ('clinical' in cfg.feature_name.lower() or 'symptom' in cfg.feature_name.lower())):
                    target_config = cfg
                    break

        if target_config is None:
            # No matching config, return unchanged
            return train_data, test_data

        # Get preprocessing config
        if hasattr(target_config, 'preprocessing_config'):
            preproc_cfg = target_config.preprocessing_config
        else:
            return train_data, test_data

        # Build processing plan (inside-CV mode for evaluation)
        plan = build_processing_plan(preproc_cfg, mode="with_hpo")

        # Apply inside-CV steps: fit on train, transform both
        # Fit and transform train
        train_processed = train_df.copy()
        for step in plan.inside_cv_steps:
            step.fit(train_processed)
            train_processed = step.transform(train_processed)

        # Transform test using fitted steps
        test_processed = test_df.copy()
        for step in plan.inside_cv_steps:
            test_processed = step.transform(test_processed)

        return train_processed.values, test_processed.values

    def _get_all_train_correlations(
        self,
        estimator,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        n_components: int
    ) -> np.ndarray:
        """
        Get training correlations for all CCA components.

        The fitted CCA model stores correlations_ which are the canonical
        correlations from SVD. For consistency with test evaluation, we
        use the same approach: compute Pearson correlation of variates.

        Args:
            estimator: Fitted CCAEstimator
            X_train: Training X data
            Y_train: Training Y data
            n_components: Number of CCA components

        Returns:
            Array of training correlations (n_components,)
        """
        # Use stored correlations if available (from SVD decomposition)
        if hasattr(estimator, 'cca_model_') and hasattr(estimator.cca_model_, 'correlations_'):
            stored_corrs = estimator.cca_model_.correlations_
            if len(stored_corrs) >= n_components:
                return np.array(stored_corrs[:n_components])

        # Fallback: compute Pearson correlation of variates (same as test)
        # This ensures train/test metrics are computed the same way
        return estimator.evaluate_test_set(X_train, Y_train)

    def aggregate_results(
        self,
        fold_results: List[List[CCAFoldResult]],
        n_repeats: int,
        n_folds: int
    ) -> Dict[str, Any]:
        """
        Aggregate results across all folds and repeats.

        Args:
            fold_results: Nested list [repeat][fold] of CCAFoldResult
            n_repeats: Number of CV repeats
            n_folds: Number of folds per repeat

        Returns:
            Aggregated results dict
        """
        # Collect all correlations
        n_components = fold_results[0][0].train_correlations.shape[0]

        all_train_corrs = np.zeros((n_repeats, n_folds, n_components))
        all_test_corrs = np.zeros((n_repeats, n_folds, n_components))
        all_lambda_x = []
        all_lambda_y = []

        for r, repeat_results in enumerate(fold_results):
            for f, fold_result in enumerate(repeat_results):
                all_train_corrs[r, f, :] = fold_result.train_correlations[:n_components]
                all_test_corrs[r, f, :] = fold_result.test_correlations[:n_components]

                if fold_result.best_lambda_x is not None:
                    all_lambda_x.append(fold_result.best_lambda_x)
                    all_lambda_y.append(fold_result.best_lambda_y)

        # Compute summary statistics with NaN handling
        # NaN/inf can occur if CCA fails for some folds
        mean_train = np.nanmean(all_train_corrs[:, :, 0])  # First component
        mean_test = np.nanmean(all_test_corrs[:, :, 0])
        std_test = np.nanstd(all_test_corrs[:, :, 0])

        # Handle edge cases for generalization ratio
        if np.isnan(mean_train) or np.isnan(mean_test) or mean_train == 0:
            gen_ratio = 0.0
        else:
            gen_ratio = mean_test / mean_train

        return {
            'n_repeats': n_repeats,
            'n_folds': n_folds,
            'n_components': n_components,
            'train_correlations': all_train_corrs,
            'test_correlations': all_test_corrs,
            'mean_train_correlation': float(mean_train),
            'mean_test_correlation': float(mean_test),
            'std_test_correlation': float(std_test),
            'generalization_ratio': float(gen_ratio),
            'lambda_x_values': all_lambda_x if all_lambda_x else None,
            'lambda_y_values': all_lambda_y if all_lambda_y else None
        }
