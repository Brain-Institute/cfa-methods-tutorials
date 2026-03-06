"""
PipelineValidator - End-to-end CCA + Clustering validation.

Coordinates CCA validation and clustering validation using the same
train/test splits, building consensus matrices and tracking subject
consistency across the full pipeline.
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
import numpy as np
from joblib import Parallel, delayed

from .config import EvaluationConfig
from .splitter import TrainTestSplitter
from .cca_validator import CCAValidator, CCAFoldResult
from .clustering_validator import ClusteringValidator, ClusteringFoldResult
from .results import EvaluationResults, CCAValidationResults, ClusteringValidationResults
from .permutation import PermutationTester


@dataclass
class PipelineFoldResult:
    """Results from one fold of pipeline validation."""
    cca_result: CCAFoldResult
    clustering_results: Dict[str, ClusteringFoldResult]
    train_idx: np.ndarray
    test_idx: np.ndarray


class PipelineValidator:
    """
    End-to-end pipeline validator for CCA + Clustering.

    Runs nested cross-validation where:
    1. CCA is fit on train data
    2. CCA variates are computed for both train and test
    3. Clustering is fit on train CCA variates
    4. Test subjects are assigned to clusters
    5. Results are aggregated across repeats

    Supports multiple clustering configs against the same CCA run.
    """

    def __init__(
        self,
        n_subjects: int,
        config: Optional[EvaluationConfig] = None
    ):
        """
        Initialize pipeline validator.

        Args:
            n_subjects: Total number of subjects
            config: Evaluation configuration
        """
        self.n_subjects = n_subjects
        self.config = config or EvaluationConfig()
        self.splitter = TrainTestSplitter(self.config)

    def run(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clustering_configs: List[Dict[str, Any]],
        cca_settings: Optional[Dict[str, Any]] = None,
        feature_processing_configs: Optional[List] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResults:
        """
        Run full pipeline validation.

        Args:
            X: X features (n_subjects x n_features_x)
            Y: Y features (n_subjects x n_features_y)
            clustering_configs: List of clustering configs, each a dict with:
                - 'name': Config name (e.g., 'kmeans_3')
                - 'n_clusters': Number of clusters
                - 'method': 'kmeans', 'hierarchical', 'dbscan'
            cca_settings: CCA settings dict
            feature_processing_configs: Inside-CV preprocessing configs
                (passed to CCAValidator for per-fold application)
            progress_callback: Optional callback (current_repeat, total_repeats)

        Returns:
            EvaluationResults with CCA and clustering validation results
        """
        cca_settings = cca_settings or {
            'cca_type': 'regularized',
            'n_components': 2,
            'regularization_x': 0.1,
            'regularization_y': 0.1,
            'use_matlab_style': True
        }

        # Initialize validators — pass preprocessing configs for inside-CV steps
        cca_validator = CCAValidator(
            feature_processing_configs=feature_processing_configs
        )

        clustering_validators = {
            cfg['name']: ClusteringValidator(
                n_subjects=self.n_subjects,
                config=self.config,
                reference_strategy=self.config.label_reference_strategy
            )
            for cfg in clustering_configs
        }

        # Force sequential when progress_callback or cancellation needed
        use_parallel = (
            self.config.n_jobs != 1
            and progress_callback is None
            and not hasattr(self.config, '_cancel_event')
        )

        if use_parallel:
            all_results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(self._run_repeat)(
                    repeat_idx, X, Y, clustering_configs, cca_settings,
                    cca_validator
                )
                for repeat_idx in range(self.config.n_repeats)
            )
        else:
            all_results = []
            for repeat_idx in range(self.config.n_repeats):
                self.config.check_cancelled()
                if progress_callback:
                    progress_callback(repeat_idx, self.config.n_repeats)
                result = self._run_repeat(
                    repeat_idx, X, Y, clustering_configs, cca_settings,
                    cca_validator
                )
                all_results.append(result)

        # Process results and update validators
        for repeat_idx, (cca_fold_results, clustering_fold_results) in enumerate(all_results):
            for cfg_name, validator in clustering_validators.items():
                validator.process_repeat(
                    clustering_fold_results[cfg_name],
                    repeat_idx
                )

        # Aggregate CCA results
        cca_aggregated = cca_validator.aggregate_results(
            [[r[0][f] for f in range(self.config.n_outer_folds)] for r in all_results],
            self.config.n_repeats,
            self.config.n_outer_folds
        )

        # Aggregate clustering results
        clustering_aggregated = {}
        for cfg in clustering_configs:
            cfg_name = cfg['name']
            fold_results_by_repeat = [
                all_results[r][1][cfg_name]
                for r in range(self.config.n_repeats)
            ]
            clustering_aggregated[cfg_name] = clustering_validators[cfg_name].aggregate_results(
                fold_results_by_repeat,
                self.config.n_repeats,
                self.config.n_outer_folds
            )

        # Build EvaluationResults
        cca_validation = CCAValidationResults(
            n_repeats=self.config.n_repeats,
            n_folds=self.config.n_outer_folds,
            train_correlations=cca_aggregated['train_correlations'],
            test_correlations=cca_aggregated['test_correlations']
        )

        clustering_validation = {}
        for cfg in clustering_configs:
            cfg_name = cfg['name']
            agg = clustering_aggregated[cfg_name]
            clustering_validation[cfg_name] = ClusteringValidationResults(
                config_name=cfg_name,
                n_repeats=self.config.n_repeats,
                n_folds=self.config.n_outer_folds,
                test_silhouettes=agg['test_silhouettes'],
                subject_consistency=agg['subject_consistency'],
                final_labels=agg['final_labels'],
                consensus_matrix=agg['consensus_matrix']
            )

        return EvaluationResults(
            cca_validation=cca_validation,
            clustering_validation=clustering_validation,
            config=self.config.to_dict()
        )

    def _run_repeat(
        self,
        repeat_idx: int,
        X: np.ndarray,
        Y: np.ndarray,
        clustering_configs: List[Dict[str, Any]],
        cca_settings: Dict[str, Any],
        cca_validator: CCAValidator
    ) -> tuple:
        """
        Run all folds for one repeat.

        Returns:
            Tuple of (cca_fold_results, clustering_fold_results_by_config)
        """
        cca_fold_results = []
        clustering_fold_results = {cfg['name']: [] for cfg in clustering_configs}

        for fold_idx, train_idx, test_idx in self.splitter.generate_folds(
            self.n_subjects, repeat_idx
        ):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            Y_train, Y_test = Y[train_idx], Y[test_idx]

            # Run CCA
            cca_result = cca_validator.validate_fold(
                X_train, Y_train, X_test, Y_test,
                cca_type=cca_settings.get('cca_type', 'regularized'),
                n_components=cca_settings.get('n_components', 2),
                regularization_x=cca_settings.get('regularization_x', 0.1),
                regularization_y=cca_settings.get('regularization_y', 0.1),
                use_matlab_style=cca_settings.get('use_matlab_style', True),
                run_hpo=cca_settings.get('run_hpo', False),
                reduced_hpo_repeats=self.config.reduced_inner_hpo_repeats
            )
            cca_fold_results.append(cca_result)

            cluster_train, cluster_test = self._extract_variates(
                cca_result.fitted_model, X_train, Y_train, X_test, Y_test
            )

            # Run clustering for each config
            for cfg in clustering_configs:
                cfg_name = cfg['name']

                # Create temporary validator for this fold
                temp_validator = ClusteringValidator(
                    n_subjects=self.n_subjects,
                    config=self.config
                )

                cluster_result = temp_validator.validate_fold(
                    cluster_train, cluster_test,
                    train_idx, test_idx,
                    n_clusters=cfg.get('n_clusters', 3),
                    clustering_method=cfg.get('method', 'kmeans'),
                    random_state=42 + repeat_idx
                )

                clustering_fold_results[cfg_name].append(cluster_result)

        return cca_fold_results, clustering_fold_results

    @staticmethod
    def _extract_variates(
        model: Dict[str, Any],
        X_train: np.ndarray,
        Y_train: np.ndarray,
        X_test: np.ndarray,
        Y_test: np.ndarray
    ) -> tuple:
        """Extract CCA variates from fitted model, dispatch by model type.

        Returns (cluster_train, cluster_test) — concatenated U,V variates.
        """
        if 'x_weights' in model and 'y_weights' in model:
            # MATLABStyleRCCA — weight matrices + means
            x_mean = model.get('x_mean', np.zeros(X_train.shape[1]))
            y_mean = model.get('y_mean', np.zeros(Y_train.shape[1]))
            U_train = (X_train - x_mean) @ model['x_weights']
            V_train = (Y_train - y_mean) @ model['y_weights']
            U_test = (X_test - x_mean) @ model['x_weights']
            V_test = (Y_test - y_mean) @ model['y_weights']
        elif 'weights' in model:
            # cca-zoo models — list of weight matrices [W_x, W_y]
            weights = model['weights']
            W_x, W_y = np.array(weights[0]), np.array(weights[1])
            # cca-zoo stores centered weights; center data with train mean
            x_mean = X_train.mean(axis=0)
            y_mean = Y_train.mean(axis=0)
            U_train = (X_train - x_mean) @ W_x
            V_train = (Y_train - y_mean) @ W_y
            U_test = (X_test - x_mean) @ W_x
            V_test = (Y_test - y_mean) @ W_y
        else:
            # Fallback: raw features
            return np.hstack([X_train, Y_train]), np.hstack([X_test, Y_test])

        return np.hstack([U_train, V_train]), np.hstack([U_test, V_test])

    def run_with_permutation_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        clustering_configs: List[Dict[str, Any]],
        cca_settings: Optional[Dict[str, Any]] = None,
        feature_processing_configs: Optional[List] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> EvaluationResults:
        """Run pipeline validation with permutation test."""
        results = self.run(
            X, Y, clustering_configs, cca_settings,
            feature_processing_configs, progress_callback
        )

        if not self.config.run_permutation_test:
            return results

        # Run permutation test
        tester = PermutationTester(n_permutations=1, random_state=self.config.random_state)

        def validation_fn(X_data, Y_data):
            perm_results = self.run(
                X_data, Y_data, clustering_configs, cca_settings,
                feature_processing_configs
            )
            return perm_results.cca_validation.test_correlations.flatten()

        perm_result = tester.run_permutation_test(X, Y, validation_fn)

        p_value = tester.compute_p_value(
            perm_result.observed_distribution,
            perm_result.null_distribution
        )

        from .results import PermutationTestResults
        results.permutation_test = PermutationTestResults(
            observed_distribution=perm_result.observed_distribution,
            null_distribution=perm_result.null_distribution,
            p_value=p_value
        )

        return results
