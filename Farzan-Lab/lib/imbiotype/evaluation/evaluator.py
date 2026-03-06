"""
BiotypeEvaluator - Main class for validating biotype pipelines.

Validates existing fusion/clustering runs with nested cross-validation.
Delegates to PipelineValidator for actual computation.
"""

import os
import json
from typing import Optional, List, Dict, Any, Union
import numpy as np

from .config import EvaluationConfig
from .results import EvaluationResults
from .pipeline_validator import PipelineValidator


class BiotypeEvaluator:
    """
    Validates existing fusion/clustering runs with nested CV.

    Delegates to PipelineValidator for computation. Handles:
    - Loading features from Study
    - Extracting CCA settings from FusionConfig
    - Translating clustering config names to dicts
    - Applying outside-CV preprocessing upfront

    Example:
        evaluator = BiotypeEvaluator(study)
        results = evaluator.validate(
            fusion_run_id='20260108_221701',
            clustering_configs=['kmeans_3', 'kmeans_4'],
            config=EvaluationConfig(n_repeats=100, n_outer_folds=10)
        )
    """

    def __init__(self, study):
        self.study = study

    def validate(
        self,
        fusion_run_id: str,
        clustering_configs: Union[List[str], List[Dict[str, Any]]],
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationResults:
        """Validate fusion + clustering configs with nested CV."""
        config = config or EvaluationConfig()

        fusion_run = self._get_fusion_run(fusion_run_id)
        fusion_config = fusion_run['config']
        features = self._load_features(fusion_config)
        n_subjects = self._get_n_subjects(features)

        cca_settings = self._extract_cca_settings(
            fusion_config, fusion_run.get('optimized_params')
        )
        clustering_dicts = self._translate_clustering_configs(clustering_configs)
        feature_processing_configs = self._get_inside_cv_configs(fusion_config)

        validator = PipelineValidator(n_subjects=n_subjects, config=config)

        if config.run_permutation_test:
            return validator.run_with_permutation_test(
                X=features['x'], Y=features['y'],
                clustering_configs=clustering_dicts,
                cca_settings=cca_settings,
                feature_processing_configs=feature_processing_configs,
                progress_callback=config.progress_callback
            )
        else:
            return validator.run(
                X=features['x'], Y=features['y'],
                clustering_configs=clustering_dicts,
                cca_settings=cca_settings,
                feature_processing_configs=feature_processing_configs,
                progress_callback=config.progress_callback
            )

    def validate_cca_only(
        self,
        fusion_run_id: str,
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationResults:
        """Validate CCA only (no clustering)."""
        config = config or EvaluationConfig()

        fusion_run = self._get_fusion_run(fusion_run_id)
        fusion_config = fusion_run['config']
        features = self._load_features(fusion_config)
        n_subjects = self._get_n_subjects(features)

        cca_settings = self._extract_cca_settings(
            fusion_config, fusion_run.get('optimized_params')
        )
        feature_processing_configs = self._get_inside_cv_configs(fusion_config)

        validator = PipelineValidator(n_subjects=n_subjects, config=config)

        # Empty clustering configs = CCA only
        if config.run_permutation_test:
            return validator.run_with_permutation_test(
                X=features['x'], Y=features['y'],
                clustering_configs=[],
                cca_settings=cca_settings,
                feature_processing_configs=feature_processing_configs,
                progress_callback=config.progress_callback
            )
        else:
            return validator.run(
                X=features['x'], Y=features['y'],
                clustering_configs=[],
                cca_settings=cca_settings,
                feature_processing_configs=feature_processing_configs,
                progress_callback=config.progress_callback
            )

    def validate_clustering_only(
        self,
        clustering_input: np.ndarray,
        clustering_configs: Union[List[str], List[Dict[str, Any]]],
        config: Optional[EvaluationConfig] = None
    ) -> EvaluationResults:
        """Validate clustering only using pre-computed variates."""
        config = config or EvaluationConfig()
        n_subjects = clustering_input.shape[0]
        n_features = clustering_input.shape[1]

        clustering_dicts = self._translate_clustering_configs(clustering_configs)

        # Split input in half as dummy X/Y so PipelineValidator can run CCA
        # but the CCA results won't be meaningful — clustering uses the variates
        half = n_features // 2
        X = clustering_input[:, :half]
        Y = clustering_input[:, half:]

        validator = PipelineValidator(n_subjects=n_subjects, config=config)
        return validator.run(
            X=X, Y=Y,
            clustering_configs=clustering_dicts,
            progress_callback=config.progress_callback
        )

    # -- Private helpers --

    def _get_fusion_run(self, fusion_run_id: str) -> Dict[str, Any]:
        from ..models.fusion import FusionConfig

        run_dir = os.path.join(self.study.fusion_path, fusion_run_id)
        config_path = os.path.join(run_dir, "config.json")

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Fusion run config not found: {config_path}")

        with open(config_path, 'r') as f:
            config_dict = json.load(f)

        # Load optimized parameters from results if available
        optimized_params = self._load_optimized_params(run_dir)

        return {
            'run_id': fusion_run_id,
            'config': FusionConfig.from_dict(config_dict),
            'run_dir': run_dir,
            'optimized_params': optimized_params
        }

    def _load_optimized_params(self, run_dir: str) -> Optional[Dict[str, float]]:
        """Load HPO-optimized parameters from fusion run results."""
        results_path = os.path.join(run_dir, "standardized_results.json")
        if not os.path.exists(results_path):
            return None
        try:
            with open(results_path, 'r') as f:
                std_results = json.load(f)
            opt = (std_results.get('results', {})
                   .get('cca', {})
                   .get('optimization_results', {}))
            best = opt.get('best_parameters')
            if best and opt.get('hyperparameter_optimization_enabled'):
                return best
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
        return None

    def _load_features(self, fusion_config) -> Dict[str, np.ndarray]:
        """Load feature matrices, applying outside-CV preprocessing upfront."""
        features = {}
        cca_settings = fusion_config.cca_settings

        x_name = cca_settings.x_main_feature_vector_name
        if x_name:
            x_data = self.study.data_manager.get_feature_data(x_name)
            features['x'] = x_data.values if hasattr(x_data, 'values') else x_data

        y_name = cca_settings.y_main_feature_vector_name
        if y_name:
            y_data = self.study.data_manager.get_feature_data(y_name)
            features['y'] = y_data.values if hasattr(y_data, 'values') else y_data

        # Apply outside-CV steps upfront (transform, variance_filter_pre)
        features = self._apply_outside_cv_preprocessing(features, fusion_config)

        return features

    def _apply_outside_cv_preprocessing(
        self, features: Dict[str, np.ndarray], fusion_config
    ) -> Dict[str, np.ndarray]:
        """Apply outside-CV preprocessing steps (transform, variance_filter_pre)."""
        try:
            from ..processing.feature_processor import (
                build_processing_plan
            )
            import pandas as pd

            for cfg in (fusion_config.feature_processing_configs or []):
                if not hasattr(cfg, 'preprocessing_config'):
                    continue
                plan = build_processing_plan(cfg.preprocessing_config, mode="with_hpo")
                if not plan.outside_cv_steps:
                    continue

                # Determine which view this config applies to
                key = self._config_to_view_key(cfg)
                if key and key in features:
                    df = pd.DataFrame(features[key])
                    for step in plan.outside_cv_steps:
                        step.fit(df)
                        df = step.transform(df)
                    features[key] = df.values
        except (ImportError, AttributeError):
            pass
        return features

    def _config_to_view_key(self, cfg) -> Optional[str]:
        """Map a feature processing config to 'x' or 'y'."""
        if not hasattr(cfg, 'feature_name'):
            return None
        name = cfg.feature_name.lower()
        if 'bio' in name or 'brain' in name or 'neuro' in name:
            return 'x'
        elif 'clinical' in name or 'symptom' in name or 'behav' in name:
            return 'y'
        return None

    def _get_inside_cv_configs(self, fusion_config) -> Optional[List]:
        """Extract inside-CV preprocessing configs from FusionConfig."""
        configs = fusion_config.feature_processing_configs
        if configs:
            return configs
        return None

    def _extract_cca_settings(
        self, fusion_config, optimized_params: Optional[Dict] = None
    ) -> Dict[str, Any]:
        model = fusion_config.cca_settings.model_settings

        # Use HPO-optimized lambdas from the fusion run if available;
        # skip re-running HPO since we already have the optimal values.
        if optimized_params:
            reg_x = optimized_params.get('regularization_x', model.regularization_x)
            reg_y = optimized_params.get('regularization_y', model.regularization_y)
            run_hpo = False
        else:
            reg_x = model.regularization_x
            reg_y = model.regularization_y
            run_hpo = fusion_config.cca_settings.hyperparameter_optimization_enabled

        return {
            'cca_type': model.cca_type,
            'n_components': model.n_components,
            'regularization_x': reg_x,
            'regularization_y': reg_y,
            'use_matlab_style': model.regularization_method == 'covariance',
            'run_hpo': run_hpo
        }

    def _get_n_subjects(self, features: Dict[str, np.ndarray]) -> int:
        return features.get('x', features.get('y', np.array([]))).shape[0]

    def _translate_clustering_configs(
        self, configs: Union[List[str], List[Dict[str, Any]]]
    ) -> List[Dict[str, Any]]:
        """Translate clustering config identifiers to dicts.

        Accepts:
        - List of dicts (already in {name, n_clusters, method} format)
        - List of strings like 'kmeans_3' → {name: 'kmeans_3', method: 'kmeans', n_clusters: 3}
        """
        result = []
        for cfg in configs:
            if isinstance(cfg, dict):
                result.append(cfg)
            elif isinstance(cfg, str):
                result.append(self._parse_clustering_config_string(cfg))
            else:
                # Assume it's a ClusteringConfig-like object
                result.append({
                    'name': getattr(cfg, 'name', str(cfg)),
                    'n_clusters': getattr(cfg, 'n_clusters', 3),
                    'method': getattr(cfg, 'method', 'kmeans')
                })
        return result

    @staticmethod
    def _parse_clustering_config_string(config_str: str) -> Dict[str, Any]:
        """Parse 'method_k' string to config dict. E.g. 'kmeans_3'."""
        parts = config_str.rsplit('_', 1)
        if len(parts) == 2:
            try:
                return {
                    'name': config_str,
                    'method': parts[0],
                    'n_clusters': int(parts[1])
                }
            except ValueError:
                pass
        # Fallback
        return {'name': config_str, 'method': 'kmeans', 'n_clusters': 3}
