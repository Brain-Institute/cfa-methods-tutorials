"""
IM-Biotype Validation Module

Provides synthetic data generation with known ground truth for validating
the IM-Biotype toolbox. Useful for publication benchmarking and testing.

Main Classes:
    ValidationStudyGenerator: Generate complete validation studies
    ValidationStudyConfig: Configuration for study generation
    BiotypeMixtureModel: Ground truth biotype cluster model
    MultimodalGenerator: Correlated multimodal data generation
    NoiseController: Site effects, missing data, confounds

Scenario Suites:
    publication_suite: Complete validation suite for publication
    quick_test_suite: Minimal suite for development testing

Metrics:
    clustering_recovery_metrics: ARI, NMI, V-measure
    cca_recovery_metrics: Correlation error, subspace angles
    compute_all_validation_metrics: All metrics in one call

Example:
    >>> from imbiotype.validation import ValidationStudyGenerator, ValidationStudyConfig
    >>>
    >>> config = ValidationStudyConfig(
    ...     name="my_validation",
    ...     n_samples=200,
    ...     n_clusters=3,
    ...     modality_x="eeg",
    ...     modality_y="clinical"
    ... )
    >>> generator = ValidationStudyGenerator(config)
    >>> study_path = generator.generate("/path/to/output")
"""

from .modality_templates import (
    ModalityTemplate,
    MODALITY_TEMPLATES,
    CLINICAL_TEMPLATE,
    EEG_TEMPLATE,
    MRI_TEMPLATE,
    GENETICS_TEMPLATE,
    BIOMETRICS_TEMPLATE,
    BEHAVIORAL_TEMPLATE,
    get_template,
    list_modalities,
    list_categories,
    generate_feature_names
)

from .mixture_model import (
    BiotypeMixtureModel,
    create_mixture_model
)

from .multimodal_generator import (
    ViewConfig,
    MultimodalGenerator,
    create_two_view_generator
)

from .multiview_generator import (
    MultiviewGenerator,
    create_multiview_generator
)

from .noise_controller import (
    SiteEffectConfig,
    MissingDataConfig,
    ConfoundConfig,
    NoiseController,
    create_noise_controller
)

from .metrics import (
    clustering_recovery_metrics,
    cluster_quality_metrics,
    canonical_correlation_error,
    subspace_angle,
    loading_correlation,
    cca_recovery_metrics,
    compute_all_validation_metrics,
    summarize_validation_results
)

from .study_generator import (
    ValidationStudyConfig,
    ValidationStudyGenerator
)

from .visualizer import (
    ValidationVisualizer
)

from .scenario_suites import (
    create_baseline_config,
    cluster_number_suite,
    separation_suite,
    imbalance_suite,
    snr_suite,
    sample_size_suite,
    dimensionality_suite,
    missing_data_suite,
    multisite_suite,
    confound_suite,
    observation_model_suite,
    covariance_suite,
    snr_heterogeneity_suite,
    multiview_suite,
    publication_suite,
    generate_suite,
    quick_test_suite
)

__all__ = [
    # Modality templates
    "ModalityTemplate",
    "MODALITY_TEMPLATES",
    "CLINICAL_TEMPLATE",
    "EEG_TEMPLATE",
    "MRI_TEMPLATE",
    "GENETICS_TEMPLATE",
    "BIOMETRICS_TEMPLATE",
    "BEHAVIORAL_TEMPLATE",
    "get_template",
    "list_modalities",
    "list_categories",
    "generate_feature_names",

    # Mixture model
    "BiotypeMixtureModel",
    "create_mixture_model",

    # Multimodal generator
    "ViewConfig",
    "MultimodalGenerator",
    "create_two_view_generator",

    # Multiview generator
    "MultiviewGenerator",
    "create_multiview_generator",

    # Noise controller
    "SiteEffectConfig",
    "MissingDataConfig",
    "ConfoundConfig",
    "NoiseController",
    "create_noise_controller",

    # Metrics
    "clustering_recovery_metrics",
    "cluster_quality_metrics",
    "canonical_correlation_error",
    "subspace_angle",
    "loading_correlation",
    "cca_recovery_metrics",
    "compute_all_validation_metrics",
    "summarize_validation_results",

    # Study generator
    "ValidationStudyConfig",
    "ValidationStudyGenerator",

    # Visualizer
    "ValidationVisualizer",

    # Scenario suites
    "create_baseline_config",
    "cluster_number_suite",
    "separation_suite",
    "imbalance_suite",
    "snr_suite",
    "sample_size_suite",
    "dimensionality_suite",
    "missing_data_suite",
    "multisite_suite",
    "confound_suite",
    "observation_model_suite",
    "covariance_suite",
    "snr_heterogeneity_suite",
    "multiview_suite",
    "publication_suite",
    "generate_suite",
    "quick_test_suite"
]
