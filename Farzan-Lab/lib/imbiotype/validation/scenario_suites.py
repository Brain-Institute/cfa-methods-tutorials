"""
Pre-defined Validation Scenario Suites

Provides systematic validation experiments for publication-ready benchmarking.
Each suite varies specific parameters to test toolbox robustness.
"""

from typing import List, Dict, Optional, Tuple
from pathlib import Path
import numpy as np

from .study_generator import ValidationStudyConfig, ValidationStudyGenerator


def create_baseline_config(
    name: str = "baseline",
    random_state: int = 42,
    **kwargs
) -> ValidationStudyConfig:
    """
    Create a baseline configuration with sensible defaults.

    High SNR, clear separation, balanced clusters, no noise.
    Should achieve near-perfect recovery.
    """
    defaults = {
        "name": name,
        "n_samples": 200,
        "n_clusters": 3,
        "cluster_separation": 2.5,
        "cluster_balance": "balanced",
        "n_latent_dims": 5,
        "n_shared_components": 3,
        "n_features_x": 50,
        "n_features_y": 30,
        "modality_x": "eeg",
        "modality_y": "clinical",
        "snr_x": 3.0,
        "snr_y": 3.0,
        "n_sites": 1,
        "missing_rate": 0.0,
        "random_state": random_state
    }
    defaults.update(kwargs)
    return ValidationStudyConfig(**defaults)


# =============================================================================
# Cluster Number Sweep
# =============================================================================

def cluster_number_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying number of clusters (k=2,3,4,5,6).

    Tests toolbox ability to recover different numbers of biotypes.
    """
    configs = []
    for k in [2, 3, 4, 5, 6]:
        config = create_baseline_config(
            name=f"k{k}_clusters",
            n_clusters=k,
            n_samples=n_samples,
            random_state=base_seed + k,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Cluster Separation Sweep
# =============================================================================

def separation_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying cluster separation (Cohen's d from 0.5 to 3.0).

    Tests toolbox sensitivity to cluster overlap.
    """
    configs = []
    separations = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for i, sep in enumerate(separations):
        config = create_baseline_config(
            name=f"sep_{sep:.1f}",
            cluster_separation=sep,
            n_samples=n_samples,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Cluster Imbalance Sweep
# =============================================================================

def imbalance_suite(
    base_seed: int = 42,
    n_samples: int = 300,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying cluster balance (balanced to 90/10 imbalance).

    Tests toolbox robustness to unequal cluster sizes.
    """
    configs = []

    # Balanced
    configs.append(create_baseline_config(
        name="balanced",
        cluster_balance="balanced",
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # Various imbalance ratios
    imbalance_ratios = [0.6, 0.7, 0.8, 0.9]
    for i, ratio in enumerate(imbalance_ratios):
        config = create_baseline_config(
            name=f"imbalance_{int(ratio*100)}pct",
            cluster_balance="imbalanced",
            imbalance_ratio=ratio,
            n_samples=n_samples,
            random_state=base_seed + i + 1,
            **kwargs
        )
        configs.append(config)

    return configs


# =============================================================================
# SNR (Noise Robustness) Sweep
# =============================================================================

def snr_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying signal-to-noise ratio (0.5 to 5.0).

    Tests toolbox robustness to noisy data.
    """
    configs = []
    snr_values = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for i, snr in enumerate(snr_values):
        config = create_baseline_config(
            name=f"snr_{snr:.1f}",
            snr_x=snr,
            snr_y=snr,
            n_samples=n_samples,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Sample Size Sweep
# =============================================================================

def sample_size_suite(
    base_seed: int = 42,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying sample size (n=50 to n=500).

    Tests toolbox performance with limited samples.
    """
    configs = []
    sample_sizes = [50, 100, 150, 200, 300, 500]
    for i, n in enumerate(sample_sizes):
        config = create_baseline_config(
            name=f"n{n}",
            n_samples=n,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Dimensionality Sweep
# =============================================================================

def dimensionality_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying feature dimensionality (10 to 200 features per view).

    Tests toolbox scalability with high-dimensional data.
    """
    configs = []
    dims = [(10, 10), (30, 20), (50, 30), (100, 50), (150, 100), (200, 150)]
    for i, (dim_x, dim_y) in enumerate(dims):
        config = create_baseline_config(
            name=f"dim_{dim_x}x{dim_y}",
            n_features_x=dim_x,
            n_features_y=dim_y,
            n_samples=n_samples,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Missing Data Sweep
# =============================================================================

def missing_data_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying missing data proportion (0% to 20%).

    Tests toolbox tolerance to incomplete data.
    """
    configs = []
    missing_rates = [0.0, 0.05, 0.10, 0.15, 0.20]
    for i, rate in enumerate(missing_rates):
        config = create_baseline_config(
            name=f"missing_{int(rate*100)}pct",
            missing_rate=rate,
            missing_mechanism="MCAR",
            n_samples=n_samples,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Multi-Site Sweep
# =============================================================================

def multisite_suite(
    base_seed: int = 42,
    n_samples: int = 300,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying number of sites and site effect magnitude.

    Tests toolbox ability to handle batch effects.
    """
    configs = []

    # Single site (baseline)
    configs.append(create_baseline_config(
        name="single_site",
        n_sites=1,
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # Multiple sites with varying effect magnitudes
    site_configs = [
        (3, 0.3),  # 3 sites, low effect
        (3, 0.5),  # 3 sites, medium effect
        (3, 0.8),  # 3 sites, high effect
        (5, 0.3),  # 5 sites, low effect
        (5, 0.5),  # 5 sites, medium effect
    ]

    for i, (n_sites, magnitude) in enumerate(site_configs):
        config = create_baseline_config(
            name=f"sites{n_sites}_mag{magnitude:.1f}",
            n_sites=n_sites,
            site_effect_magnitude=magnitude,
            n_samples=n_samples,
            random_state=base_seed + i + 1,
            **kwargs
        )
        configs.append(config)

    return configs


# =============================================================================
# Confound Suite
# =============================================================================

def confound_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying confound complexity.

    Tests residualization effectiveness.
    """
    configs = []

    # No confounds (baseline)
    configs.append(create_baseline_config(
        name="no_confounds",
        include_age=False,
        include_sex=False,
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # Age only
    configs.append(create_baseline_config(
        name="age_confound",
        include_age=True,
        include_sex=False,
        confound_effect_magnitude=0.3,
        n_samples=n_samples,
        random_state=base_seed + 1,
        **kwargs
    ))

    # Sex only
    configs.append(create_baseline_config(
        name="sex_confound",
        include_age=False,
        include_sex=True,
        confound_effect_magnitude=0.3,
        n_samples=n_samples,
        random_state=base_seed + 2,
        **kwargs
    ))

    # Both age and sex
    configs.append(create_baseline_config(
        name="age_sex_confounds",
        include_age=True,
        include_sex=True,
        confound_effect_magnitude=0.3,
        n_samples=n_samples,
        random_state=base_seed + 3,
        **kwargs
    ))

    # Strong confound effects
    configs.append(create_baseline_config(
        name="strong_confounds",
        include_age=True,
        include_sex=True,
        confound_effect_magnitude=0.6,
        n_samples=n_samples,
        random_state=base_seed + 4,
        **kwargs
    ))

    return configs


# =============================================================================
# Full Publication Suite
# =============================================================================

# =============================================================================
# Observation Model Sweep
# =============================================================================

def observation_model_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying observation models (gaussian, poisson, bernoulli, ordinal).

    Tests toolbox robustness to non-Gaussian data types.
    """
    configs = []
    models = [
        ("gaussian", "gaussian"),
        ("poisson", "poisson"),
        ("bernoulli", "bernoulli"),
        ("ordinal", "ordinal"),
        ("gaussian", "ordinal"),   # Mixed: continuous + ordinal (realistic)
        ("gaussian", "bernoulli"), # Mixed: continuous + binary
    ]
    for i, (obs_x, obs_y) in enumerate(models):
        config = create_baseline_config(
            name=f"obs_{obs_x}_{obs_y}",
            observation_model_x=obs_x,
            observation_model_y=obs_y,
            n_samples=n_samples,
            random_state=base_seed + i,
            **kwargs
        )
        configs.append(config)
    return configs


# =============================================================================
# Covariance Structure Sweep
# =============================================================================

def covariance_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying latent covariance structure.

    Tests toolbox robustness to autocorrelated latent structure.
    """
    configs = []

    # Identity baseline
    configs.append(create_baseline_config(
        name="cov_identity",
        covariance_type="identity",
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # AR(1) with varying rho
    for i, rho in enumerate([0.3, 0.5, 0.7, 0.9]):
        configs.append(create_baseline_config(
            name=f"cov_ar1_rho{int(rho*10)}",
            covariance_type="ar1",
            ar1_rho=rho,
            n_samples=n_samples,
            random_state=base_seed + i + 1,
            **kwargs
        ))

    # AR(1) with Beta variance scaling
    configs.append(create_baseline_config(
        name="cov_ar1_beta_scaling",
        covariance_type="ar1",
        ar1_rho=0.5,
        variance_scaling="beta",
        variance_scaling_magnitude=1.5,
        n_samples=n_samples,
        random_state=base_seed + 10,
        **kwargs
    ))

    return configs


# =============================================================================
# Per-Feature SNR Heterogeneity Sweep
# =============================================================================

def snr_heterogeneity_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying per-feature SNR heterogeneity.

    Tests toolbox robustness to heteroscedastic noise.
    """
    configs = []

    # Uniform baseline
    configs.append(create_baseline_config(
        name="snr_uniform",
        snr_distribution_x="uniform",
        snr_distribution_y="uniform",
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # Beta distribution with varying scale
    for i, scale in enumerate([0.5, 1.0, 1.5, 2.0]):
        configs.append(create_baseline_config(
            name=f"snr_beta_scale{scale:.1f}",
            snr_distribution_x="beta",
            snr_distribution_y="beta",
            snr_scale=scale,
            n_samples=n_samples,
            random_state=base_seed + i + 1,
            **kwargs
        ))

    return configs


# =============================================================================
# Multi-View Sweep
# =============================================================================

def multiview_suite(
    base_seed: int = 42,
    n_samples: int = 200,
    **kwargs
) -> List[ValidationStudyConfig]:
    """
    Suite varying number of views (2 to 4).

    Tests toolbox with >2 modalities.
    """
    configs = []

    # 2-view baseline
    configs.append(create_baseline_config(
        name="views_2",
        n_views=2,
        n_samples=n_samples,
        random_state=base_seed,
        **kwargs
    ))

    # 3-view: add MRI
    configs.append(create_baseline_config(
        name="views_3",
        n_views=3,
        additional_view_configs=[
            {"name": "view_z", "n_features": 20, "modality": "mri", "snr": 2.0}
        ],
        n_samples=n_samples,
        random_state=base_seed + 1,
        **kwargs
    ))

    # 4-view: add MRI + genetics
    configs.append(create_baseline_config(
        name="views_4",
        n_views=4,
        additional_view_configs=[
            {"name": "view_z", "n_features": 20, "modality": "mri", "snr": 2.0},
            {"name": "view_w", "n_features": 15, "modality": "genetics", "snr": 1.5},
        ],
        n_samples=n_samples,
        random_state=base_seed + 2,
        **kwargs
    ))

    return configs


# =============================================================================
# Full Publication Suite
# =============================================================================

def publication_suite(
    base_seed: int = 42,
    output_dir: Optional[Path] = None,
    **kwargs
) -> Dict[str, List[ValidationStudyConfig]]:
    """
    Complete validation suite for publication.

    Returns all scenario suites organized by category.

    Args:
        base_seed: Base random seed
        output_dir: If provided, generate all studies to this directory
        **kwargs: Additional arguments passed to all configs

    Returns:
        Dictionary mapping suite names to lists of configs
    """
    suites = {
        "cluster_number": cluster_number_suite(base_seed, **kwargs),
        "separation": separation_suite(base_seed + 100, **kwargs),
        "imbalance": imbalance_suite(base_seed + 200, **kwargs),
        "snr": snr_suite(base_seed + 300, **kwargs),
        "sample_size": sample_size_suite(base_seed + 400, **kwargs),
        "dimensionality": dimensionality_suite(base_seed + 500, **kwargs),
        "missing_data": missing_data_suite(base_seed + 600, **kwargs),
        "multisite": multisite_suite(base_seed + 700, **kwargs),
        "confound": confound_suite(base_seed + 800, **kwargs),
        "observation_model": observation_model_suite(base_seed + 900, **kwargs),
        "covariance": covariance_suite(base_seed + 1000, **kwargs),
        "snr_heterogeneity": snr_heterogeneity_suite(base_seed + 1100, **kwargs),
        "multiview": multiview_suite(base_seed + 1200, **kwargs),
    }

    if output_dir is not None:
        generate_suite(suites, output_dir)

    return suites


def generate_suite(
    suites: Dict[str, List[ValidationStudyConfig]],
    output_dir: Path
) -> Dict[str, List[Path]]:
    """
    Generate all studies in a suite collection.

    Args:
        suites: Dictionary of suite name -> list of configs
        output_dir: Base output directory

    Returns:
        Dictionary mapping suite names to lists of generated study paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_paths = {}

    for suite_name, configs in suites.items():
        suite_dir = output_dir / suite_name
        suite_dir.mkdir(exist_ok=True)

        paths = []
        for config in configs:
            generator = ValidationStudyGenerator(config)
            study_path = generator.generate(suite_dir)
            paths.append(study_path)
            print(f"Generated: {study_path}")

        generated_paths[suite_name] = paths

    return generated_paths


def quick_test_suite(base_seed: int = 42, **kwargs) -> List[ValidationStudyConfig]:
    """
    Quick test suite with minimal scenarios for development.

    Uses smaller sample sizes and fewer variations.
    """
    configs = [
        create_baseline_config(
            name="quick_baseline",
            n_samples=100,
            random_state=base_seed,
            **kwargs
        ),
        create_baseline_config(
            name="quick_noisy",
            n_samples=100,
            snr_x=1.0,
            snr_y=1.0,
            random_state=base_seed + 1,
            **kwargs
        ),
        create_baseline_config(
            name="quick_multisite",
            n_samples=100,
            n_sites=3,
            site_effect_magnitude=0.5,
            random_state=base_seed + 2,
            **kwargs
        )
    ]
    return configs
