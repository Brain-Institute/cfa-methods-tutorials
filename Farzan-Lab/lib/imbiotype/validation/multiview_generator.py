"""
N-View Multimodal Data Generator for Validation Study Generator

Generalizes the 2-view MultimodalGenerator to arbitrary N views.
All views share a common latent factor space; cross-view correlations
are controlled via a reference-view model.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field

from .mixture_model import BiotypeMixtureModel
from .multimodal_generator import ViewConfig


@dataclass
class MultiviewGenerator:
    """
    Generates correlated N-view multimodal data with known ground truth.

    Reference-view model:
        z ~ BiotypeMixtureModel (shared latent factors)
        View 0 (reference): X_0 = Z @ W_0.T + noise
        View k (k>0):
            Z_k_shared = rho_k * Z_shared + sqrt(1 - rho_k^2) * E_k
            X_k = Z_k @ W_k.T + noise

    Pairwise canonical correlations:
        Corr(view_0, view_k) = rho_k (by construction)
        Corr(view_i, view_j) ≈ rho_i * rho_j (through shared latent)

    Attributes:
        mixture_model: BiotypeMixtureModel for latent factors
        view_configs: List of ViewConfig for each view
        n_shared_components: Number of shared latent dimensions
        target_correlations: Correlation decay for shared dims (applied to all non-reference views)
        per_view_correlation_scales: Optional per-view scaling of target_correlations.
            If provided, view k's effective rho = target_correlations * scale_k.
            Default: all 1.0 (all views same correlation with reference).
        random_state: Random seed
    """
    mixture_model: BiotypeMixtureModel
    view_configs: List[ViewConfig]
    n_shared_components: Optional[int] = None
    target_correlations: Optional[np.ndarray] = None
    per_view_correlation_scales: Optional[List[float]] = None
    random_state: Optional[int] = None

    # Generated (populated during __post_init__)
    loadings: List[np.ndarray] = field(default_factory=list, init=False)
    true_canonical_correlations: np.ndarray = field(default=None, init=False)
    per_view_correlations: Dict[int, np.ndarray] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize loading matrices for all views."""
        self.rng = np.random.default_rng(self.random_state)
        n_views = len(self.view_configs)

        if n_views < 2:
            raise ValueError(f"Need at least 2 views, got {n_views}")

        if self.n_shared_components is None:
            min_features = min(vc.n_features for vc in self.view_configs)
            self.n_shared_components = min(
                self.mixture_model.n_latent_dims, min_features
            )

        if self.target_correlations is None:
            self.target_correlations = np.linspace(0.9, 0.3, self.n_shared_components)

        if self.per_view_correlation_scales is None:
            # All non-reference views get same correlations
            self.per_view_correlation_scales = [1.0] * n_views

        self._generate_loading_matrices()
        self._compute_per_view_correlations()

    def _generate_loading_matrices(self):
        """Generate orthonormal loading matrices for each view."""
        n_latent = self.mixture_model.n_latent_dims
        self.loadings = []

        for vc in self.view_configs:
            n_feat = vc.n_features
            W_random = self.rng.standard_normal((n_feat, n_latent))
            W_orth, _ = np.linalg.qr(W_random)
            self.loadings.append(W_orth[:, :min(n_latent, n_feat)])

    def _compute_per_view_correlations(self):
        """Compute effective target correlations per view."""
        n_shared = self.n_shared_components
        base_rho = self.target_correlations[:n_shared]

        self.per_view_correlations = {}
        for k in range(len(self.view_configs)):
            if k == 0:
                # Reference view: correlation with itself = 1
                self.per_view_correlations[k] = np.ones(n_shared)
            else:
                scale = self.per_view_correlation_scales[k]
                self.per_view_correlations[k] = np.clip(base_rho * scale, 0.0, 1.0)

        self.true_canonical_correlations = base_rho.copy()

    def generate(self, n_samples: int,
                 balance: str = "balanced",
                 imbalance_ratio: float = 0.8) -> Dict:
        """
        Generate N-view multimodal data with known ground truth.

        Args:
            n_samples: Number of samples
            balance: "balanced" or "imbalanced" cluster sizes
            imbalance_ratio: If imbalanced, proportion for largest cluster

        Returns:
            Dictionary containing:
                - views: Dict[str, ndarray] mapping view names to data
                - view_x, view_y: Legacy keys (only if exactly 2 views)
                - latent_factors: Shared latent z
                - cluster_labels: True assignments
                - feature_names: Dict[str, List[str]] per view
                - ground_truth: All ground truth parameters
        """
        # Sample latent factors
        if balance == "balanced":
            latent_factors, cluster_labels = self.mixture_model.sample_balanced(n_samples)
        elif balance == "imbalanced":
            latent_factors, cluster_labels = self.mixture_model.sample_imbalanced(
                n_samples, imbalance_ratio
            )
        else:
            latent_factors, cluster_labels = self.mixture_model.sample(n_samples)

        n_latent = latent_factors.shape[1]
        n_shared = self.n_shared_components

        views = {}
        feature_names = {}

        for k, vc in enumerate(self.view_configs):
            n_latent_k = min(n_latent, self.loadings[k].shape[1])

            if k == 0:
                # Reference view: direct projection
                latent_k = latent_factors[:, :n_latent_k]
            else:
                # Non-reference: inject correlation structure
                rho_k = self.per_view_correlations[k]
                n_shared_k = min(n_shared, n_latent_k, len(rho_k))

                latent_k = np.zeros((n_samples, n_latent_k))

                if n_shared_k > 0:
                    Z_shared = latent_factors[:, :n_shared_k]
                    E_k = self.rng.standard_normal((n_samples, n_shared_k))
                    Z_std = np.maximum(np.std(Z_shared, axis=0), 1e-10)
                    rho_actual = rho_k[:n_shared_k]
                    sqrt_complement = np.sqrt(np.maximum(0, 1 - rho_actual ** 2))
                    latent_k[:, :n_shared_k] = (
                        Z_shared * rho_actual + E_k * Z_std * sqrt_complement
                    )

                if n_latent_k > n_shared_k:
                    latent_k[:, n_shared_k:] = self.rng.standard_normal(
                        (n_samples, n_latent_k - n_shared_k)
                    )

            # Project to observed space
            signal = latent_k @ self.loadings[k][:, :n_latent_k].T

            # Per-feature SNR noise
            signal_std = np.maximum(np.std(signal, axis=0), 1e-10)
            snr_per_feat = vc.get_per_feature_snr(vc.n_features, self.rng)
            noise_std = signal_std / snr_per_feat
            noise = self.rng.standard_normal(signal.shape) * noise_std[np.newaxis, :]

            view_data = signal + noise

            # Apply observation model
            view_data = self._apply_observation_model(view_data, vc)

            views[vc.name] = view_data
            feature_names[vc.name] = vc.feature_names

        result = {
            "views": views,
            "latent_factors": latent_factors,
            "cluster_labels": cluster_labels,
            "feature_names": feature_names,
            "ground_truth": self.get_ground_truth()
        }

        # Legacy compatibility: if exactly 2 views, add view_x/view_y keys
        if len(self.view_configs) == 2:
            names = list(views.keys())
            result["view_x"] = views[names[0]]
            result["view_y"] = views[names[1]]
            result["feature_names_x"] = feature_names[names[0]]
            result["feature_names_y"] = feature_names[names[1]]

        return result

    def _apply_observation_model(self, data: np.ndarray, view_config: ViewConfig) -> np.ndarray:
        """Apply non-Gaussian observation model (delegates to same logic as MultimodalGenerator)."""
        model = view_config.observation_model

        if model == "gaussian":
            return data
        elif model == "poisson":
            clipped = np.clip(data, -20, 20)
            return self.rng.poisson(np.exp(clipped)).astype(np.float64)
        elif model == "bernoulli":
            probs = 1.0 / (1.0 + np.exp(-data))
            return self.rng.binomial(1, probs).astype(np.float64)
        elif model == "ordinal":
            thresholds = view_config.ordinal_thresholds
            ordinal = np.zeros_like(data, dtype=np.float64)
            for threshold in thresholds:
                ordinal += (data > threshold).astype(np.float64)
            return ordinal
        else:
            raise ValueError(f"Unknown observation_model: '{model}'")

    def get_ground_truth(self) -> Dict:
        """Export all ground truth parameters."""
        gt = {
            "mixture_model": self.mixture_model.get_ground_truth(),
            "n_views": len(self.view_configs),
            "n_shared_components": self.n_shared_components,
            "true_canonical_correlations": self.true_canonical_correlations.copy(),
            "view_names": [vc.name for vc in self.view_configs],
        }

        # Per-view ground truth
        for k, vc in enumerate(self.view_configs):
            prefix = vc.name
            gt[f"true_loadings_{prefix}"] = self.loadings[k].copy()
            gt[f"snr_{prefix}"] = vc.snr
            gt[f"n_features_{prefix}"] = vc.n_features
            gt[f"observation_model_{prefix}"] = vc.observation_model
            gt[f"snr_distribution_{prefix}"] = vc.snr_distribution
            gt[f"per_view_correlations_{prefix}"] = self.per_view_correlations[k].copy()

            if vc.observation_model == "ordinal":
                gt[f"ordinal_thresholds_{prefix}"] = vc.ordinal_thresholds.tolist()

        # Legacy 2-view keys
        if len(self.view_configs) == 2:
            gt["true_loadings_x"] = self.loadings[0].copy()
            gt["true_loadings_y"] = self.loadings[1].copy()
            gt["snr_x"] = self.view_configs[0].snr
            gt["snr_y"] = self.view_configs[1].snr
            gt["n_features_x"] = self.view_configs[0].n_features
            gt["n_features_y"] = self.view_configs[1].n_features
            gt["observation_model_x"] = self.view_configs[0].observation_model
            gt["observation_model_y"] = self.view_configs[1].observation_model

        return gt


def create_multiview_generator(
    n_clusters: int = 3,
    n_latent_dims: int = 5,
    separation: float = 2.0,
    view_configs: Optional[List[ViewConfig]] = None,
    n_shared_components: Optional[int] = None,
    target_correlations: Optional[np.ndarray] = None,
    per_view_correlation_scales: Optional[List[float]] = None,
    random_state: Optional[int] = None,
    covariance_type: str = "identity",
    ar1_rho: float = 0.5,
    variance_scaling: str = "none",
    variance_scaling_magnitude: float = 1.5,
) -> MultiviewGenerator:
    """
    Factory function to create an N-view generator.

    Args:
        n_clusters: Number of biotype clusters
        n_latent_dims: Dimensionality of shared latent space
        separation: Between-cluster separation (Cohen's d)
        view_configs: List of ViewConfig for each view (required)
        n_shared_components: Number of shared latent dimensions
        target_correlations: Desired canonical correlation values
        per_view_correlation_scales: Per-view scaling of correlations
        random_state: Random seed
        covariance_type: "identity" or "ar1"
        ar1_rho: AR(1) autocorrelation
        variance_scaling: "none" or "beta"
        variance_scaling_magnitude: Scale for Beta draws

    Returns:
        Configured MultiviewGenerator

    Example:
        >>> views = [
        ...     ViewConfig("eeg", 65, modality="eeg"),
        ...     ViewConfig("clinical", 12, modality="clinical"),
        ...     ViewConfig("mri", 30, modality="mri"),
        ... ]
        >>> gen = create_multiview_generator(view_configs=views, random_state=42)
        >>> data = gen.generate(n_samples=200)
        >>> data["views"].keys()  # dict_keys(['eeg', 'clinical', 'mri'])
    """
    if view_configs is None:
        raise ValueError("view_configs is required for create_multiview_generator")

    mixture_model = BiotypeMixtureModel(
        n_clusters=n_clusters,
        n_latent_dims=n_latent_dims,
        separation=separation,
        random_state=random_state,
        covariance_type=covariance_type,
        ar1_rho=ar1_rho,
        variance_scaling=variance_scaling,
        variance_scaling_magnitude=variance_scaling_magnitude,
    )

    return MultiviewGenerator(
        mixture_model=mixture_model,
        view_configs=view_configs,
        n_shared_components=n_shared_components,
        target_correlations=target_correlations,
        per_view_correlation_scales=per_view_correlation_scales,
        random_state=random_state,
    )
