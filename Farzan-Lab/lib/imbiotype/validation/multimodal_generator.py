"""
Multimodal Data Generator for Validation Study Generator

Generates correlated multimodal data from shared latent factors.
The cross-view correlation structure is fully controlled via known loading matrices.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field

from .mixture_model import BiotypeMixtureModel
from .modality_templates import generate_feature_names, get_template


@dataclass
class ViewConfig:
    """Configuration for a single data view/modality."""
    name: str
    n_features: int
    modality: str = "generic"
    categories: Optional[List[str]] = None
    cross_product: Optional[Tuple[str, str]] = None
    snr: float = 2.0  # Signal-to-noise ratio
    feature_names: Optional[List[str]] = None

    # Enhancement 1: Non-Gaussian observation model
    observation_model: str = "gaussian"  # gaussian, poisson, bernoulli, ordinal
    ordinal_n_levels: int = 5  # Number of ordinal levels (for observation_model="ordinal")
    ordinal_thresholds: Optional[np.ndarray] = None  # Custom thresholds; auto-generated if None

    # Enhancement 3: Per-feature heterogeneous SNR
    snr_distribution: str = "uniform"  # "uniform" (all same) or "beta" (per-feature variation)
    snr_beta_a: float = 1.0  # Beta(a, b) shape parameter
    snr_beta_b: float = 1.0  # Beta(a, b) shape parameter
    snr_scale: float = 1.5   # Multiplier on Beta draws

    def __post_init__(self):
        """Generate feature names if not provided."""
        if self.feature_names is None:
            if self.modality != "generic":
                self.feature_names = generate_feature_names(
                    self.modality,
                    categories=self.categories,
                    n_features=self.n_features,
                    cross_product=self.cross_product
                )
            else:
                self.feature_names = [f"{self.name}_feat_{i}" for i in range(self.n_features)]

        # Ensure we have enough feature names
        if len(self.feature_names) < self.n_features:
            # Pad with generic names
            for i in range(len(self.feature_names), self.n_features):
                self.feature_names.append(f"{self.name}_feat_{i}")
        elif len(self.feature_names) > self.n_features:
            self.feature_names = self.feature_names[:self.n_features]

        # Auto-generate ordinal thresholds if needed
        if self.observation_model == "ordinal" and self.ordinal_thresholds is None:
            # Evenly spaced thresholds from standard normal quantiles
            from scipy.stats import norm
            n_thresholds = self.ordinal_n_levels - 1
            probs = np.linspace(0, 1, n_thresholds + 2)[1:-1]  # Exclude 0 and 1
            self.ordinal_thresholds = norm.ppf(probs)

    def get_per_feature_snr(self, n_features: int, rng: np.random.Generator) -> np.ndarray:
        """
        Get per-feature SNR values based on distribution config.

        Returns:
            Array of shape (n_features,) with SNR values per feature.
        """
        if self.snr_distribution == "uniform":
            return np.full(n_features, self.snr)
        elif self.snr_distribution == "beta":
            beta_samples = rng.beta(self.snr_beta_a, self.snr_beta_b, n_features)
            per_feature_snr = self.snr * self.snr_scale * beta_samples
            # Floor at 0.1 to avoid near-zero SNR (infinite noise)
            return np.maximum(per_feature_snr, 0.1)
        else:
            raise ValueError(f"Unknown snr_distribution: {self.snr_distribution}")


@dataclass
class MultimodalGenerator:
    """
    Generates correlated multimodal data with known ground truth.

    The generative model:
        z_i ~ BiotypeMixtureModel (shared latent factors)
        X_i = W_x @ z_i + ε_x  (view X)
        Y_i = W_y @ z_i + ε_y  (view Y)

    The canonical correlations between X and Y arise from the shared latent z.
    W_x and W_y are the true loading matrices (ground truth for CCA recovery).

    Attributes:
        mixture_model: BiotypeMixtureModel for latent factors
        view_x_config: Configuration for view X
        view_y_config: Configuration for view Y
        n_shared_components: Number of shared latent dimensions used for correlation
        target_correlations: Desired canonical correlation values
        random_state: Random seed
    """
    mixture_model: BiotypeMixtureModel
    view_x_config: ViewConfig
    view_y_config: ViewConfig
    n_shared_components: Optional[int] = None
    target_correlations: Optional[np.ndarray] = None
    random_state: Optional[int] = None

    # Generated matrices (populated during generation)
    loadings_x: np.ndarray = field(default=None, init=False)
    loadings_y: np.ndarray = field(default=None, init=False)
    true_canonical_correlations: np.ndarray = field(default=None, init=False)

    def __post_init__(self):
        """Initialize loading matrices."""
        self.rng = np.random.default_rng(self.random_state)

        if self.n_shared_components is None:
            self.n_shared_components = min(
                self.mixture_model.n_latent_dims,
                self.view_x_config.n_features,
                self.view_y_config.n_features
            )

        if self.target_correlations is None:
            # Default: decreasing correlations from 0.9 to 0.3
            self.target_correlations = np.linspace(0.9, 0.3, self.n_shared_components)

        self._generate_loading_matrices()

    def _generate_loading_matrices(self):
        """
        Generate loading matrices that produce target canonical correlations.

        The loading matrices W_x and W_y map from latent space to observed features.
        They are constructed to achieve specific canonical correlation values.

        The correlation structure is achieved by:
        1. Both views share the first n_shared latent dimensions
        2. For view Y, the shared dimensions are scaled by target_correlations
           and mixed with independent noise to achieve exact correlations
        """
        n_latent = self.mixture_model.n_latent_dims
        n_x = self.view_x_config.n_features
        n_y = self.view_y_config.n_features
        n_shared = self.n_shared_components

        # Validate dimensions
        if n_shared > n_latent:
            import warnings
            warnings.warn(
                f"n_shared_components ({n_shared}) > n_latent_dims ({n_latent}). "
                f"Truncating to {n_latent}."
            )
            n_shared = n_latent
            self.n_shared_components = n_shared

        if n_x < n_latent or n_y < n_latent:
            import warnings
            warnings.warn(
                f"n_features ({min(n_x, n_y)}) < n_latent_dims ({n_latent}). "
                f"Loading matrices will be truncated."
            )

        # Generate random orthonormal bases for each view
        # View X loadings: (n_features_x, n_latent)
        W_x_random = self.rng.standard_normal((n_x, n_latent))
        W_x_orth, _ = np.linalg.qr(W_x_random)
        self.loadings_x = W_x_orth[:, :min(n_latent, n_x)]

        # View Y loadings: (n_features_y, n_latent)
        W_y_random = self.rng.standard_normal((n_y, n_latent))
        W_y_orth, _ = np.linalg.qr(W_y_random)
        self.loadings_y = W_y_orth[:, :min(n_latent, n_y)]

        # Validate and store target correlations
        if len(self.target_correlations) < n_shared:
            import warnings
            warnings.warn(
                f"target_correlations length ({len(self.target_correlations)}) < "
                f"n_shared_components ({n_shared}). Padding with zeros."
            )
            self.target_correlations = np.pad(
                self.target_correlations,
                (0, n_shared - len(self.target_correlations)),
                constant_values=0.0
            )

        self.true_canonical_correlations = self.target_correlations[:n_shared].copy()

    def generate(self, n_samples: int,
                 balance: str = "balanced",
                 imbalance_ratio: float = 0.8) -> Dict:
        """
        Generate multimodal data with known ground truth.

        The correlation structure is implemented as:
        - View X uses latent factors directly: X = Z @ W_x.T + noise
        - View Y uses correlated latent factors for shared dimensions:
          Z_y_shared = rho * Z_shared + sqrt(1-rho^2) * E  (where E is independent noise)
          This achieves exact canonical correlations of rho for each shared dimension.

        Args:
            n_samples: Number of samples to generate
            balance: "balanced" or "imbalanced" cluster sizes
            imbalance_ratio: If imbalanced, proportion for largest cluster

        Returns:
            Dictionary containing:
                - view_x: (n_samples, n_features_x) observed data
                - view_y: (n_samples, n_features_y) observed data
                - latent_factors: (n_samples, n_latent) shared latent z
                - cluster_labels: (n_samples,) true cluster assignments
                - feature_names_x: List of feature names for view X
                - feature_names_y: List of feature names for view Y
                - ground_truth: Dict of all ground truth parameters
        """
        # Sample latent factors and cluster labels
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
        rho = self.true_canonical_correlations

        # --- Generate View X signal ---
        # X = Z @ W_x.T (direct projection)
        n_latent_x = min(n_latent, self.loadings_x.shape[1])
        signal_x = latent_factors[:, :n_latent_x] @ self.loadings_x[:, :n_latent_x].T

        # --- Generate View Y signal with correlation structure ---
        # For shared dimensions: Z_y = rho * Z + sigma * sqrt(1-rho^2) * E
        # where sigma = std(Z) per dimension. This achieves Corr(Z, Z_y) = rho exactly
        # even when Z has variance != 1 (e.g., due to cluster separation).
        n_latent_y = min(n_latent, self.loadings_y.shape[1])

        # Create correlated latent factors for Y
        latent_for_y = np.zeros((n_samples, n_latent_y))

        # Shared dimensions: inject target correlations
        n_shared_actual = min(n_shared, n_latent_y, len(rho))
        if n_shared_actual > 0:
            Z_shared = latent_factors[:, :n_shared_actual]
            independent_noise = self.rng.standard_normal((n_samples, n_shared_actual))

            # Compute per-dimension std of Z (may be > 1 due to cluster separation)
            Z_std = np.std(Z_shared, axis=0)
            Z_std = np.maximum(Z_std, 1e-10)  # Avoid division by zero

            # Z_y = rho * Z + sigma * sqrt(1-rho^2) * E
            # This ensures: Var(Z_y) = rho^2 * sigma^2 + sigma^2 * (1-rho^2) = sigma^2
            #               Cov(Z, Z_y) = rho * sigma^2
            #               Corr(Z, Z_y) = rho * sigma^2 / (sigma * sigma) = rho
            rho_actual = rho[:n_shared_actual]
            sqrt_one_minus_rho_sq = np.sqrt(np.maximum(0, 1 - rho_actual ** 2))
            latent_for_y[:, :n_shared_actual] = (
                Z_shared * rho_actual + independent_noise * Z_std * sqrt_one_minus_rho_sq
            )

        # Remaining dimensions: use independent noise (view-specific variation)
        if n_latent_y > n_shared_actual:
            latent_for_y[:, n_shared_actual:] = self.rng.standard_normal(
                (n_samples, n_latent_y - n_shared_actual)
            )

        signal_y = latent_for_y @ self.loadings_y[:, :n_latent_y].T

        # --- Add observation noise based on per-feature SNR ---
        # SNR = signal_std / noise_std, so noise_std = signal_std / SNR
        # Use per-feature signal std for correct SNR
        signal_std_x = np.std(signal_x, axis=0)  # (n_features_x,)
        signal_std_y = np.std(signal_y, axis=0)  # (n_features_y,)

        # Avoid division by zero for constant features
        signal_std_x = np.maximum(signal_std_x, 1e-10)
        signal_std_y = np.maximum(signal_std_y, 1e-10)

        # Enhancement 3: Per-feature heterogeneous SNR
        snr_x = self.view_x_config.get_per_feature_snr(signal_x.shape[1], self.rng)
        snr_y = self.view_y_config.get_per_feature_snr(signal_y.shape[1], self.rng)

        noise_std_x = signal_std_x / snr_x  # Per-feature noise std
        noise_std_y = signal_std_y / snr_y

        noise_x = self.rng.standard_normal(signal_x.shape) * noise_std_x[np.newaxis, :]
        noise_y = self.rng.standard_normal(signal_y.shape) * noise_std_y[np.newaxis, :]

        view_x = signal_x + noise_x
        view_y = signal_y + noise_y

        # Enhancement 1: Apply non-Gaussian observation model
        view_x = self._apply_observation_model(view_x, self.view_x_config)
        view_y = self._apply_observation_model(view_y, self.view_y_config)

        return {
            "view_x": view_x,
            "view_y": view_y,
            "latent_factors": latent_factors,
            "cluster_labels": cluster_labels,
            "feature_names_x": self.view_x_config.feature_names,
            "feature_names_y": self.view_y_config.feature_names,
            "ground_truth": self.get_ground_truth()
        }

    def _apply_observation_model(self, data: np.ndarray, view_config: ViewConfig) -> np.ndarray:
        """
        Apply non-Gaussian observation model via link function.

        Args:
            data: Continuous Gaussian data (n_samples, n_features)
            view_config: ViewConfig with observation_model setting

        Returns:
            Transformed data array
        """
        model = view_config.observation_model

        if model == "gaussian":
            return data

        elif model == "poisson":
            # Count data: Y ~ Poisson(lambda=exp(X))
            # Clip X to avoid overflow in exp (exp(20) ~ 5e8, plenty for counts)
            clipped = np.clip(data, -20, 20)
            lambdas = np.exp(clipped)
            return self.rng.poisson(lambdas).astype(np.float64)

        elif model == "bernoulli":
            # Binary data: Y ~ Bernoulli(p=sigmoid(X))
            probs = 1.0 / (1.0 + np.exp(-data))
            return self.rng.binomial(1, probs).astype(np.float64)

        elif model == "ordinal":
            # Ordinal/Likert data: Y = sum(X > threshold_k) for k thresholds
            thresholds = view_config.ordinal_thresholds
            # Result: count of thresholds exceeded, giving values in [0, n_levels-1]
            ordinal = np.zeros_like(data, dtype=np.float64)
            for threshold in thresholds:
                ordinal += (data > threshold).astype(np.float64)
            return ordinal

        else:
            raise ValueError(
                f"Unknown observation_model: '{model}'. "
                f"Supported: 'gaussian', 'poisson', 'bernoulli', 'ordinal'."
            )

    def get_ground_truth(self) -> Dict:
        """
        Export all ground truth parameters.

        Returns:
            Dictionary containing:
                - mixture_model_params: BiotypeMixtureModel ground truth
                - true_loadings_x: W_x matrix
                - true_loadings_y: W_y matrix
                - true_canonical_correlations: Target correlation values
                - n_shared_components: Number of shared dimensions
                - snr_x: Signal-to-noise ratio for view X
                - snr_y: Signal-to-noise ratio for view Y
                - observation_model_x/y: Observation model type per view
                - snr_distribution_x/y: SNR distribution type per view
        """
        gt = {
            "mixture_model": self.mixture_model.get_ground_truth(),
            "true_loadings_x": self.loadings_x.copy(),
            "true_loadings_y": self.loadings_y.copy(),
            "true_canonical_correlations": self.true_canonical_correlations.copy(),
            "n_shared_components": self.n_shared_components,
            "snr_x": self.view_x_config.snr,
            "snr_y": self.view_y_config.snr,
            "n_features_x": self.view_x_config.n_features,
            "n_features_y": self.view_y_config.n_features,
            # Enhancement 1: observation model
            "observation_model_x": self.view_x_config.observation_model,
            "observation_model_y": self.view_y_config.observation_model,
            # Enhancement 3: SNR distribution
            "snr_distribution_x": self.view_x_config.snr_distribution,
            "snr_distribution_y": self.view_y_config.snr_distribution,
        }

        # Record ordinal thresholds if used
        if self.view_x_config.observation_model == "ordinal":
            gt["ordinal_thresholds_x"] = self.view_x_config.ordinal_thresholds.tolist()
        if self.view_y_config.observation_model == "ordinal":
            gt["ordinal_thresholds_y"] = self.view_y_config.ordinal_thresholds.tolist()

        return gt


def create_two_view_generator(
    n_clusters: int = 3,
    n_latent_dims: int = 5,
    separation: float = 2.0,
    n_features_x: int = 50,
    n_features_y: int = 30,
    modality_x: str = "eeg",
    modality_y: str = "clinical",
    categories_x: Optional[List[str]] = None,
    categories_y: Optional[List[str]] = None,
    snr_x: float = 2.0,
    snr_y: float = 2.0,
    target_correlations: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
    # Enhancement 1: observation models
    observation_model_x: str = "gaussian",
    observation_model_y: str = "gaussian",
    ordinal_n_levels_x: int = 5,
    ordinal_n_levels_y: int = 5,
    # Enhancement 2: covariance structure (passed to mixture model)
    covariance_type: str = "identity",
    ar1_rho: float = 0.5,
    variance_scaling: str = "none",
    variance_scaling_magnitude: float = 1.5,
    # Enhancement 3: per-feature SNR
    snr_distribution_x: str = "uniform",
    snr_distribution_y: str = "uniform",
    snr_beta_a: float = 1.0,
    snr_beta_b: float = 1.0,
    snr_scale: float = 1.5,
) -> MultimodalGenerator:
    """
    Factory function to create a two-view generator with common configurations.

    Args:
        n_clusters: Number of biotype clusters
        n_latent_dims: Dimensionality of shared latent space
        separation: Between-cluster separation (Cohen's d)
        n_features_x: Number of features in view X
        n_features_y: Number of features in view Y
        modality_x: Modality type for view X (eeg, mri, clinical, etc.)
        modality_y: Modality type for view Y
        categories_x: Specific categories from modality_x
        categories_y: Specific categories from modality_y
        snr_x: Signal-to-noise ratio for view X
        snr_y: Signal-to-noise ratio for view Y
        target_correlations: Desired canonical correlation values
        random_state: Random seed

    Returns:
        Configured MultimodalGenerator

    Example:
        >>> generator = create_two_view_generator(
        ...     n_clusters=3,
        ...     n_features_x=65,  # 5 bands × 13 channels
        ...     n_features_y=12,
        ...     modality_x="eeg",
        ...     modality_y="clinical",
        ...     categories_x=["power_bands"],
        ...     categories_y=["depression", "anxiety"],
        ...     random_state=42
        ... )
        >>> data = generator.generate(n_samples=200)
    """
    from .mixture_model import BiotypeMixtureModel

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

    view_x_config = ViewConfig(
        name="view_x",
        n_features=n_features_x,
        modality=modality_x,
        categories=categories_x,
        snr=snr_x,
        observation_model=observation_model_x,
        ordinal_n_levels=ordinal_n_levels_x,
        snr_distribution=snr_distribution_x,
        snr_beta_a=snr_beta_a,
        snr_beta_b=snr_beta_b,
        snr_scale=snr_scale,
    )

    view_y_config = ViewConfig(
        name="view_y",
        n_features=n_features_y,
        modality=modality_y,
        categories=categories_y,
        snr=snr_y,
        observation_model=observation_model_y,
        ordinal_n_levels=ordinal_n_levels_y,
        snr_distribution=snr_distribution_y,
        snr_beta_a=snr_beta_a,
        snr_beta_b=snr_beta_b,
        snr_scale=snr_scale,
    )

    return MultimodalGenerator(
        mixture_model=mixture_model,
        view_x_config=view_x_config,
        view_y_config=view_y_config,
        target_correlations=target_correlations,
        random_state=random_state
    )
