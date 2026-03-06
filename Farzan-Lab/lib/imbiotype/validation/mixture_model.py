"""
Biotype Mixture Model for Validation Study Generator

Defines ground truth latent biotype structure as a Gaussian mixture model.
Samples are drawn from clusters with known centroids, covariances, and proportions.
"""

import numpy as np
from typing import Optional, Dict, List, Tuple, Union
from dataclasses import dataclass, field


def _build_ar1_matrix(n_dims: int, rho: float) -> np.ndarray:
    """
    Build AR(1) correlation matrix: R[i,j] = rho^|i-j|.

    Args:
        n_dims: Matrix dimension
        rho: Autocorrelation coefficient (0 to 1)

    Returns:
        (n_dims, n_dims) correlation matrix with AR(1) structure
    """
    i, j = np.ogrid[:n_dims, :n_dims]
    return rho ** np.abs(i - j)


@dataclass
class BiotypeMixtureModel:
    """
    Gaussian mixture model representing ground truth biotypes.

    The model generates samples z_i from a mixture of Gaussians in a shared
    latent space. These latent factors are then projected to observed features
    via loading matrices in the multimodal generator.

    Attributes:
        n_clusters: Number of biotype clusters (k)
        n_latent_dims: Dimensionality of shared latent space
        centroids: Cluster centroids in latent space (k, n_latent_dims)
        covariances: Cluster covariances (k, n_latent_dims, n_latent_dims)
        mixing_proportions: Cluster weights (k,), sum to 1
        separation: Cohen's d-like separation between clusters
        covariance_type: "identity" or "ar1" covariance structure
        ar1_rho: Autocorrelation coefficient for AR(1) covariance
        variance_scaling: "none" or "beta" for per-dimension variance heterogeneity
        variance_scaling_magnitude: Scale factor when variance_scaling="beta"
        random_state: Random seed for reproducibility
    """
    n_clusters: int
    n_latent_dims: int
    centroids: Optional[np.ndarray] = None
    covariances: Optional[np.ndarray] = None
    mixing_proportions: Optional[np.ndarray] = None
    separation: float = 2.0  # Cohen's d equivalent
    # Enhancement 2: Covariance structure
    covariance_type: str = "identity"  # "identity" or "ar1"
    ar1_rho: float = 0.5  # AR(1) autocorrelation
    variance_scaling: str = "none"  # "none" or "beta"
    variance_scaling_magnitude: float = 1.5  # Scale for Beta(1,1) draws
    random_state: Optional[int] = None

    def __post_init__(self):
        """Initialize or validate model parameters."""
        self.rng = np.random.default_rng(self.random_state)

        # Initialize centroids if not provided
        if self.centroids is None:
            self.centroids = self._generate_separated_centroids()

        # Initialize covariances if not provided
        if self.covariances is None:
            self.covariances = self._generate_covariances()

        # Initialize mixing proportions if not provided
        if self.mixing_proportions is None:
            self.mixing_proportions = np.ones(self.n_clusters) / self.n_clusters

        self._validate()

    def _validate(self):
        """Validate model parameters."""
        assert self.centroids.shape == (self.n_clusters, self.n_latent_dims), \
            f"Centroids shape {self.centroids.shape} != ({self.n_clusters}, {self.n_latent_dims})"

        assert self.covariances.shape == (self.n_clusters, self.n_latent_dims, self.n_latent_dims), \
            f"Covariances shape {self.covariances.shape} != ({self.n_clusters}, {self.n_latent_dims}, {self.n_latent_dims})"

        assert len(self.mixing_proportions) == self.n_clusters, \
            f"Mixing proportions length {len(self.mixing_proportions)} != {self.n_clusters}"

        assert np.isclose(self.mixing_proportions.sum(), 1.0), \
            f"Mixing proportions sum to {self.mixing_proportions.sum()}, expected 1.0"

    def _generate_separated_centroids(self) -> np.ndarray:
        """
        Generate cluster centroids with specified separation.

        For k=2: Places centroids along first axis at distance = separation.
        For k>2: Uses a regular simplex arrangement where all pairwise distances
                 equal the separation parameter (Cohen's d equivalent).

        The separation parameter is the Euclidean distance between any two
        cluster centroids (with identity covariance, this equals Cohen's d).
        """
        if self.n_clusters == 1:
            return np.zeros((1, self.n_latent_dims))

        if self.n_clusters == 2:
            # Two clusters: opposite directions along first axis
            centroids = np.zeros((2, self.n_latent_dims))
            centroids[0, 0] = -self.separation / 2
            centroids[1, 0] = self.separation / 2
            return centroids

        # For k > 2: construct regular simplex with uniform pairwise distances
        # A regular (k-1)-simplex has k vertices, all pairs at equal distance
        # We embed it in n_latent_dims and scale to achieve target separation
        k = self.n_clusters
        d = min(k - 1, self.n_latent_dims)  # Simplex requires k-1 dimensions

        if self.n_latent_dims < k - 1:
            import warnings
            warnings.warn(
                f"n_latent_dims ({self.n_latent_dims}) < n_clusters-1 ({k-1}). "
                f"Cannot embed {k}-point simplex in {self.n_latent_dims}D space. "
                f"Some cluster centroids will overlap, reducing effective separation."
            )

        # Construct centered regular simplex using the standard method:
        # Start with k points, each point i has coordinate i = 1 on axis i, else 0
        # Then center and orthonormalize
        simplex = np.zeros((k, d))

        for i in range(k):
            if i < d:
                simplex[i, i] = 1.0

        # Center the simplex
        simplex = simplex - simplex.mean(axis=0)

        # Orthonormalize to get uniform distances
        # After centering, apply SVD to get a proper simplex configuration
        U, S, Vt = np.linalg.svd(simplex, full_matrices=False)
        # Use the left singular vectors (orthonormal representation)
        simplex_orth = U[:, :d] * np.sqrt(k)  # Scale factor for centering

        # Current pairwise distance in simplex_orth
        current_dist = np.linalg.norm(simplex_orth[0] - simplex_orth[1])
        if current_dist > 1e-10:
            scale = self.separation / current_dist
        else:
            scale = self.separation

        centroids_low_dim = simplex_orth * scale

        # Embed in full latent space
        centroids = np.zeros((k, self.n_latent_dims))
        centroids[:, :d] = centroids_low_dim

        # Add small random rotation if n_latent_dims > k-1 for variety
        if self.n_latent_dims > d:
            # Random orthogonal rotation matrix
            Q, _ = np.linalg.qr(self.rng.standard_normal((self.n_latent_dims, self.n_latent_dims)))
            centroids = centroids @ Q

        return centroids

    def _generate_covariances(self) -> np.ndarray:
        """
        Generate cluster covariance matrices.

        Supports:
        - "identity": Spherical clusters (default, backward compatible)
        - "ar1": AR(1) correlation structure rho^|i-j|, optionally with
          per-dimension Beta-distributed variance scaling (matches R script).
        """
        covariances = np.zeros((self.n_clusters, self.n_latent_dims, self.n_latent_dims))

        if self.covariance_type == "identity":
            for k in range(self.n_clusters):
                covariances[k] = np.eye(self.n_latent_dims)

        elif self.covariance_type == "ar1":
            # Build AR(1) correlation matrix: R[i,j] = rho^|i-j|
            corr_matrix = _build_ar1_matrix(self.n_latent_dims, self.ar1_rho)

            for k in range(self.n_clusters):
                if self.variance_scaling == "beta":
                    # Per-dimension variance scaling via Beta(1,1) * magnitude
                    # Beta(1,1) = Uniform[0,1], so scales are in [0, magnitude]
                    scales = self.rng.beta(1, 1, self.n_latent_dims) * self.variance_scaling_magnitude
                    # Ensure no zero scales
                    scales = np.maximum(scales, 0.01)
                    D = np.diag(scales)
                    # Sigma = D @ R @ D (variance-scaled correlation)
                    covariances[k] = D @ corr_matrix @ D
                else:
                    # Pure AR(1) correlation (unit variance per dimension)
                    covariances[k] = corr_matrix.copy()
        else:
            raise ValueError(
                f"Unknown covariance_type: '{self.covariance_type}'. "
                f"Supported: 'identity', 'ar1'."
            )

        return covariances

    def sample(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample from the mixture model.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Tuple of:
                - latent_factors: (n_samples, n_latent_dims) array of latent z values
                - cluster_labels: (n_samples,) array of true cluster assignments
        """
        # Determine cluster assignments based on mixing proportions
        cluster_labels = self.rng.choice(
            self.n_clusters,
            size=n_samples,
            p=self.mixing_proportions
        )

        # Sample latent factors from each cluster's Gaussian
        latent_factors = np.zeros((n_samples, self.n_latent_dims))

        for k in range(self.n_clusters):
            mask = cluster_labels == k
            n_k = mask.sum()
            if n_k > 0:
                latent_factors[mask] = self.rng.multivariate_normal(
                    mean=self.centroids[k],
                    cov=self.covariances[k],
                    size=n_k
                )

        return latent_factors, cluster_labels

    def sample_balanced(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample with exactly balanced cluster sizes.

        Args:
            n_samples: Total number of samples (will be rounded to nearest multiple of k)

        Returns:
            Tuple of latent_factors, cluster_labels
        """
        samples_per_cluster = n_samples // self.n_clusters
        remainder = n_samples % self.n_clusters

        cluster_labels = []
        latent_factors_list = []

        for k in range(self.n_clusters):
            n_k = samples_per_cluster + (1 if k < remainder else 0)
            cluster_labels.extend([k] * n_k)
            samples = self.rng.multivariate_normal(
                mean=self.centroids[k],
                cov=self.covariances[k],
                size=n_k
            )
            latent_factors_list.append(samples)

        latent_factors = np.vstack(latent_factors_list)
        cluster_labels = np.array(cluster_labels)

        # Shuffle to mix clusters
        shuffle_idx = self.rng.permutation(len(cluster_labels))
        return latent_factors[shuffle_idx], cluster_labels[shuffle_idx]

    def sample_imbalanced(self, n_samples: int,
                          imbalance_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample with imbalanced cluster sizes.

        Args:
            n_samples: Total number of samples
            imbalance_ratio: Fraction of samples in the largest cluster (0.5-0.99)

        Returns:
            Tuple of latent_factors, cluster_labels
        """
        if self.n_clusters == 1:
            # With only one cluster, imbalance is meaningless - just sample normally
            return self.sample(n_samples)

        if not 0.5 <= imbalance_ratio < 1.0:
            raise ValueError("imbalance_ratio must be in [0.5, 1.0)")

        # Create imbalanced proportions
        # Largest cluster gets imbalance_ratio, rest share remainder
        proportions = np.zeros(self.n_clusters)
        proportions[0] = imbalance_ratio
        remaining = 1.0 - imbalance_ratio
        proportions[1:] = remaining / (self.n_clusters - 1)

        # Sample using local proportions (don't modify self.mixing_proportions)
        cluster_labels = self.rng.choice(
            self.n_clusters,
            size=n_samples,
            p=proportions
        )

        # Sample latent factors from each cluster's Gaussian
        latent_factors = np.zeros((n_samples, self.n_latent_dims))

        for k in range(self.n_clusters):
            mask = cluster_labels == k
            n_k = mask.sum()
            if n_k > 0:
                latent_factors[mask] = self.rng.multivariate_normal(
                    mean=self.centroids[k],
                    cov=self.covariances[k],
                    size=n_k
                )

        return latent_factors, cluster_labels

    def compute_pairwise_distances(self) -> np.ndarray:
        """
        Compute pairwise Mahalanobis distances between cluster centroids.

        Returns:
            (n_clusters, n_clusters) distance matrix
        """
        distances = np.zeros((self.n_clusters, self.n_clusters))

        for i in range(self.n_clusters):
            for j in range(i + 1, self.n_clusters):
                diff = self.centroids[i] - self.centroids[j]
                # Use average covariance for Mahalanobis distance
                avg_cov = (self.covariances[i] + self.covariances[j]) / 2
                # Use pinv for robustness against singular/near-singular covariances
                avg_cov_inv = np.linalg.pinv(avg_cov)
                d = np.sqrt(np.maximum(0, diff @ avg_cov_inv @ diff))  # Clamp to 0 for numerical stability
                distances[i, j] = d
                distances[j, i] = d

        return distances

    def get_ground_truth(self) -> Dict:
        """
        Export ground truth parameters for validation.

        Returns:
            Dictionary containing all ground truth model parameters
        """
        return {
            "n_clusters": self.n_clusters,
            "n_latent_dims": self.n_latent_dims,
            "cluster_centroids": self.centroids.copy(),
            "cluster_covariances": self.covariances.copy(),
            "mixing_proportions": self.mixing_proportions.copy(),
            "separation": self.separation,
            "pairwise_distances": self.compute_pairwise_distances(),
            "covariance_type": self.covariance_type,
            "ar1_rho": self.ar1_rho if self.covariance_type == "ar1" else None,
            "variance_scaling": self.variance_scaling,
            "random_state": self.random_state
        }


def create_mixture_model(
    n_clusters: int = 3,
    n_latent_dims: int = 5,
    separation: float = 2.0,
    balance: str = "balanced",
    imbalance_ratio: float = 0.8,
    random_state: Optional[int] = None,
    covariance_type: str = "identity",
    ar1_rho: float = 0.5,
    variance_scaling: str = "none",
    variance_scaling_magnitude: float = 1.5,
) -> BiotypeMixtureModel:
    """
    Factory function to create a BiotypeMixtureModel with common configurations.

    Args:
        n_clusters: Number of biotype clusters
        n_latent_dims: Dimensionality of shared latent space
        separation: Between-cluster separation (Cohen's d equivalent)
        balance: "balanced" for equal proportions, "imbalanced" for skewed
        imbalance_ratio: If imbalanced, proportion for largest cluster
        random_state: Random seed

    Returns:
        Configured BiotypeMixtureModel
    """
    rng = np.random.default_rng(random_state)

    if balance == "balanced":
        mixing_proportions = np.ones(n_clusters) / n_clusters
    elif balance == "imbalanced":
        mixing_proportions = np.zeros(n_clusters)
        mixing_proportions[0] = imbalance_ratio
        remaining = 1.0 - imbalance_ratio
        mixing_proportions[1:] = remaining / (n_clusters - 1)
    else:
        raise ValueError(f"Unknown balance: {balance}")

    return BiotypeMixtureModel(
        n_clusters=n_clusters,
        n_latent_dims=n_latent_dims,
        separation=separation,
        mixing_proportions=mixing_proportions,
        random_state=random_state,
        covariance_type=covariance_type,
        ar1_rho=ar1_rho,
        variance_scaling=variance_scaling,
        variance_scaling_magnitude=variance_scaling_magnitude,
    )
