"""
PermutationTester - Compare distributions of real vs permuted test correlations.

Based on Nichols & Holmes (2002) - Nonparametric Permutation Tests for Functional Neuroimaging.
https://link.springer.com/article/10.1023/A:1024068626366

Key insight: Don't compare single means. Compare the FULL DISTRIBUTION of test correlations.
"""

from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class PermutationResult:
    """Results from permutation testing."""
    observed_distribution: np.ndarray  # (n_samples,) real test correlations
    null_distribution: np.ndarray      # (n_samples,) permuted test correlations
    observed_mean: float
    null_mean: float
    separation: str  # 'clear', 'moderate', 'overlapping'


class PermutationTester:
    """
    Permutation testing for CCA/pipeline validation.

    Compares the distribution of test correlations from real data
    against the distribution from permuted data.

    Based on Nichols & Holmes (2002):
    - Run validation on real data → distribution of 1000 test correlations
    - Run validation on permuted data → null distribution
    - Compare: How separated are the distributions?

    Example:
        tester = PermutationTester(n_permutations=1)

        # Run real validation
        real_corrs = run_nested_cv(X, Y)  # Returns 1000 test correlations

        # Run permuted validation
        Y_perm = permute_rows(Y)
        perm_corrs = run_nested_cv(X, Y_perm)

        result = tester.compare_distributions(real_corrs, perm_corrs)
        print(result.separation)  # 'clear' if real >> permuted
    """

    def __init__(self, n_permutations: int = 1, random_state: int = 42):
        """
        Initialize permutation tester.

        Args:
            n_permutations: Number of permutation runs.
                           Usually 1 since CV already provides 1000 test correlations.
            random_state: Random seed for reproducibility
        """
        self.n_permutations = n_permutations
        self.random_state = random_state

    def permute_y(
        self,
        Y: np.ndarray,
        permutation_idx: int = 0
    ) -> np.ndarray:
        """
        Permute rows of Y matrix.

        This breaks the association between X and Y while preserving
        the structure of Y itself.

        Args:
            Y: Data matrix (n_samples x n_features)
            permutation_idx: Index of permutation (for reproducibility)

        Returns:
            Permuted Y matrix
        """
        rng = np.random.RandomState(self.random_state + permutation_idx)
        perm_idx = rng.permutation(len(Y))
        return Y[perm_idx].copy()

    def compare_distributions(
        self,
        observed: np.ndarray,
        null: np.ndarray
    ) -> PermutationResult:
        """
        Compare observed and null distributions.

        Args:
            observed: Distribution of test correlations from real data
            null: Distribution of test correlations from permuted data

        Returns:
            PermutationResult with separation assessment
        """
        observed = np.asarray(observed).flatten()
        null = np.asarray(null).flatten()

        observed_mean = float(np.mean(observed))
        null_mean = float(np.mean(null))

        separation = self._assess_separation(observed, null)

        return PermutationResult(
            observed_distribution=observed,
            null_distribution=null,
            observed_mean=observed_mean,
            null_mean=null_mean,
            separation=separation
        )

    def _assess_separation(
        self,
        observed: np.ndarray,
        null: np.ndarray
    ) -> str:
        """
        Assess how well-separated the distributions are.

        Returns:
            'clear': Less than 5% overlap
            'moderate': 5-20% overlap
            'overlapping': More than 20% overlap
        """
        # What proportion of observed values fall below the 95th percentile of null?
        null_95 = np.percentile(null, 95)
        overlap = np.mean(observed < null_95)

        if overlap < 0.05:
            return "clear"
        elif overlap < 0.20:
            return "moderate"
        else:
            return "overlapping"

    def compute_p_value(
        self,
        observed: np.ndarray,
        null: np.ndarray,
        alternative: str = 'greater'
    ) -> float:
        """
        Compute empirical p-value from distributions.

        Args:
            observed: Observed test statistics
            null: Null distribution test statistics
            alternative: 'greater' (observed > null) or 'two-sided'

        Returns:
            Empirical p-value
        """
        observed_mean = np.mean(observed)

        if alternative == 'greater':
            # Proportion of null values >= observed mean
            p_value = np.mean(null >= observed_mean)
        elif alternative == 'two-sided':
            observed_abs = np.abs(observed_mean - np.mean(null))
            null_centered = null - np.mean(null)
            p_value = np.mean(np.abs(null_centered) >= observed_abs)
        else:
            raise ValueError(f"Unknown alternative: {alternative}")

        return float(p_value)

    def run_permutation_test(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        validation_fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> PermutationResult:
        """
        Run complete permutation test.

        Args:
            X: X data (n_samples x n_features_x)
            Y: Y data (n_samples x n_features_y)
            validation_fn: Function that runs validation and returns test correlations.
                          Should take (X, Y) and return array of test correlations.
            progress_callback: Optional callback (current, total) for progress updates

        Returns:
            PermutationResult with observed vs null distributions
        """
        # Run real validation
        if progress_callback:
            progress_callback(0, self.n_permutations + 1)

        observed = validation_fn(X, Y)

        # Run permuted validations
        all_null = []
        for perm_idx in range(self.n_permutations):
            if progress_callback:
                progress_callback(perm_idx + 1, self.n_permutations + 1)

            Y_perm = self.permute_y(Y, perm_idx)
            null_corrs = validation_fn(X, Y_perm)
            all_null.extend(null_corrs.flatten())

        null = np.array(all_null)

        return self.compare_distributions(observed, null)

    @staticmethod
    def visualize_distributions(
        observed: np.ndarray,
        null: np.ndarray,
        title: str = "Permutation Test Results"
    ) -> 'matplotlib.figure.Figure':
        """
        Create histogram visualization of observed vs null distributions.

        Args:
            observed: Observed distribution
            null: Null distribution
            title: Plot title

        Returns:
            Matplotlib figure
        """
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histograms
        ax.hist(null, bins=50, alpha=0.7, label='Null (permuted)', color='red')
        ax.hist(observed, bins=50, alpha=0.7, label='Observed', color='blue')

        # Add means
        ax.axvline(np.mean(null), color='red', linestyle='--',
                   label=f'Null mean: {np.mean(null):.3f}')
        ax.axvline(np.mean(observed), color='blue', linestyle='--',
                   label=f'Observed mean: {np.mean(observed):.3f}')

        ax.set_xlabel('Test Correlation')
        ax.set_ylabel('Frequency')
        ax.set_title(title)
        ax.legend()

        return fig
