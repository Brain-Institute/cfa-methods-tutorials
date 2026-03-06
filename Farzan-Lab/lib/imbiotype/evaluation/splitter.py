"""
TrainTestSplitter - Generate train/test splits for nested CV.
"""

from typing import Optional, Iterator, Tuple
import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold

from .config import EvaluationConfig


class TrainTestSplitter:
    """
    Generate train/test splits for nested cross-validation.

    Supports:
    - Multiple repeats with different random seeds
    - K-fold cross-validation within each repeat
    - Optional stratification by a categorical variable
    """

    def __init__(self, config: EvaluationConfig):
        """
        Initialize splitter with evaluation config.

        Args:
            config: EvaluationConfig with n_repeats, n_outer_folds, random_state
        """
        self.config = config

    def generate_splits(
        self,
        n_samples: int,
        stratify_labels: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[int, int, np.ndarray, np.ndarray]]:
        """
        Yield all (repeat_idx, fold_idx, train_idx, test_idx) tuples.

        Args:
            n_samples: Number of samples in dataset
            stratify_labels: Optional array for stratified splitting

        Yields:
            Tuple of (repeat_idx, fold_idx, train_indices, test_indices)
        """
        for repeat_idx in range(self.config.n_repeats):
            for fold_idx, train_idx, test_idx in self.generate_folds(
                n_samples, repeat_idx, stratify_labels
            ):
                yield repeat_idx, fold_idx, train_idx, test_idx

    def generate_folds(
        self,
        n_samples: int,
        repeat_idx: int,
        stratify_labels: Optional[np.ndarray] = None
    ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:
        """
        Generate folds for a single repeat.

        Args:
            n_samples: Number of samples
            repeat_idx: Index of current repeat (used for random seed)
            stratify_labels: Optional labels for stratification

        Yields:
            Tuple of (fold_idx, train_indices, test_indices)
        """
        # Vary random state by repeat to get different splits
        random_state = self.config.random_state + repeat_idx

        if stratify_labels is not None:
            kfold = StratifiedKFold(
                n_splits=self.config.n_outer_folds,
                shuffle=True,
                random_state=random_state
            )
            splits = kfold.split(np.arange(n_samples), stratify_labels)
        else:
            kfold = KFold(
                n_splits=self.config.n_outer_folds,
                shuffle=True,
                random_state=random_state
            )
            splits = kfold.split(np.arange(n_samples))

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            yield fold_idx, train_idx, test_idx

    def get_total_iterations(self) -> int:
        """Return total number of train/test iterations."""
        return self.config.n_repeats * self.config.n_outer_folds

    def get_fold_sizes(self, n_samples: int) -> Tuple[int, int]:
        """
        Get approximate train/test sizes per fold.

        Args:
            n_samples: Total number of samples

        Returns:
            Tuple of (train_size, test_size)
        """
        test_size = n_samples // self.config.n_outer_folds
        train_size = n_samples - test_size
        return train_size, test_size
