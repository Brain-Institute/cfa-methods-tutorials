"""
ConsensusMatrixBuilder - Build consensus matrix from CV repeats.

The consensus matrix tracks how often pairs of subjects are clustered together
across all CV repeats. Values range from 0 (never together) to 1 (always together).
"""

from typing import Optional
import numpy as np


class ConsensusMatrixBuilder:
    """
    Build consensus matrix from repeated clusterings.

    The consensus matrix C[i,j] = proportion of repeats where subject i and j
    are assigned to the same cluster.

    Example:
        builder = ConsensusMatrixBuilder(n_subjects=100)

        for repeat in range(100):
            labels = run_clustering(...)
            builder.add_repeat(labels)

        consensus = builder.get_normalized()
        # consensus[i,j] = frequency that subjects i,j were in same cluster
    """

    def __init__(self, n_subjects: int):
        """
        Initialize consensus matrix builder.

        Args:
            n_subjects: Total number of subjects
        """
        self.n_subjects = n_subjects
        self.matrix = np.zeros((n_subjects, n_subjects), dtype=np.float64)
        self.n_repeats = 0
        # Track how often each subject pair was included (for handling partial labelings)
        self.pair_counts = np.zeros((n_subjects, n_subjects), dtype=np.int64)

    def add_repeat(
        self,
        labels: np.ndarray,
        subject_indices: Optional[np.ndarray] = None
    ):
        """
        Add one repeat's labeling to consensus.

        Args:
            labels: Cluster labels (n_subjects,) or (n_subset,)
            subject_indices: If provided, specifies which subjects these labels are for.
                           Used when only a subset was labeled (e.g., test set).
        """
        self.n_repeats += 1

        if subject_indices is None:
            # Full dataset labeling
            if len(labels) != self.n_subjects:
                raise ValueError(
                    f"Expected {self.n_subjects} labels, got {len(labels)}"
                )

            # Update co-clustering counts
            for i in range(self.n_subjects):
                # Diagonal: subject always clusters with itself
                self.matrix[i, i] += 1
                self.pair_counts[i, i] += 1

                # Off-diagonal pairs (j > i)
                for j in range(i + 1, self.n_subjects):
                    if labels[i] == labels[j]:
                        self.matrix[i, j] += 1
                        self.matrix[j, i] += 1
                    self.pair_counts[i, j] += 1
                    self.pair_counts[j, i] += 1
        else:
            # Subset labeling (e.g., test set)
            subject_indices = np.asarray(subject_indices)
            n_subset = len(subject_indices)

            for idx_i in range(n_subset):
                i = subject_indices[idx_i]
                # Diagonal: subject always clusters with itself
                self.matrix[i, i] += 1
                self.pair_counts[i, i] += 1

                # Off-diagonal pairs (idx_j > idx_i)
                for idx_j in range(idx_i + 1, n_subset):
                    j = subject_indices[idx_j]
                    if labels[idx_i] == labels[idx_j]:
                        self.matrix[i, j] += 1
                        self.matrix[j, i] += 1
                    self.pair_counts[i, j] += 1
                    self.pair_counts[j, i] += 1

    def add_repeat_vectorized(self, labels: np.ndarray):
        """
        Add repeat using vectorized operations (faster for full labelings).

        Args:
            labels: Cluster labels (n_subjects,)
        """
        if len(labels) != self.n_subjects:
            raise ValueError(
                f"Expected {self.n_subjects} labels, got {len(labels)}"
            )

        self.n_repeats += 1

        # Create co-clustering matrix for this repeat
        # co_cluster[i,j] = 1 if labels[i] == labels[j]
        labels_row = labels.reshape(-1, 1)
        labels_col = labels.reshape(1, -1)
        co_cluster = (labels_row == labels_col).astype(np.float64)

        self.matrix += co_cluster
        self.pair_counts += 1

    def get_normalized(self) -> np.ndarray:
        """
        Return consensus matrix normalized by number of repeats.

        Returns:
            Normalized consensus matrix (n_subjects, n_subjects)
            Values in [0, 1] representing co-clustering frequency
        """
        if self.n_repeats == 0:
            return np.zeros((self.n_subjects, self.n_subjects))

        # Normalize by pair counts to handle partial labelings
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized = self.matrix / self.pair_counts
            normalized = np.nan_to_num(normalized, nan=0.0)

        return normalized

    def get_raw(self) -> np.ndarray:
        """
        Return raw (unnormalized) consensus matrix.

        Returns:
            Raw count matrix (n_subjects, n_subjects)
        """
        return self.matrix.copy()

    def get_cluster_quality(self) -> float:
        """
        Compute overall cluster quality from consensus matrix.

        Quality = mean of consensus values on the block diagonal
        (subjects in the same final cluster should have high consensus).

        Returns:
            Quality score in [0, 1]
        """
        normalized = self.get_normalized()

        # Ideal consensus matrix has 1s on block diagonal, 0s elsewhere
        # Simple metric: mean of non-diagonal values should be close to 0 or 1
        # (indicating clear separation)

        # Remove diagonal for this computation
        off_diag = normalized[~np.eye(self.n_subjects, dtype=bool)]

        if len(off_diag) == 0:
            return 1.0

        # Bimodality: values should be close to 0 or 1
        # Measure as variance from nearest extreme
        dist_to_extreme = np.minimum(off_diag, 1 - off_diag)
        quality = 1 - 2 * np.mean(dist_to_extreme)  # 1 = perfect bimodality

        return float(quality)

    def reset(self):
        """Clear the consensus matrix."""
        self.matrix.fill(0)
        self.pair_counts.fill(0)
        self.n_repeats = 0

    def summary(self) -> dict:
        """
        Get summary statistics.

        Returns:
            Dict with consensus matrix statistics
        """
        normalized = self.get_normalized()
        off_diag = normalized[~np.eye(self.n_subjects, dtype=bool)]

        return {
            'n_subjects': self.n_subjects,
            'n_repeats': self.n_repeats,
            'mean_consensus': float(np.mean(off_diag)) if len(off_diag) > 0 else 0.0,
            'std_consensus': float(np.std(off_diag)) if len(off_diag) > 0 else 0.0,
            'min_consensus': float(np.min(off_diag)) if len(off_diag) > 0 else 0.0,
            'max_consensus': float(np.max(off_diag)) if len(off_diag) > 0 else 0.0,
            'cluster_quality': self.get_cluster_quality()
        }
