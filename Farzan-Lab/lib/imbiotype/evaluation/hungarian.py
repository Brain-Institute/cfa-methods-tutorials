"""
Hungarian algorithm for label correspondence across CV repeats.

When clustering is performed on different data splits, the cluster labels
are arbitrary (e.g., cluster "1" in split A might correspond to cluster "2"
in split B). The Hungarian algorithm finds the optimal mapping between
label sets to maximize agreement.
"""

from typing import Optional, Dict
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def match_labels(
    labels_reference: np.ndarray,
    labels_to_match: np.ndarray
) -> np.ndarray:
    """
    Match labels using Hungarian algorithm.

    Finds the optimal one-to-one mapping from labels_to_match to labels_reference
    that maximizes overlap.

    Args:
        labels_reference: Reference labels (the "ground truth" for matching)
        labels_to_match: Labels to be remapped

    Returns:
        Remapped labels (same shape as labels_to_match)

    Example:
        >>> ref = np.array([0, 0, 1, 1, 2, 2])
        >>> new = np.array([2, 2, 0, 0, 1, 1])  # Same clustering, different labels
        >>> matched = match_labels(ref, new)
        >>> np.array_equal(matched, ref)  # True
    """
    # Handle edge cases
    if len(labels_reference) != len(labels_to_match):
        raise ValueError(
            f"Label arrays must have same length: {len(labels_reference)} vs {len(labels_to_match)}"
        )

    if len(labels_reference) == 0:
        return np.array([])

    # Get unique labels
    unique_ref = np.unique(labels_reference)
    unique_match = np.unique(labels_to_match)

    # Build confusion matrix
    # Rows = reference labels, Cols = labels_to_match
    conf = confusion_matrix(labels_reference, labels_to_match)

    # Handle case where label sets have different sizes
    n_ref = len(unique_ref)
    n_match = len(unique_match)

    if n_ref != n_match:
        # Pad confusion matrix to be square
        max_n = max(n_ref, n_match)
        conf_padded = np.zeros((max_n, max_n), dtype=conf.dtype)
        conf_padded[:n_ref, :n_match] = conf
        conf = conf_padded

    # Hungarian algorithm (maximize overlap = minimize negative overlap)
    row_ind, col_ind = linear_sum_assignment(-conf)

    # Create mapping from labels_to_match values to reference values
    # col_ind[i] is the column (label_to_match value) that maps to row i (reference value)
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if j < n_match:  # Only map actual labels, not padding
            # j is index in unique_match, maps to i which is index in unique_ref
            if j < len(unique_match):
                match_label = unique_match[j]
                if i < len(unique_ref):
                    ref_label = unique_ref[i]
                else:
                    # This label doesn't exist in reference, keep as is
                    ref_label = match_label
                mapping[match_label] = ref_label

    # Apply mapping
    matched = np.array([mapping.get(l, l) for l in labels_to_match])

    return matched


def compute_mapping(
    labels_reference: np.ndarray,
    labels_to_match: np.ndarray
) -> Dict[int, int]:
    """
    Compute the label mapping without applying it.

    Args:
        labels_reference: Reference labels
        labels_to_match: Labels to be mapped

    Returns:
        Dict mapping from labels_to_match values to reference values
    """
    unique_ref = np.unique(labels_reference)
    unique_match = np.unique(labels_to_match)

    conf = confusion_matrix(labels_reference, labels_to_match)

    n_ref = len(unique_ref)
    n_match = len(unique_match)

    if n_ref != n_match:
        max_n = max(n_ref, n_match)
        conf_padded = np.zeros((max_n, max_n), dtype=conf.dtype)
        conf_padded[:n_ref, :n_match] = conf
        conf = conf_padded

    row_ind, col_ind = linear_sum_assignment(-conf)

    mapping = {}
    for i, j in zip(row_ind, col_ind):
        if j < len(unique_match) and i < len(unique_ref):
            mapping[int(unique_match[j])] = int(unique_ref[i])

    return mapping


class LabelMatcher:
    """
    Manages label correspondence across CV repeats.

    Supports two reference strategies:
    - 'repeat_1': Use first CV repeat as reference
    - 'full_dataset': Use labels from user's initial full-data clustering

    Example:
        matcher = LabelMatcher(reference_strategy='repeat_1')

        # First repeat sets the reference
        labels_r1 = run_clustering(data_repeat_1)
        matcher.set_reference(labels_r1)

        # Subsequent repeats are matched to reference
        labels_r2 = run_clustering(data_repeat_2)
        matched_r2 = matcher.match(labels_r2)
    """

    def __init__(self, reference_strategy: str = 'repeat_1'):
        """
        Initialize label matcher.

        Args:
            reference_strategy:
              - 'repeat_1': Use first CV repeat as reference
              - 'full_dataset': Use labels from user's initial full-data clustering
        """
        if reference_strategy not in ('repeat_1', 'full_dataset'):
            raise ValueError(
                f"reference_strategy must be 'repeat_1' or 'full_dataset', "
                f"got '{reference_strategy}'"
            )
        self.reference_strategy = reference_strategy
        self.reference_labels: Optional[np.ndarray] = None
        self._is_reference_set = False

    def set_reference(self, labels: np.ndarray):
        """
        Set reference labels for matching.

        Args:
            labels: Reference label array
        """
        self.reference_labels = np.asarray(labels).copy()
        self._is_reference_set = True

    def match(self, labels_to_match: np.ndarray) -> np.ndarray:
        """
        Match labels to reference using Hungarian algorithm.

        Args:
            labels_to_match: Labels to be remapped

        Returns:
            Remapped labels aligned with reference
        """
        if not self._is_reference_set:
            raise ValueError(
                "Reference labels not set. Call set_reference() first."
            )

        return match_labels(self.reference_labels, labels_to_match)

    def get_mapping(self, labels_to_match: np.ndarray) -> Dict[int, int]:
        """
        Get the mapping without applying it.

        Args:
            labels_to_match: Labels to compute mapping for

        Returns:
            Dict mapping labels_to_match values to reference values
        """
        if not self._is_reference_set:
            raise ValueError(
                "Reference labels not set. Call set_reference() first."
            )

        return compute_mapping(self.reference_labels, labels_to_match)

    @property
    def is_reference_set(self) -> bool:
        """Check if reference has been set."""
        return self._is_reference_set

    def reset(self):
        """Reset the matcher (clear reference)."""
        self.reference_labels = None
        self._is_reference_set = False
