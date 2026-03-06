"""
SubjectTracker - Track cluster assignments across CV repeats.

Computes subject consistency (% of repeats where subject lands in modal cluster)
and final consensus labels.
"""

from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import numpy as np
from scipy import stats


class SubjectTracker:
    """
    Track cluster assignments for each subject across CV repeats.

    Example:
        tracker = SubjectTracker(n_subjects=100)

        for repeat in range(100):
            labels = run_clustering(...)  # len = 100
            tracker.record(labels, repeat_idx=repeat)

        consistency = tracker.compute_consistency()
        final_labels = tracker.get_final_labels()
    """

    def __init__(self, n_subjects: int):
        """
        Initialize subject tracker.

        Args:
            n_subjects: Total number of subjects in the dataset
        """
        self.n_subjects = n_subjects
        # assignments[subject_id] = list of (repeat_idx, label) tuples
        self.assignments: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        self.n_repeats_recorded = 0

    def record(
        self,
        labels: np.ndarray,
        repeat_idx: int,
        subject_indices: Optional[np.ndarray] = None
    ):
        """
        Record labels for a repeat.

        Args:
            labels: Cluster labels (n_subjects,) or (n_subset,) if subject_indices given
            repeat_idx: Index of the CV repeat
            subject_indices: If provided, specifies which subjects these labels are for.
                           Used when evaluating on test set (subset of subjects).
        """
        if subject_indices is None:
            # Full dataset labeling
            if len(labels) != self.n_subjects:
                raise ValueError(
                    f"Expected {self.n_subjects} labels, got {len(labels)}"
                )
            for subj_id, label in enumerate(labels):
                self.assignments[subj_id].append((repeat_idx, int(label)))
        else:
            # Subset labeling (test set)
            if len(labels) != len(subject_indices):
                raise ValueError(
                    f"Labels and subject_indices must have same length"
                )
            for subj_id, label in zip(subject_indices, labels):
                self.assignments[int(subj_id)].append((repeat_idx, int(label)))

        self.n_repeats_recorded = max(self.n_repeats_recorded, repeat_idx + 1)

    def compute_consistency(self) -> np.ndarray:
        """
        Compute per-subject consistency scores.

        Consistency = proportion of repeats where subject is in modal cluster.

        Returns:
            Array of consistency scores (n_subjects,), values in [0, 1]
        """
        consistency = np.zeros(self.n_subjects)

        for subj_id in range(self.n_subjects):
            if subj_id not in self.assignments or len(self.assignments[subj_id]) == 0:
                # Subject was never assigned (shouldn't happen in normal use)
                consistency[subj_id] = 0.0
                continue

            # Get all labels for this subject
            labels = [label for _, label in self.assignments[subj_id]]

            # Find modal label
            mode_result = stats.mode(labels, keepdims=False)
            modal_count = mode_result.count

            # Consistency = modal_count / total_assignments
            consistency[subj_id] = modal_count / len(labels)

        return consistency

    def get_final_labels(self) -> np.ndarray:
        """
        Get final (modal) label for each subject.

        Returns:
            Array of final labels (n_subjects,)
        """
        final_labels = np.zeros(self.n_subjects, dtype=int)

        for subj_id in range(self.n_subjects):
            if subj_id not in self.assignments or len(self.assignments[subj_id]) == 0:
                final_labels[subj_id] = -1  # Unknown
                continue

            labels = [label for _, label in self.assignments[subj_id]]
            mode_result = stats.mode(labels, keepdims=False)
            final_labels[subj_id] = mode_result.mode

        return final_labels

    def get_unstable_subjects(self, threshold: float = 0.7) -> np.ndarray:
        """
        Get indices of subjects with consistency below threshold.

        Args:
            threshold: Minimum consistency to be considered stable

        Returns:
            Array of subject indices
        """
        consistency = self.compute_consistency()
        return np.where(consistency < threshold)[0]

    def get_assignment_counts(self, subject_id: int) -> Dict[int, int]:
        """
        Get count of each label assignment for a subject.

        Args:
            subject_id: Subject index

        Returns:
            Dict mapping label -> count
        """
        if subject_id not in self.assignments:
            return {}

        counts = defaultdict(int)
        for _, label in self.assignments[subject_id]:
            counts[label] += 1

        return dict(counts)

    def get_assignment_probabilities(self) -> np.ndarray:
        """
        Get probability distribution over clusters for each subject.

        Returns:
            Array (n_subjects, n_clusters) of assignment probabilities
        """
        # First, determine number of clusters (max label + 1)
        all_labels = []
        for subj_assignments in self.assignments.values():
            all_labels.extend([label for _, label in subj_assignments])

        if not all_labels:
            return np.zeros((self.n_subjects, 1))

        n_clusters = max(all_labels) + 1
        probs = np.zeros((self.n_subjects, n_clusters))

        for subj_id in range(self.n_subjects):
            if subj_id not in self.assignments:
                continue

            labels = [label for _, label in self.assignments[subj_id]]
            total = len(labels)

            if total == 0:
                continue

            for label in labels:
                probs[subj_id, label] += 1

            probs[subj_id] /= total

        return probs

    def summary(self) -> Dict:
        """
        Get summary statistics.

        Returns:
            Dict with summary metrics
        """
        consistency = self.compute_consistency()

        return {
            'n_subjects': self.n_subjects,
            'n_repeats_recorded': self.n_repeats_recorded,
            'mean_consistency': float(np.mean(consistency)),
            'std_consistency': float(np.std(consistency)),
            'min_consistency': float(np.min(consistency)),
            'max_consistency': float(np.max(consistency)),
            'n_unstable_70': int(np.sum(consistency < 0.7)),
            'n_unstable_50': int(np.sum(consistency < 0.5)),
        }

    def reset(self):
        """Clear all recorded assignments."""
        self.assignments.clear()
        self.n_repeats_recorded = 0
