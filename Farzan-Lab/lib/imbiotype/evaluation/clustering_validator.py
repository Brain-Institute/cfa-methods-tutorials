"""
ClusteringValidator - Validates clustering within nested CV loop.
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans

from .hungarian import LabelMatcher
from .subject_tracker import SubjectTracker
from .consensus import ConsensusMatrixBuilder
from .config import EvaluationConfig


@dataclass
class ClusteringFoldResult:
    """Results from a single CV fold."""
    train_labels: np.ndarray
    test_labels: np.ndarray
    train_silhouette: float
    test_silhouette: float
    train_idx: np.ndarray
    test_idx: np.ndarray


class ClusteringValidator:
    """
    Handles clustering validation within nested CV loop.

    Responsible for:
    - Fitting clustering on train data
    - Assigning test data to clusters
    - Hungarian matching across repeats
    - Tracking subject consistency
    - Building consensus matrix
    """

    def __init__(
        self,
        n_subjects: int,
        config: EvaluationConfig,
        reference_strategy: str = 'repeat_1'
    ):
        """
        Initialize clustering validator.

        Args:
            n_subjects: Total number of subjects
            config: Evaluation configuration
            reference_strategy: Label matching strategy ('repeat_1' or 'full_dataset')
        """
        self.n_subjects = n_subjects
        self.config = config
        self.label_matcher = LabelMatcher(reference_strategy=reference_strategy)
        self.subject_tracker = SubjectTracker(n_subjects)
        self.consensus_builder = ConsensusMatrixBuilder(n_subjects)

    def validate_fold(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        n_clusters: int = 3,
        clustering_method: str = 'kmeans',
        random_state: int = 42
    ) -> ClusteringFoldResult:
        """
        Run clustering on train, assign test for one fold.

        Args:
            X_train: Training features (n_train x n_features)
            X_test: Test features (n_test x n_features)
            train_idx: Indices of training subjects
            test_idx: Indices of test subjects
            n_clusters: Number of clusters
            clustering_method: 'kmeans', 'hierarchical', 'dbscan'
            random_state: Random seed

        Returns:
            ClusteringFoldResult with labels and metrics
        """
        # Fit clustering on train
        train_labels, model = self._fit_clustering(
            X_train, n_clusters, clustering_method, random_state
        )

        # Assign test to nearest cluster
        test_labels = self._predict_test(X_test, model, clustering_method)

        # Compute metrics
        train_silhouette = self._compute_silhouette(X_train, train_labels)
        test_silhouette = self._compute_silhouette(X_test, test_labels)

        return ClusteringFoldResult(
            train_labels=train_labels,
            test_labels=test_labels,
            train_silhouette=train_silhouette,
            test_silhouette=test_silhouette,
            train_idx=train_idx,
            test_idx=test_idx
        )

    def _fit_clustering(
        self,
        X: np.ndarray,
        n_clusters: int,
        method: str,
        random_state: int
    ) -> Tuple[np.ndarray, Any]:
        """Fit clustering and return labels + model."""
        if method == 'kmeans':
            model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
            labels = model.fit_predict(X)
        elif method == 'hierarchical':
            from sklearn.cluster import AgglomerativeClustering
            model = AgglomerativeClustering(n_clusters=n_clusters)
            labels = model.fit_predict(X)
            # Store centroids for test assignment
            model.centroids_ = self._compute_centroids(X, labels)
        elif method == 'dbscan':
            from sklearn.cluster import DBSCAN
            model = DBSCAN()
            labels = model.fit_predict(X)
            # Store core samples for test assignment
            model.core_sample_features_ = X[model.core_sample_indices_]
            model.core_sample_labels_ = labels[model.core_sample_indices_]
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        return labels, model

    def _predict_test(
        self,
        X_test: np.ndarray,
        model: Any,
        method: str
    ) -> np.ndarray:
        """Assign test samples to clusters."""
        if method == 'kmeans':
            return model.predict(X_test)
        elif method == 'hierarchical':
            # Assign to nearest centroid
            return self._assign_to_centroids(X_test, model.centroids_)
        elif method == 'dbscan':
            # Assign to nearest core sample
            return self._assign_to_nearest_core(
                X_test,
                model.core_sample_features_,
                model.core_sample_labels_
            )
        else:
            raise ValueError(f"Unknown clustering method: {method}")

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """Compute cluster centroids."""
        unique_labels = np.unique(labels[labels >= 0])  # Exclude noise (-1)
        centroids = np.zeros((len(unique_labels), X.shape[1]))

        for i, label in enumerate(unique_labels):
            centroids[i] = X[labels == label].mean(axis=0)

        return centroids

    def _assign_to_centroids(
        self,
        X: np.ndarray,
        centroids: np.ndarray
    ) -> np.ndarray:
        """Assign samples to nearest centroid."""
        from sklearn.metrics import pairwise_distances

        distances = pairwise_distances(X, centroids)
        return np.argmin(distances, axis=1)

    def _assign_to_nearest_core(
        self,
        X: np.ndarray,
        core_features: np.ndarray,
        core_labels: np.ndarray
    ) -> np.ndarray:
        """Assign samples to nearest core sample's cluster."""
        from sklearn.metrics import pairwise_distances

        if len(core_features) == 0:
            # No core samples - assign all as noise
            return np.full(len(X), -1)

        distances = pairwise_distances(X, core_features)
        nearest_core_idx = np.argmin(distances, axis=1)
        return core_labels[nearest_core_idx]

    def _compute_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """Compute silhouette score, handling edge cases including noise labels."""
        # Filter out noise points (label = -1) from DBSCAN
        # sklearn's silhouette_score doesn't handle negative labels
        valid_mask = labels >= 0
        X_valid = X[valid_mask]
        labels_valid = labels[valid_mask]

        unique_labels = np.unique(labels_valid)
        n_clusters = len(unique_labels)

        if n_clusters < 2 or len(X_valid) < 2:
            return 0.0

        try:
            return silhouette_score(X_valid, labels_valid)
        except Exception:
            return 0.0

    def process_repeat(
        self,
        fold_results: List[ClusteringFoldResult],
        repeat_idx: int
    ):
        """
        Process all folds from one repeat.

        - Hungarian match full labeling to reference
        - Update subject tracker
        - Update consensus matrix

        Args:
            fold_results: Results from all folds in this repeat
            repeat_idx: Index of the repeat
        """
        # Reconstruct full labeling from test sets
        full_labels = np.full(self.n_subjects, -1, dtype=int)

        for fold_result in fold_results:
            for idx, label in zip(fold_result.test_idx, fold_result.test_labels):
                full_labels[idx] = label

        # Check that all subjects were labeled
        if np.any(full_labels == -1):
            # Some subjects missing - shouldn't happen with proper K-fold
            pass

        # Hungarian match to reference
        if repeat_idx == 0 and not self.label_matcher.is_reference_set:
            # First repeat sets the reference
            self.label_matcher.set_reference(full_labels)
            matched_labels = full_labels
        else:
            matched_labels = self.label_matcher.match(full_labels)

        # Update tracking
        self.subject_tracker.record(matched_labels, repeat_idx)
        self.consensus_builder.add_repeat_vectorized(matched_labels)

    def set_reference_labels(self, labels: np.ndarray):
        """
        Set reference labels from full-dataset clustering.

        Used when reference_strategy='full_dataset'.

        Args:
            labels: Reference labels from initial clustering
        """
        self.label_matcher.set_reference(labels)

    def aggregate_results(
        self,
        all_fold_results: List[List[ClusteringFoldResult]],
        n_repeats: int,
        n_folds: int
    ) -> Dict[str, Any]:
        """
        Aggregate results across all folds and repeats.

        Args:
            all_fold_results: Nested list [repeat][fold] of ClusteringFoldResult
            n_repeats: Number of CV repeats
            n_folds: Number of folds per repeat

        Returns:
            Aggregated results dict
        """
        # Collect silhouette scores
        all_train_silhouettes = np.zeros((n_repeats, n_folds))
        all_test_silhouettes = np.zeros((n_repeats, n_folds))

        for r, repeat_results in enumerate(all_fold_results):
            for f, fold_result in enumerate(repeat_results):
                all_train_silhouettes[r, f] = fold_result.train_silhouette
                all_test_silhouettes[r, f] = fold_result.test_silhouette

        # Get subject consistency
        consistency = self.subject_tracker.compute_consistency()
        final_labels = self.subject_tracker.get_final_labels()
        consensus_matrix = self.consensus_builder.get_normalized()

        return {
            'n_repeats': n_repeats,
            'n_folds': n_folds,
            'train_silhouettes': all_train_silhouettes,
            'test_silhouettes': all_test_silhouettes,
            'mean_train_silhouette': float(np.mean(all_train_silhouettes)),
            'mean_test_silhouette': float(np.mean(all_test_silhouettes)),
            'std_test_silhouette': float(np.std(all_test_silhouettes)),
            'subject_consistency': consistency,
            'mean_subject_consistency': float(np.mean(consistency)),
            'unstable_subject_count': int(np.sum(consistency < 0.7)),
            'final_labels': final_labels,
            'consensus_matrix': consensus_matrix,
            'consensus_quality': self.consensus_builder.get_cluster_quality()
        }

    def reset(self):
        """Reset all tracking state."""
        self.label_matcher.reset()
        self.subject_tracker.reset()
        self.consensus_builder.reset()
