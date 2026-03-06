"""
Validation Metrics for Validation Study Generator

Computes ground truth comparison metrics for:
- Clustering recovery (ARI, NMI, V-measure)
- CCA recovery (canonical correlation error, subspace angles, loading correlations)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_score,
    completeness_score,
    v_measure_score,
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score
)


# =============================================================================
# Clustering Recovery Metrics
# =============================================================================

def clustering_recovery_metrics(true_labels: np.ndarray,
                                 predicted_labels: np.ndarray) -> Dict[str, float]:
    """
    Compute clustering recovery metrics comparing predicted to true labels.

    Args:
        true_labels: Ground truth cluster assignments
        predicted_labels: Predicted cluster assignments

    Returns:
        Dictionary of metrics:
            - ari: Adjusted Rand Index (-1 to 1, 1 = perfect)
            - nmi: Normalized Mutual Information (0 to 1, 1 = perfect)
            - homogeneity: Homogeneity score (0 to 1)
            - completeness: Completeness score (0 to 1)
            - v_measure: Harmonic mean of homogeneity and completeness
    """
    return {
        "ari": adjusted_rand_score(true_labels, predicted_labels),
        "nmi": normalized_mutual_info_score(true_labels, predicted_labels),
        "homogeneity": homogeneity_score(true_labels, predicted_labels),
        "completeness": completeness_score(true_labels, predicted_labels),
        "v_measure": v_measure_score(true_labels, predicted_labels)
    }


def cluster_quality_metrics(data: np.ndarray,
                            labels: np.ndarray) -> Dict[str, float]:
    """
    Compute internal cluster quality metrics.

    Args:
        data: Data matrix (n_samples, n_features)
        labels: Cluster assignments

    Returns:
        Dictionary of metrics:
            - silhouette: Silhouette score (-1 to 1, higher = better)
            - calinski_harabasz: Calinski-Harabasz index (higher = better)
            - davies_bouldin: Davies-Bouldin index (lower = better)
    """
    n_unique = len(np.unique(labels))
    if n_unique < 2:
        return {
            "silhouette": np.nan,
            "calinski_harabasz": np.nan,
            "davies_bouldin": np.nan
        }

    return {
        "silhouette": silhouette_score(data, labels),
        "calinski_harabasz": calinski_harabasz_score(data, labels),
        "davies_bouldin": davies_bouldin_score(data, labels)
    }


# =============================================================================
# CCA Recovery Metrics
# =============================================================================

def canonical_correlation_error(true_correlations: np.ndarray,
                                 estimated_correlations: np.ndarray) -> Dict[str, float]:
    """
    Compute error between true and estimated canonical correlations.

    Args:
        true_correlations: Ground truth canonical correlations
        estimated_correlations: Estimated canonical correlations

    Returns:
        Dictionary of metrics:
            - mae: Mean absolute error
            - rmse: Root mean squared error
            - max_error: Maximum absolute error
            - correlation: Pearson correlation between true and estimated
    """
    n_true = len(true_correlations)
    n_est = len(estimated_correlations)
    n_compare = min(n_true, n_est)

    true_cc = true_correlations[:n_compare]
    est_cc = estimated_correlations[:n_compare]

    errors = np.abs(true_cc - est_cc)

    metrics = {
        "mae": np.mean(errors),
        "rmse": np.sqrt(np.mean(errors ** 2)),
        "max_error": np.max(errors),
        "n_components_compared": n_compare
    }

    if n_compare >= 2:
        metrics["correlation"] = pearsonr(true_cc, est_cc)[0]
    else:
        metrics["correlation"] = np.nan

    return metrics


def subspace_angle(U: np.ndarray, V: np.ndarray) -> float:
    """
    Compute principal angle between two subspaces.

    The subspace angle measures how similar two sets of canonical directions are.
    An angle of 0 means identical subspaces, π/2 means orthogonal.

    Args:
        U: First subspace basis (n_features, n_components)
        V: Second subspace basis (n_features, n_components)

    Returns:
        Principal angle in radians (0 to π/2)
    """
    # Orthonormalize both bases
    Q_U, _ = np.linalg.qr(U)
    Q_V, _ = np.linalg.qr(V)

    # Compute SVD of Q_U.T @ Q_V
    _, s, _ = np.linalg.svd(Q_U.T @ Q_V)

    # Principal angle is arccos of largest singular value
    # Clamp to valid range for arccos
    cos_angle = np.clip(s[0], -1, 1)
    return np.arccos(cos_angle)


def loading_correlation(true_loadings: np.ndarray,
                        estimated_loadings: np.ndarray,
                        method: str = "pearson") -> Dict[str, float]:
    """
    Compute correlation between true and estimated loading matrices.

    Args:
        true_loadings: Ground truth loading matrix (n_features, n_components)
        estimated_loadings: Estimated loading matrix
        method: "pearson" or "spearman"

    Returns:
        Dictionary of metrics:
            - mean_correlation: Mean correlation across components
            - component_correlations: List of per-component correlations
            - overall_correlation: Correlation of flattened matrices
    """
    n_true_comp = true_loadings.shape[1]
    n_est_comp = estimated_loadings.shape[1]
    n_compare = min(n_true_comp, n_est_comp)

    corr_func = pearsonr if method == "pearson" else spearmanr

    component_corrs = []
    for i in range(n_compare):
        # Note: CCA loadings may have sign ambiguity
        # Use absolute correlation to handle sign flips
        r, _ = corr_func(true_loadings[:, i], estimated_loadings[:, i])
        component_corrs.append(abs(r))

    # Overall correlation (flattened)
    true_flat = true_loadings[:, :n_compare].flatten()
    est_flat = estimated_loadings[:, :n_compare].flatten()
    overall_r, _ = corr_func(true_flat, est_flat)

    return {
        "mean_correlation": np.mean(component_corrs),
        "component_correlations": component_corrs,
        "overall_correlation": abs(overall_r),
        "n_components_compared": n_compare
    }


def cca_recovery_metrics(ground_truth: Dict,
                          estimated_results: Dict) -> Dict[str, any]:
    """
    Compute comprehensive CCA recovery metrics.

    Args:
        ground_truth: Dictionary from ValidationStudyGenerator ground truth
        estimated_results: Dictionary with estimated CCA results containing:
            - canonical_correlations: Estimated correlations
            - loadings_x: Estimated X loadings (optional)
            - loadings_y: Estimated Y loadings (optional)

    Returns:
        Dictionary of all CCA recovery metrics
    """
    metrics = {}

    # Canonical correlation error
    if "true_canonical_correlations" in ground_truth and \
       "canonical_correlations" in estimated_results:
        metrics["correlation_error"] = canonical_correlation_error(
            ground_truth["true_canonical_correlations"],
            estimated_results["canonical_correlations"]
        )

    # Subspace angles for loadings
    if "true_loadings_x" in ground_truth and "loadings_x" in estimated_results:
        metrics["subspace_angle_x"] = subspace_angle(
            ground_truth["true_loadings_x"],
            estimated_results["loadings_x"]
        )
        metrics["loading_correlation_x"] = loading_correlation(
            ground_truth["true_loadings_x"],
            estimated_results["loadings_x"]
        )

    if "true_loadings_y" in ground_truth and "loadings_y" in estimated_results:
        metrics["subspace_angle_y"] = subspace_angle(
            ground_truth["true_loadings_y"],
            estimated_results["loadings_y"]
        )
        metrics["loading_correlation_y"] = loading_correlation(
            ground_truth["true_loadings_y"],
            estimated_results["loadings_y"]
        )

    return metrics


# =============================================================================
# Combined Validation Metrics
# =============================================================================

def compute_all_validation_metrics(ground_truth: Dict,
                                    predicted_labels: np.ndarray,
                                    estimated_cca_results: Optional[Dict] = None,
                                    clustering_data: Optional[np.ndarray] = None) -> Dict:
    """
    Compute all validation metrics for a complete validation run.

    Args:
        ground_truth: Full ground truth dictionary from ValidationStudyGenerator
        predicted_labels: Predicted cluster labels from clustering
        estimated_cca_results: Estimated CCA results (optional)
        clustering_data: Data used for clustering (for internal metrics)

    Returns:
        Dictionary with all validation metrics organized by category
    """
    results = {}

    # Get true cluster labels from ground truth
    if "cluster_labels" in ground_truth:
        true_labels = ground_truth["cluster_labels"]
        results["clustering_recovery"] = clustering_recovery_metrics(
            true_labels, predicted_labels
        )

    # Cluster quality metrics
    if clustering_data is not None:
        results["cluster_quality_predicted"] = cluster_quality_metrics(
            clustering_data, predicted_labels
        )
        if "cluster_labels" in ground_truth:
            results["cluster_quality_true"] = cluster_quality_metrics(
                clustering_data, ground_truth["cluster_labels"]
            )

    # CCA recovery metrics
    if estimated_cca_results is not None:
        results["cca_recovery"] = cca_recovery_metrics(
            ground_truth, estimated_cca_results
        )

    return results


def summarize_validation_results(results: Dict) -> Dict[str, float]:
    """
    Summarize validation results into key metrics for reporting.

    Args:
        results: Full validation results dictionary

    Returns:
        Flat dictionary with key summary metrics
    """
    summary = {}

    # Clustering summary
    if "clustering_recovery" in results:
        summary["ari"] = results["clustering_recovery"]["ari"]
        summary["nmi"] = results["clustering_recovery"]["nmi"]

    # CCA summary
    if "cca_recovery" in results:
        cca = results["cca_recovery"]
        if "correlation_error" in cca:
            summary["cca_mae"] = cca["correlation_error"]["mae"]
        if "subspace_angle_x" in cca:
            summary["subspace_angle_x_deg"] = np.degrees(cca["subspace_angle_x"])
        if "subspace_angle_y" in cca:
            summary["subspace_angle_y_deg"] = np.degrees(cca["subspace_angle_y"])
        if "loading_correlation_x" in cca:
            summary["loading_corr_x"] = cca["loading_correlation_x"]["mean_correlation"]
        if "loading_correlation_y" in cca:
            summary["loading_corr_y"] = cca["loading_correlation_y"]["mean_correlation"]

    return summary
