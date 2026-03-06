"""
EvaluationResults - Container for all evaluation outputs.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import json
from pathlib import Path
import numpy as np


@dataclass
class CCAValidationResults:
    """Results from CCA validation."""
    n_repeats: int
    n_folds: int
    train_correlations: np.ndarray  # (n_repeats, n_folds, n_components)
    test_correlations: np.ndarray   # (n_repeats, n_folds, n_components)

    @property
    def mean_train_correlation(self) -> float:
        """Mean train correlation (first component)."""
        return float(np.mean(self.train_correlations[:, :, 0]))

    @property
    def mean_test_correlation(self) -> float:
        """Mean test correlation (first component)."""
        return float(np.mean(self.test_correlations[:, :, 0]))

    @property
    def std_test_correlation(self) -> float:
        """Std of test correlation (first component)."""
        return float(np.std(self.test_correlations[:, :, 0]))

    @property
    def generalization_ratio(self) -> float:
        """Mean test / mean train correlation."""
        train = self.mean_train_correlation
        if train == 0:
            return 0.0
        return self.mean_test_correlation / train

    @property
    def n_components(self) -> int:
        return self.test_correlations.shape[2]

    @property
    def per_component_test_correlations(self) -> Dict[int, Dict[str, float]]:
        """Per-component test correlation stats: {idx: {mean, std}}."""
        result = {}
        for c in range(self.n_components):
            vals = self.test_correlations[:, :, c]
            result[c] = {
                'mean': float(np.nanmean(vals)),
                'std': float(np.nanstd(vals))
            }
        return result

    @property
    def per_component_generalization_ratio(self) -> Dict[int, float]:
        """Per-component test/train ratio."""
        result = {}
        for c in range(self.n_components):
            train_mean = float(np.nanmean(self.train_correlations[:, :, c]))
            test_mean = float(np.nanmean(self.test_correlations[:, :, c]))
            result[c] = test_mean / train_mean if train_mean != 0 else 0.0
        return result

    def to_dict(self) -> Dict:
        d = {
            'n_repeats': self.n_repeats,
            'n_folds': self.n_folds,
            'mean_train_correlation': self.mean_train_correlation,
            'mean_test_correlation': self.mean_test_correlation,
            'std_test_correlation': self.std_test_correlation,
            'generalization_ratio': self.generalization_ratio,
            'per_component_test_correlations': {
                str(k): v for k, v in self.per_component_test_correlations.items()
            },
            'per_component_generalization_ratio': {
                str(k): v for k, v in self.per_component_generalization_ratio.items()
            },
            'train_correlations': self.train_correlations.tolist(),
            'test_correlations': self.test_correlations.tolist()
        }
        return d


@dataclass
class ClusteringValidationResults:
    """Results from clustering validation for one config."""
    config_name: str
    n_repeats: int
    n_folds: int
    test_silhouettes: np.ndarray  # (n_repeats, n_folds)
    subject_consistency: np.ndarray  # (n_subjects,) - % in modal cluster
    final_labels: np.ndarray  # (n_subjects,) - modal label per subject
    consensus_matrix: Optional[np.ndarray] = None  # (n_subjects, n_subjects)

    @property
    def mean_test_silhouette(self) -> float:
        """Mean test silhouette score."""
        return float(np.nanmean(self.test_silhouettes))

    @property
    def std_test_silhouette(self) -> float:
        """Std of test silhouette score."""
        return float(np.nanstd(self.test_silhouettes))

    @property
    def mean_subject_consistency(self) -> float:
        """Mean subject consistency across all subjects."""
        return float(np.mean(self.subject_consistency))

    @property
    def unstable_subject_count(self) -> int:
        """Number of subjects with consistency < 0.7."""
        return int(np.sum(self.subject_consistency < 0.7))

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {
            'config_name': self.config_name,
            'n_repeats': self.n_repeats,
            'n_folds': self.n_folds,
            'test_silhouettes': self.test_silhouettes.tolist(),
            'mean_test_silhouette': self.mean_test_silhouette,
            'std_test_silhouette': self.std_test_silhouette,
            'mean_subject_consistency': self.mean_subject_consistency,
            'unstable_subject_count': self.unstable_subject_count,
            'subject_consistency': self.subject_consistency.tolist(),
            'final_labels': self.final_labels.tolist()
        }
        if self.consensus_matrix is not None:
            result['consensus_matrix'] = self.consensus_matrix.tolist()
        return result


@dataclass
class PermutationTestResults:
    """Results from permutation testing."""
    observed_distribution: np.ndarray  # (n_samples,) real test correlations
    null_distribution: np.ndarray      # (n_samples,) permuted test correlations
    p_value: Optional[float] = None

    @property
    def observed_mean(self) -> float:
        return float(np.mean(self.observed_distribution))

    @property
    def null_mean(self) -> float:
        return float(np.mean(self.null_distribution))

    @property
    def computed_p_value(self) -> float:
        """Empirical p-value: proportion of null >= observed mean."""
        if self.p_value is not None:
            return self.p_value
        observed = self.observed_mean
        return float(np.mean(self.null_distribution >= observed))

    @property
    def separation(self) -> str:
        threshold = np.percentile(self.null_distribution, 95)
        overlap = np.mean(self.observed_distribution < threshold)
        if overlap < 0.05:
            return "clear"
        elif overlap < 0.20:
            return "moderate"
        else:
            return "overlapping"

    def to_dict(self) -> Dict:
        return {
            'observed_mean': self.observed_mean,
            'null_mean': self.null_mean,
            'p_value': self.computed_p_value,
            'separation': self.separation,
            'observed_distribution': self.observed_distribution.tolist(),
            'null_distribution': self.null_distribution.tolist()
        }


@dataclass
class EvaluationResults:
    """
    Container for all evaluation outputs.

    Contains CCA validation, clustering validation (per config),
    and permutation test results.
    """
    cca_validation: Optional[CCAValidationResults] = None
    clustering_validation: Dict[str, ClusteringValidationResults] = field(
        default_factory=dict
    )
    permutation_test: Optional[PermutationTestResults] = None
    config: Optional[Dict] = None  # EvaluationConfig.to_dict()

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = ["=" * 50, "BIOTYPE EVALUATION RESULTS", "=" * 50, ""]

        if self.cca_validation:
            cca = self.cca_validation
            lines.extend([
                "CCA VALIDATION",
                "-" * 30,
                f"  Test correlation: {cca.mean_test_correlation:.3f} "
                f"± {cca.std_test_correlation:.3f}",
                f"  Generalization ratio: {cca.generalization_ratio:.3f}",
            ])
            if cca.n_components > 1:
                lines.append("  Per-component:")
                for c, stats in cca.per_component_test_correlations.items():
                    ratio = cca.per_component_generalization_ratio[c]
                    lines.append(
                        f"    CC{c+1}: {stats['mean']:.3f} ± {stats['std']:.3f} "
                        f"(gen ratio {ratio:.3f})"
                    )
            lines.append("")

        if self.clustering_validation:
            lines.extend(["CLUSTERING VALIDATION", "-" * 30])
            for name, results in self.clustering_validation.items():
                lines.extend([
                    f"  [{name}]",
                    f"    Test silhouette: {results.mean_test_silhouette:.3f} "
                    f"± {results.std_test_silhouette:.3f}",
                    f"    Subject consistency: {results.mean_subject_consistency:.1%}",
                    f"    Unstable subjects: {results.unstable_subject_count}",
                    ""
                ])

        if self.permutation_test:
            pt = self.permutation_test
            lines.extend([
                "PERMUTATION TEST",
                "-" * 30,
                f"  Observed mean: {pt.observed_mean:.3f}",
                f"  Null mean: {pt.null_mean:.3f}",
                f"  p-value: {pt.computed_p_value:.4f}",
                f"  Separation: {pt.separation}",
                ""
            ])

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize to dictionary."""
        result = {}

        if self.cca_validation:
            result['cca_validation'] = self.cca_validation.to_dict()

        if self.clustering_validation:
            result['clustering_validation'] = {
                name: cv.to_dict()
                for name, cv in self.clustering_validation.items()
            }

        if self.permutation_test:
            result['permutation_test'] = self.permutation_test.to_dict()

        if self.config:
            result['config'] = self.config

        return result

    @classmethod
    def from_dict(cls, d: Dict) -> 'EvaluationResults':
        """Create from dictionary."""
        cca_validation = None
        if 'cca_validation' in d and d['cca_validation']:
            cca_data = d['cca_validation']
            if 'train_correlations' in cca_data:
                cca_validation = CCAValidationResults(
                    n_repeats=cca_data['n_repeats'],
                    n_folds=cca_data['n_folds'],
                    train_correlations=np.array(cca_data['train_correlations']),
                    test_correlations=np.array(cca_data['test_correlations'])
                )

        clustering_validation = {}
        if 'clustering_validation' in d:
            for name, cv_data in d['clustering_validation'].items():
                if 'subject_consistency' in cv_data:
                    consensus = None
                    if 'consensus_matrix' in cv_data:
                        consensus = np.array(cv_data['consensus_matrix'])
                    # Handle test_silhouettes with proper shape for backwards compatibility
                    n_repeats = cv_data['n_repeats']
                    n_folds = cv_data['n_folds']
                    if 'test_silhouettes' in cv_data and cv_data['test_silhouettes']:
                        test_sil = np.array(cv_data['test_silhouettes'])
                    else:
                        # Create NaN array with proper shape if missing
                        test_sil = np.full((n_repeats, n_folds), np.nan)

                    clustering_validation[name] = ClusteringValidationResults(
                        config_name=name,
                        n_repeats=n_repeats,
                        n_folds=n_folds,
                        test_silhouettes=test_sil,
                        subject_consistency=np.array(cv_data['subject_consistency']),
                        final_labels=np.array(cv_data['final_labels']),
                        consensus_matrix=consensus
                    )

        permutation_test = None
        if 'permutation_test' in d and d['permutation_test']:
            pt_data = d['permutation_test']
            if 'observed_distribution' in pt_data:
                permutation_test = PermutationTestResults(
                    observed_distribution=np.array(pt_data['observed_distribution']),
                    null_distribution=np.array(pt_data['null_distribution']),
                    p_value=pt_data.get('p_value')
                )

        return cls(
            cca_validation=cca_validation,
            clustering_validation=clustering_validation,
            permutation_test=permutation_test,
            config=d.get('config')
        )

    def save(self, path: str):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'EvaluationResults':
        """Load results from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
