"""
Configuration classes for biotype evaluation.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional, List, Callable


class EvaluationCancelled(Exception):
    """Raised when an evaluation run is cancelled."""
    pass


@dataclass
class EvaluationConfig:
    """Configuration for biotype evaluation with nested cross-validation."""
    n_outer_folds: int = 10
    n_repeats: int = 100
    random_state: int = 42
    stratify_by: Optional[str] = None

    reduced_inner_hpo_repeats: int = 10

    optimize_n_clusters: bool = False
    n_clusters_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5])

    run_permutation_test: bool = True

    label_reference_strategy: str = 'repeat_1'

    n_jobs: int = -1

    progress_callback: Optional[Callable[[int, int], None]] = None

    # Cancellation — not serialized
    _cancel_event: threading.Event = field(
        default_factory=threading.Event, repr=False, compare=False
    )

    def __post_init__(self):
        if self.n_outer_folds < 2:
            raise ValueError("n_outer_folds must be at least 2")
        if self.n_repeats < 1:
            raise ValueError("n_repeats must be at least 1")
        if self.label_reference_strategy not in ('repeat_1', 'full_dataset'):
            raise ValueError(
                f"label_reference_strategy must be 'repeat_1' or 'full_dataset', "
                f"got '{self.label_reference_strategy}'"
            )

    def request_cancel(self):
        """Signal cancellation from another thread."""
        self._cancel_event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def check_cancelled(self):
        """Raise EvaluationCancelled if cancellation was requested."""
        if self.is_cancelled:
            raise EvaluationCancelled("Evaluation cancelled by user")

    def to_dict(self) -> dict:
        return {
            'n_outer_folds': self.n_outer_folds,
            'n_repeats': self.n_repeats,
            'random_state': self.random_state,
            'stratify_by': self.stratify_by,
            'reduced_inner_hpo_repeats': self.reduced_inner_hpo_repeats,
            'optimize_n_clusters': self.optimize_n_clusters,
            'n_clusters_range': self.n_clusters_range,
            'run_permutation_test': self.run_permutation_test,
            'label_reference_strategy': self.label_reference_strategy,
            'n_jobs': self.n_jobs,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'EvaluationConfig':
        # Exclude non-serializable fields
        exclude = {'progress_callback', '_cancel_event'}
        return cls(**{k: v for k, v in d.items() if k not in exclude})
