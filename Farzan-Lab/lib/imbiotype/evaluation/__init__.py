"""
Biotype Evaluation Framework

Provides nested cross-validation for validating CCA + clustering biotype pipelines.
"""

from .config import EvaluationConfig, EvaluationCancelled
from .evaluator import BiotypeEvaluator
from .results import (
    EvaluationResults,
    CCAValidationResults,
    ClusteringValidationResults,
    PermutationTestResults
)
from .splitter import TrainTestSplitter
from .cca_validator import CCAValidator, CCAFoldResult
from .clustering_validator import ClusteringValidator, ClusteringFoldResult
from .hungarian import LabelMatcher, match_labels, compute_mapping
from .subject_tracker import SubjectTracker
from .consensus import ConsensusMatrixBuilder
from .permutation import PermutationTester, PermutationResult
from .pipeline_validator import PipelineValidator, PipelineFoldResult

__all__ = [
    # Main classes
    'EvaluationConfig',
    'EvaluationCancelled',
    'BiotypeEvaluator',
    'EvaluationResults',
    # Result containers
    'CCAValidationResults',
    'ClusteringValidationResults',
    'PermutationTestResults',
    # Validators
    'CCAValidator',
    'CCAFoldResult',
    'ClusteringValidator',
    'ClusteringFoldResult',
    # Utilities
    'TrainTestSplitter',
    'LabelMatcher',
    'match_labels',
    'compute_mapping',
    'SubjectTracker',
    'ConsensusMatrixBuilder',
    'PermutationTester',
    'PermutationResult',
    'PipelineValidator',
    'PipelineFoldResult',
]
