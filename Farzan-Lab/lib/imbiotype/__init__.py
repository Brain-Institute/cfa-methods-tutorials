"""
IMBiotype - A library for multimodal data integration and biotype analysis
"""

__version__ = "0.1.0"

from .core.study import Study
from .core.data_manager import DataManager
from .pipeline.pipeline import Pipeline
from .models.fusion import FusionModel
from .models.clustering import ClusteringModel 