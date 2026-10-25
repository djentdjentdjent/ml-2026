"""ML-компоненты для компенсации систематической ошибки ближнего поля."""
from .config import MLConfig, TrainingConfig
from .dataset import FeatureExtractor, generate_dataset, source_descriptor
from .metrics import evaluate_predictions, distance_bin_metrics
from .models import NearFieldCorrector
from .preprocessing import annotate_filtering, filter_dataset
from .splitting import split_dataset

__all__ = [
    "MLConfig",
    "TrainingConfig",
    "FeatureExtractor",
    "generate_dataset",
    "source_descriptor",
    "NearFieldCorrector",
    "filter_dataset",
    "annotate_filtering",
    "split_dataset",
    "evaluate_predictions",
    "distance_bin_metrics",
]
