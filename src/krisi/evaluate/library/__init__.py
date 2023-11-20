from typing import List

from krisi.evaluate.metric import Metric
from krisi.evaluate.type import DatasetType

from .benchmarking_models import *
from .default_metrics_classification import ClassificationRegistry
from .default_metrics_regression import RegressionRegistry


def get_default_metrics_for_dataset_type(type: DatasetType) -> List[Metric]:
    if type == DatasetType.classification_binary_balanced:
        return ClassificationRegistry().binary_classification_balanced_metrics
    elif type == DatasetType.classification_binary_imbalanced:
        return ClassificationRegistry().binary_classification_imbalanced_metrics
    elif type == DatasetType.classification_multiclass:
        return ClassificationRegistry().multiclass_classification_metrics
    elif type == DatasetType.regression:
        return RegressionRegistry().all_regression_metrics
    else:
        raise ValueError(f"Unknown dataset type {type}")
