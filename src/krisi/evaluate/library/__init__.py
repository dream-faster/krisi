from typing import List

from krisi.evaluate.metric import Metric
from krisi.evaluate.type import DatasetType

from .default_metrics_classification import *
from .default_metrics_classification import (
    binary_classification_balanced_metrics,
    binary_classification_imbalanced_metrics,
    binary_classification_metrics_balanced_benchmarking,
    binary_classification_metrics_imbalanced_benchmarking,
    minimal_binary_classification_metrics,
    minimal_multiclass_classification_metrics,
    multiclass_classification_metrics,
)
from .default_metrics_regression import *
from .default_metrics_regression import (
    all_regression_metrics,
    low_computation_regression_metrics,
    minimal_regression_metrics,
)


def get_default_metrics_for_dataset_type(type: DatasetType) -> List[Metric]:
    if type == DatasetType.classification_binary_balanced:
        return binary_classification_metrics_balanced_benchmarking
    elif type == DatasetType.classification_binary_imbalanced:
        return binary_classification_metrics_imbalanced_benchmarking
    elif type == DatasetType.classification_multiclass:
        return multiclass_classification_metrics
    elif type == DatasetType.regression:
        return all_regression_metrics
    else:
        raise ValueError(f"Unknown dataset type {type}")
