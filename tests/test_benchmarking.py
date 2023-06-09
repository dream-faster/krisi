from typing import List, Optional

import pandas as pd

from krisi.evaluate import score
from krisi.evaluate.group import Group
from krisi.evaluate.library.default_metrics_classification import f_one_score_macro
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import PredictionsDS, ProbabilitiesDF, TargetsDS, WeightsDS
from krisi.utils.devutils.data import (
    generate_synthetic_data,
    generate_synthetic_predictions_binary,
)
from krisi.utils.devutils.type import Task


def example_postporcess_func(
    all_metrics: List[Metric],
    y: TargetsDS,
    predictions: Optional[PredictionsDS],
    probabilities: Optional[ProbabilitiesDF],
    sample_weight: Optional[WeightsDS],
    rolling: bool,
) -> List[Metric]:
    metric = all_metrics[0]
    if not rolling:
        if isinstance(metric.result, Exception) or metric.result is None:
            return metric
        if isinstance(metric.result, List):
            metric.result = pd.Series(metric.result) + 1
        else:
            metric.result = metric.result + 1
    return all_metrics


def test_benchmarking_random():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[f_one_score_macro],
        postprocess_funcs=example_postporcess_func,
    )
    X, y = generate_synthetic_data(task=Task.classification)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    probabilities = preds_probs.iloc[:, :2]
    predictions = preds_probs.iloc[:, 2]

    score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=[groupped_metric],
    )
