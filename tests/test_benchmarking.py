from copy import deepcopy
from typing import List, Optional

import pandas as pd

from krisi.evaluate import score
from krisi.evaluate.group import Group
from krisi.evaluate.library.benchmarking import RandomClassifier
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

    model = RandomClassifier([0, 1], [0.0, 1.0])
    df = model.predict(y)
    predictions = df.iloc[:, 0]
    probabilities = df.iloc[:, 1:]

    new_metrics = []
    for metric in all_metrics:
        copy_of_metric = deepcopy(metric)
        copy_of_metric.result = None
        copy_of_metric.result_rolling = None
        copy_of_metric.key += f"_difference_{model.__class__.__name__}"
        copy_of_metric.name += f" difference with {model.__class__.__name__}"
        if copy_of_metric.accepts_probabilities:
            copy_of_metric.evaluate(y, probabilities)
        else:
            copy_of_metric.evaluate(y, predictions)
        copy_of_metric.result = copy_of_metric.result - metric.result
        new_metrics.append(copy_of_metric)
    return all_metrics + new_metrics


def test_benchmarking_random():
    groupped_metric = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=[f_one_score_macro],
        postprocess_funcs=example_postporcess_func,
    )
    X, y = generate_synthetic_data(task=Task.classification)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    probabilities = preds_probs.iloc[:, :2]
    predictions = preds_probs.iloc[:, 2]

    sc = score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=[groupped_metric],
    )
    sc.print()


test_benchmarking_random()
