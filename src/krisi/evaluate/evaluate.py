from typing import Any, Callable, Tuple, Union

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import MetricFunction, Predictions, SampleTypes, Targets


def metric_hoc(func: MetricFunction, *args, **kwargs) -> Callable:
    def wrap(y: Targets, predictions: Predictions) -> Any:
        return func(y, predictions, *args, **kwargs)

    return wrap


def evaluate(
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    y: Targets,
    predictions: Predictions,
) -> ScoreCard:

    sc = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )

    """ Custom Metrics """
    for metric in sc.get_default_metrics():
        if (
            metric.restrict_to_sample is not None
            and metric.restrict_to_sample is sample_type
        ):
            continue

        wrapped_func = metric_hoc(metric.func, **metric.hyperparameters)
        metric.result = wrapped_func(y, predictions)
        sc[metric.name] = metric

    return sc


def evaluate_in_out_sample(
    model_name: str,
    dataset_name: str,
    y_insample: pd.Series,
    insample_predictions: pd.Series,
    y_outsample: pd.Series,
    outsample_predictions: pd.Series,
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.insample,
        y_insample,
        insample_predictions,
    )
    outsample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.outsample,
        y_outsample,
        outsample_predictions,
    )

    return insample_summary, outsample_summary
