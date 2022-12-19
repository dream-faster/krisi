from typing import Any, Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf, pacf, q_stat

from krisi.evaluate.library.default_metrics import default_metrics
from krisi.evaluate.scorecard import Metric, ScoreCard
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
    default_metrics: List[Metric] = default_metrics,
) -> ScoreCard:

    sc = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )

    """ Custom Metrics """
    for metric in default_metrics:
        if (
            metric.restrict_to_sample is not None
            and metric.restrict_to_sample is not sample_type
        ):
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
    default_metrics: List[Metric] = default_metrics,
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.insample,
        y_insample,
        insample_predictions,
        default_metrics=default_metrics,
    )
    outsample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.outsample,
        y_outsample,
        outsample_predictions,
        default_metrics=default_metrics,
    )

    return insample_summary, outsample_summary
