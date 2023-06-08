from typing import List

import numpy as np
import pandas as pd

from krisi.evaluate.group import Group
from krisi.evaluate.library.default_metrics_regression import residuals_mean
from krisi.evaluate.metric import Metric


def example_postporcess_func(
    metric: Metric, sample_weight: pd.Series, rolling: bool
) -> Metric:
    if rolling:
        if isinstance(metric.result, Exception) or metric.result is None:
            return metric
        if isinstance(metric.result, List):
            metric.result = pd.Series(metric.result) + 1
        else:
            metric.result = metric.result + 1
    return metric


def test_group_preprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[residuals_mean],
        preprocess_func=lambda y, pred, **kwargs: y - pred,
        postprocess_funcs=example_postporcess_func,
    )

    y = pd.Series(np.random.randint(0, 2, 1000))
    predictions = pd.Series(np.random.randint(0, 2, 1000))
    groupped_metric.evaluate(y, predictions, None, None)
    groupped_metric.evaluate_over_time(
        y,
        predictions,
        None,
        None,
        rolling_args={"window": 10, "min_periods": 10, "step": 10, "closed": "left"},
    )


def test_group_postprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[residuals_mean],
        postprocess_funcs=example_postporcess_func,
    )

    y = pd.Series(np.random.randint(0, 2, 1000))
    predictions = pd.Series(np.random.randint(0, 2, 1000))

    groupped_metric.evaluate_over_time(
        y,
        predictions,
        None,
        None,
        rolling_args={"window": 10, "min_periods": 10, "step": 10, "closed": "left"},
    )
