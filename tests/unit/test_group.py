from typing import List, Optional

import numpy as np
import pandas as pd

from krisi.evaluate.group import Group
from krisi.evaluate.library.default_metrics_classification import f_one_score_macro
from krisi.evaluate.library.default_metrics_regression import residuals_mean
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import PredictionsDS, ProbabilitiesDF, TargetsDS, WeightsDS


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


def test_group_preprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[residuals_mean()],
        preprocess_func=lambda y, pred, probs, **kwargs: y - pred,
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
        metrics=[f_one_score_macro()],
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
