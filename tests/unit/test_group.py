import numpy as np
import pandas as pd

from krisi.evaluate.group import Group
from krisi.evaluate.library.default_metrics_classification import (
    f_one_score_macro,
    standard_deviation,
)


def test_group_preprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[f_one_score_macro],
        preprocess_func=lambda y, pred, **kwargs: y - pred,
        postprocess_funcs=standard_deviation,
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


def test_group_postprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[f_one_score_macro],
        postprocess_funcs=standard_deviation,
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
