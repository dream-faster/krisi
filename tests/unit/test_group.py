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
        preprocess_func=lambda y, pred: y - pred,
        postprocess_funcs=standard_deviation,
    )

    y = pd.Series(np.random.randint(0, 2, 1000))
    predictions = pd.Series(np.random.randint(0, 2, 1000))

    groupped_metric.evaluate_over_time(y, predictions, None, {"window": 10})


def test_group_postprocess():
    groupped_metric = Group[pd.Series](
        name="residual_group",
        key="residual_group",
        metrics=[f_one_score_macro],
        postprocess_funcs=standard_deviation,
    )

    y = pd.Series(np.random.randint(0, 2, 1000))
    predictions = pd.Series(np.random.randint(0, 2, 1000))

    groupped_metric.evaluate_over_time(y, predictions, None, {"window": 10})
