from typing import Callable, Optional, Union

import numpy as np
import pandas as pd

from krisi import library, score
from krisi.evaluate.group import Group
from krisi.evaluate.type import (
    Calculation,
    PredictionsDS,
    ProbabilitiesDF,
    Targets,
    WeightsDS,
)


def modify_metric(metric, func_: Callable):
    if isinstance(metric, Group):
        for m in metric.metrics:
            m.func = func_(m.parameters)
    else:
        metric.func = func_(metric.parameters)

    return metric


def test_args():
    data_len = 100

    def dummy_func_inject_kwags(parameters):
        def dummy_func(
            y: Targets,
            pred: Union[PredictionsDS, ProbabilitiesDF],
            sample_weight: Optional[WeightsDS],
            **kwargs,
        ):
            assert len(y) == data_len and y.isna().sum() == 0
            assert len(pred) == data_len

            if isinstance(pred, pd.DataFrame):
                assert pred.isna().sum().sum() == 0
            else:
                assert pred.isna().sum() == 0

            assert len(sample_weight) == data_len and sample_weight.isna().sum() == 0
            assert parameters == kwargs

            return 1.0

        return dummy_func

    modified_metrics = [
        modify_metric(metric, dummy_func_inject_kwags)
        for metric in (
            library.ClassificationRegistry().binary_classification_imbalanced_metrics
        )
    ]
    probs_0 = pd.Series(np.random.rand(data_len))
    score(
        pd.Series(np.random.randint(100, size=data_len)),
        pd.Series(np.random.randint(100, size=data_len)),
        pd.concat([probs_0, probs_0 - 1], axis="columns", copy=False),
        sample_weight=pd.Series(np.random.rand(data_len)),
        default_metrics=modified_metrics,
        calculation=Calculation.single,
    )


def test_args_rolling():
    data_len = 100
    window_size = 5
    min_periods = window_size

    def dummy_func_inject_kwags(parameters):
        def dummy_func(
            y: Targets,
            pred: Union[PredictionsDS, ProbabilitiesDF],
            sample_weight: Optional[WeightsDS],
            **kwargs,
        ):
            assert len(y) == window_size and y.isna().sum() == 0
            assert len(pred) == window_size

            if isinstance(pred, pd.DataFrame):
                assert pred.isna().sum().sum() == 0
            else:
                assert pred.isna().sum() == 0

            assert len(sample_weight) == window_size and sample_weight.isna().sum() == 0
            assert parameters == kwargs

            return 1.0

        return dummy_func

    modified_metrics = [
        modify_metric(metric, dummy_func_inject_kwags)
        for metric in (
            library.ClassificationRegistry().binary_classification_imbalanced_metrics
        )
    ]
    probs_0 = pd.Series(np.random.rand(data_len))
    score(
        pd.Series(np.random.randint(100, size=data_len)),
        pd.Series(np.random.randint(100, size=data_len)),
        pd.concat([probs_0, probs_0 - 1], axis="columns", copy=False),
        sample_weight=pd.Series(np.random.rand(data_len)),
        default_metrics=modified_metrics,
        calculation=Calculation.rolling,
        rolling_args={"window": window_size, "min_periods": min_periods},
    )
