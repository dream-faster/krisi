from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf, pacf, q_stat

from krisi.evaluate.scorecard import SampleTypes, ScoreCard


def metric_hoc(func: Callable, *args, **kwargs) -> Callable:
    def wrap(y, predictions):
        return func(y, predictions, *args, **kwargs)

    return wrap


default_scoring_functions = [
    # ("pacf", metric_hoc(pacf, alpha=0.05)),
    # ("acf", metric_hoc(acf, alpha=0.05)),
]


def evaluate(
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    y: pd.Series,
    predictions: Union[np.ndarray, pd.Series],
    scoring_functions: List[Tuple[str, Callable]] = default_scoring_functions,
) -> ScoreCard:
    summary = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )

    for score_name, score_function in scoring_functions:
        summary[score_name] = score_function(predictions, y)

    # alpha = 0.05
    # summary.pacf_res = pacf(predictions, alpha=alpha)
    # summary.acf_res = acf(predictions, alpha=alpha)

    """ Residual Diagnostics """
    residuals = y - predictions
    residuals_mean = residuals.mean()
    residual_std = residuals.std()

    """ Forecast Errors - Regression """
    # mae =
    summary.mse = mean_squared_error(y, predictions, squared=True)
    # rmse = mean_squared_error(y, predictions, squared=False)

    """ Forecast Errors - Classification """

    # if sample_type.insample:
    #     summary["ljung_box"] = q_stat(acf_res, len(predictions))

    return summary


def evaluate_in_out_sample(
    model_name: str,
    dataset_name: str,
    y_insample: pd.Series,
    insample_predictions: pd.Series,
    y_outsample: pd.Series,
    outsample_predictions: pd.Series,
    scoring_functions: List[Tuple[str, Callable]] = default_scoring_functions,
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.insample,
        y_insample,
        insample_predictions,
        scoring_functions=scoring_functions,
    )
    outsample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.outsample,
        y_outsample,
        outsample_predictions,
        scoring_functions=scoring_functions,
    )

    return insample_summary, outsample_summary
