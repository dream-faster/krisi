from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.stattools import acf, pacf, q_stat

from krisi.evaluate.scorecard import Metric, SampleTypes, ScoreCard


def metric_hoc(func: Callable, *args, **kwargs) -> Callable:
    def wrap(y, predictions):
        return func(y, predictions, *args, **kwargs)

    return wrap


default_scoring_functions = [
    ("mse", mean_squared_error, {"squared": True}),
]


def evaluate(
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    y: pd.Series,
    predictions: Union[np.ndarray, pd.Series],
    scoring_functions: List[Tuple[str, Callable, dict]] = default_scoring_functions,
) -> ScoreCard:
    sc = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )

    """ Custom Metrics """
    for score_name, score_function, score_hyperparameters in scoring_functions:
        func = metric_hoc(score_function, **score_hyperparameters)
        sc[score_name] = Metric(
            name=score_name,
            metric_result=func(predictions, y),
            hyperparameters=score_hyperparameters,
        )

    """ Correlations """
    alpha = 0.05
    sc.pacf_res = pacf(predictions, alpha=alpha)
    sc.pacf_res.hyperparameters = {"alpha": alpha}
    sc.acf_res = acf(predictions, alpha=alpha)
    sc.acf_res.hyperparameters = {"alpha": alpha}

    """ Residual Diagnostics """
    residuals = y - predictions
    sc.residuals_mean = residuals.mean()
    sc.residuals_std = residuals.std()

    """ Forecast Errors - Regression """
    sc.mae = mean_absolute_error(y, predictions)
    # sc.mse = mean_squared_error(y, predictions, squared=True)
    sc.rmse = mean_squared_error(y, predictions, squared=False)

    """ Forecast Errors - Classification """

    # if sample_type.insample:
    #     summary["ljung_box"] = q_stat(acf_res, len(predictions))

    return sc


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
