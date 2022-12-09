from typing import Callable, List, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf, q_stat

from krisi.evaluate.scorecard import SampleTypes, ScoreCard
from krisi.explore.utils import generating_arima_synthetic_data
from krisi.utils.models import (
    Model,
    default_arima_model,
    default_naive_model,
    generate_univariate_predictions,
)
 


def metric_hoc(func: Callable, *args, **kwargs) -> Callable:
    def wrap(*args2, **kwargs2):
        return func(*args, **kwargs, *args2, **kwargs2)

    return wrap



default_scoring_functions = [("pacf",metric_hoc(pacf, alpha=0.05)), ("acf",metric_hoc(acf, alpha=0.05))]

def evaluate( 
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    # X: pd.DataFrame,
    y: pd.Series,
    predictions: Union[np.ndarray, pd.Series],
    scoring_functions: List[Tuple[str, Callable]] = default_scoring_functions,
) -> ScoreCard:
    summary = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )
    
    for score_name, score_function in scoring_functions:
        summary[score_name] = score_function(predictions, y)
    

    alpha = 0.05
    pacf_res = pacf(predictions, alpha=alpha)
    acf_res = acf(predictions, alpha=alpha)

    # summary.update("ljung_box", q_stat(acf_res, len(X)))

    return summary


def evaluate_in_out_sample(
    model_name, model, dataset_name, y, scoring_functions: List[Tuple[str, Callable]] = default_scoring_functions,
) -> Tuple[ScoreCard, ScoreCard]:
    insample_predictions, outsample_predictions = generate_univariate_predictions(
        model, df, dataset_name
    )

    insample_summary = evaluate(
        model_name, dataset_name, SampleTypes.insample, y, insample_predictions, scoring_functions=scoring_functions
    )
    outsample_summary = evaluate(
        model_name, dataset_name, SampleTypes.outsample, y, outsample_predictions, scoring_functions=scoring_functions
    )

    return insample_summary, outsample_summary


# def evaluate_models(
#     models: List[Tuple[str, Model]]
# ) -> List[Tuple[ScoreCard, ScoreCard]]:

#     dataset_name = "synthetic"
#     df = generating_arima_synthetic_data(
#         target_col=dataset_name,
#         nsample=1000,
#     ).to_frame()

#     return [
#         evaluate_in_out_sample(model_name, model, dataset_name, df)
#         for model_name, model in models
#     ]


# if __name__ == "__main__":
#     models = [
#         ("default_naive_last", default_naive_model),
#         ("default_arima", default_arima_model),
#     ]
#     evaluate_models(models)
