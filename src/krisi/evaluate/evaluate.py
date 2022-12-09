from typing import List, Tuple, Union

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


def evaluate(
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    features: pd.DataFrame,
    predictions: Union[np.ndarray, pd.Series],
) -> ScoreCard:
    summary = ScoreCard(
        model_name=model_name, dataset_name=dataset_name, sample_type=sample_type
    )

    alpha = 0.05
    pacf_res = pacf(predictions, alpha=alpha)
    acf_res = acf(predictions, alpha=alpha)

    summary.update("ljung_box", q_stat(acf_res, len(features)))

    return summary


def evaluate_in_out_sample(
    model_name, model, dataset_name, df
) -> Tuple[ScoreCard, ScoreCard]:
    insample_predictions, outsample_predictions = generate_univariate_predictions(
        model, df, dataset_name
    )

    insample_summary = evaluate(
        model_name, dataset_name, SampleTypes.insample, df, insample_predictions
    )
    outsample_summary = evaluate(
        model_name, dataset_name, SampleTypes.outsample, df, outsample_predictions
    )

    return insample_summary, outsample_summary


def evaluate_models(
    models: List[Tuple[str, Model]]
) -> List[Tuple[ScoreCard, ScoreCard]]:

    dataset_name = "synthetic"
    df = generating_arima_synthetic_data(
        target_col=dataset_name,
        nsample=1000,
    ).to_frame()

    return [
        evaluate_in_out_sample(model_name, model, dataset_name, df)
        for model_name, model in models
    ]


if __name__ == "__main__":
    models = [
        ("default_naive_last", default_naive_model),
        ("default_arima", default_arima_model),
    ]
    evaluate_models(models)
