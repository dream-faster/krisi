from typing import List, Tuple

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import CalculationTypes, Predictions, SampleTypes, Targets


def evaluate(
    model_name: str,
    dataset_name: str,
    sample_type: SampleTypes,
    calculation_types: List[CalculationTypes],
    y: Targets,
    predictions: Predictions,
    window: int = 30,
) -> ScoreCard:

    sc = ScoreCard(
        model_name=model_name,
        dataset_name=dataset_name,
        sample_type=sample_type,
    )

    for metric in sc.get_default_metrics():
        if metric.restrict_to_sample is sample_type:
            continue

        for calculation_type in calculation_types:
            if calculation_type == CalculationTypes.single:
                metric.evaluate(y, predictions)
            if calculation_type == CalculationTypes.rolling:
                metric.evaluate_over_time(y, predictions, window=window)

    return sc


def evaluate_in_out_sample(
    model_name: str,
    dataset_name: str,
    calculation_types: List[CalculationTypes],
    y_insample: pd.Series,
    insample_predictions: pd.Series,
    y_outsample: pd.Series,
    outsample_predictions: pd.Series,
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.insample,
        calculation_types,
        y_insample,
        insample_predictions,
    )
    outsample_summary = evaluate(
        model_name,
        dataset_name,
        SampleTypes.outofsample,
        calculation_types,
        y_outsample,
        outsample_predictions,
    )

    return insample_summary, outsample_summary
