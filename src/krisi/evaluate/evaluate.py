import uuid
from typing import List, Optional, Tuple, Union

import pandas as pd

from krisi.evaluate.metric import Metric
from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import CalculationTypes, Predictions, SampleTypes, Targets


def evaluate(
    y: Targets,
    predictions: Predictions,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    sample_type: SampleTypes = SampleTypes.outofsample,
    calculation_types: List[Union[CalculationTypes, str]] = [
        CalculationTypes.single,
        CalculationTypes.rolling,
    ],
    window: int = 30,
) -> ScoreCard:

    if dataset_name is None:
        if isinstance(y, pd.Series) and y.name is not None:
            dataset_name = y.name
        else:
            dataset_name = f"Dataset:{str(uuid.uuid4()).split('-')[0]}"

    if model_name is None:
        model_name = f"Model:{str(uuid.uuid4()).split('-')[0]}"

    sc = ScoreCard(
        model_name=model_name,
        dataset_name=dataset_name,
        sample_type=sample_type,
        custom_metrics=custom_metrics,
    )

    for calculation_type in calculation_types:
        if (
            calculation_type == CalculationTypes.single
            or calculation_type == CalculationTypes.single.value
        ):
            sc.evaluate(y, predictions)
        if (
            calculation_type == CalculationTypes.rolling
            or calculation_type == CalculationTypes.rolling.value
        ):
            sc.evaluate_over_time(y, predictions, window=window)

    return sc


def evaluate_in_out_sample(
    y_insample: pd.Series,
    insample_predictions: pd.Series,
    y_outsample: pd.Series,
    outsample_predictions: pd.Series,
    model_name: str = "Unknown model",
    dataset_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    calculation_types: List[Union[CalculationTypes, str]] = [
        CalculationTypes.single,
        CalculationTypes.rolling,
    ],
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = evaluate(
        y_insample,
        insample_predictions,
        model_name,
        dataset_name,
        custom_metrics,
        SampleTypes.insample,
        calculation_types,
    )
    outsample_summary = evaluate(
        y_outsample,
        outsample_predictions,
        model_name,
        dataset_name,
        custom_metrics,
        SampleTypes.outofsample,
        calculation_types,
    )

    return insample_summary, outsample_summary
