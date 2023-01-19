import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.evaluate.metric import Metric
from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import CalculationTypes, Predictions, SampleTypes, Targets


def score(
    y: Targets,
    predictions: Predictions,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    project_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    classification: Optional[bool] = None,
    sample_type: SampleTypes = SampleTypes.outofsample,
    calculation_types: List[Union[CalculationTypes, str]] = [
        CalculationTypes.single,
        CalculationTypes.rolling,
    ],
    window: int = 30,
) -> ScoreCard:

    sc = ScoreCard(
        y=y,
        predictions=predictions,
        model_name=model_name,
        dataset_name=dataset_name,
        project_name=project_name,
        sample_type=sample_type,
        classification=classification,
        custom_metrics=custom_metrics,
    )

    for calculation_type in calculation_types:
        if (
            calculation_type == CalculationTypes.single
            or calculation_type == CalculationTypes.single.value
        ):
            sc.evaluate()
        if (
            calculation_type == CalculationTypes.rolling
            or calculation_type == CalculationTypes.rolling.value
        ):
            sc.evaluate_over_time(window=window)

    return sc


def evaluate_in_outsample(
    y_insample: pd.Series,
    insample_predictions: pd.Series,
    y_outsample: pd.Series,
    outsample_predictions: pd.Series,
    model_name: str = "Unknown model",
    dataset_name: Optional[str] = None,
    project_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    classification: Optional[bool] = None,
    calculation_types: List[Union[CalculationTypes, str]] = [
        CalculationTypes.single,
        CalculationTypes.rolling,
    ],
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = score(
        y_insample,
        insample_predictions,
        model_name,
        dataset_name,
        project_name,
        custom_metrics,
        classification,
        SampleTypes.insample,
        calculation_types,
    )
    outsample_summary = score(
        y_outsample,
        outsample_predictions,
        model_name,
        dataset_name,
        project_name,
        custom_metrics,
        classification,
        SampleTypes.outofsample,
        calculation_types,
    )

    return insample_summary, outsample_summary
