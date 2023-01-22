import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.evaluate.metric import Metric
from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.type import Calculation, Predictions, SampleTypes, Targets


def score(
    y: Targets,
    predictions: Predictions,
    model_name: Optional[str] = None,
    dataset_name: Optional[str] = None,
    project_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    classification: Optional[bool] = None,
    sample_type: SampleTypes = SampleTypes.outofsample,
    calculation: Union[Calculation, str] = Calculation.single,
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

    if calculation == Calculation.single or calculation == Calculation.single.value:
        sc.evaluate()
    elif calculation == Calculation.rolling or calculation == Calculation.rolling.value:
        sc.evaluate_over_time(window=window)
    elif calculation == Calculation.both or calculation == Calculation.both.value:
        sc.evaluate()
        sc.evaluate_over_time(window=window)
    else:
        raise ValueError(f"Calculation type {calculation} not recognized.")

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
    calculation: Union[Calculation, str] = Calculation.single,
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
        calculation,
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
        calculation,
    )

    return insample_summary, outsample_summary
