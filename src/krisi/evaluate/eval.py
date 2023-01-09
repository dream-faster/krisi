import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .metric import Metric
from .scorecard import ScoreCard
from .type import CalculationTypes, Predictions, SampleTypes, Targets


def is_dataset_classification_like(y: Targets):
    # TODO: Should work with booleans and also check for arrays of whole floats, eg.: [1.0,2.0,3.0]
    # TODO: Create multilabel heuristic to be passed on to ScoreCard
    if isinstance(y, pd.Series):
        return pd.api.types.is_integer_dtype(y)
    elif isinstance(y, np.ndarray):
        return all([i.is_integer() for i in y if i is not None])
    else:
        return all([isinstance(i, int) for i in y if i is not None])


def eval(
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

    if dataset_name is None:
        if isinstance(y, pd.Series) and y.name is not None:
            dataset_name = y.name
        else:
            dataset_name = f"Dataset:{str(uuid.uuid4()).split('-')[0]}"

    if model_name is None:
        model_name = f"Model:{str(uuid.uuid4()).split('-')[0]}"

    if project_name is None:
        project_name = f"Project:{str(uuid.uuid4()).split('-')[0]}"

    if classification is None:
        classification = is_dataset_classification_like(y)

    sc = ScoreCard(
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
    project_name: Optional[str] = None,
    custom_metrics: List[Metric] = [],
    classification: Optional[bool] = None,
    calculation_types: List[Union[CalculationTypes, str]] = [
        CalculationTypes.single,
        CalculationTypes.rolling,
    ],
) -> Tuple[ScoreCard, ScoreCard]:

    insample_summary = eval(
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
    outsample_summary = eval(
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
