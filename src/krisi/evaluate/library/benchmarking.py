from copy import deepcopy
from typing import Callable, List, Optional, Union

import pandas as pd

from krisi.evaluate.metric import Metric
from krisi.evaluate.type import (
    PredictionsDS,
    ProbabilitiesDF,
    Purpose,
    TargetsDS,
    WeightsDS,
)
from krisi.utils.data import generate_synthetic_predictions_binary


class RandomClassifier:
    def __init__(self) -> None:
        self.name = "RandomClassifier"

    def predict(
        self, y: pd.Series, sample_weight: WeightsDS
    ) -> Union[pd.Series, pd.DataFrame]:
        preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
        predictions = preds_probs.iloc[:, 0]
        probabilities = preds_probs.iloc[:, 1:3]
        return predictions, probabilities


def model_benchmarking(model: RandomClassifier) -> Callable:
    def postporcess_func(
        all_metrics: List[Metric],
        y: TargetsDS,
        predictions: Optional[PredictionsDS],
        probabilities: Optional[ProbabilitiesDF],
        sample_weight: Optional[WeightsDS],
        rolling: bool,
    ) -> List[Metric]:
        if rolling:
            return []
        predictions, probabilities = model.predict(y, sample_weight)

        new_metrics = []
        for metric in all_metrics:
            if metric.purpose == Purpose.group or metric.purpose == Purpose.diagram:
                continue
            copy_of_metric = deepcopy(metric)
            copy_of_metric.result = None
            copy_of_metric.result_rolling = None
            copy_of_metric.key += f"_{metric.purpose.value}_{model.name}"
            copy_of_metric.name += f" {metric.purpose.value} with {model.name}"
            if copy_of_metric.accepts_probabilities:
                copy_of_metric.evaluate(y, probabilities)
            else:
                copy_of_metric.evaluate(y, predictions)

            if metric.purpose == Purpose.objective:
                copy_of_metric.result = copy_of_metric.result - metric.result
            elif metric.purpose == Purpose.loss:
                copy_of_metric.result = metric.result - copy_of_metric.result
            new_metrics.append(copy_of_metric)
        return new_metrics

    return postporcess_func
