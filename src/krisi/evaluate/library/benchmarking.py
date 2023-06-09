from copy import deepcopy
from random import choices
from typing import Callable, List, Optional, Union

import numpy as np
import pandas as pd

from krisi.evaluate.metric import Metric
from krisi.evaluate.type import (
    PredictionsDS,
    ProbabilitiesDF,
    Purpose,
    TargetsDS,
    WeightsDS,
)


class RandomClassifier:
    def __init__(
        self, all_classes: List[int], probability_mean: Optional[List[float]] = None
    ) -> None:
        self.all_classes = all_classes
        if probability_mean is not None:
            assert len(probability_mean) == len(all_classes)
        self.probability_mean = (
            [1 / len(all_classes) for _ in range(len(all_classes))]
            if probability_mean is None
            else probability_mean
        )
        self.name = f"RandomClassifier_{all_classes}_{probability_mean}"

    def predict(self, X: pd.Series) -> Union[pd.Series, pd.DataFrame]:
        predictions = pd.Series(
            choices(population=self.all_classes, k=len(X)),
            index=X.index,
            name="predictions_RandomClassifier",
        )
        probabilities = pd.concat(
            [
                pd.Series(
                    np.random.normal(prob_mean, 0.1, len(X)).clip(0, 1),
                    index=X.index,
                    name=f"probabilities_RandomClassifier_{associated_class}",
                )
                for associated_class, prob_mean in zip(
                    self.all_classes, self.probability_mean
                )
            ],
            axis="columns",
        )
        probabilities = probabilities.div(probabilities.sum(axis=1), axis=0)

        return pd.concat([predictions, probabilities], axis="columns")


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
            return all_metrics
        df = model.predict(y)
        predictions = df.iloc[:, 0]
        probabilities = df.iloc[:, 1:]

        new_metrics = []
        for metric in all_metrics:
            if metric.purpose == Purpose.objective or Purpose.diagram:
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
        return all_metrics + new_metrics

    return postporcess_func
