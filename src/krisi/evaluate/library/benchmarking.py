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
        benchmark_predictions, benchmark_probabilities = model.predict(y, sample_weight)

        for metric in all_metrics:
            if metric.purpose == Purpose.group or metric.purpose == Purpose.diagram:
                continue
            benchmark_metric = deepcopy(metric)
            benchmark_metric.result = None
            benchmark_metric.result_rolling = None

            if benchmark_metric.accepts_probabilities:
                benchmark_metric.evaluate(y, benchmark_probabilities)
            else:
                benchmark_metric.evaluate(y, benchmark_predictions)

            benchmark_result = (
                benchmark_metric.result
                if isinstance(benchmark_metric.result, (float, int))
                else None
            )
            model_result = (
                metric.result if isinstance(metric.result, (float, int)) else None
            )
            if benchmark_result is not None and model_result is not None:
                if metric.purpose == Purpose.objective:
                    comparison_result = benchmark_result - model_result
                elif metric.purpose == Purpose.loss:
                    comparison_result = model_result - benchmark_result
            else:
                comparison_result = "Either benchmark or model result is None."
            metric.comparison_result = pd.concat(
                [pd.Series([comparison_result], index=["Δ NS"])], axis=0
            )
        return all_metrics

    return postporcess_func
