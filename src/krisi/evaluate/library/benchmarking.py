from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, Callable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.evaluate.type import (
    PredictionsDS,
    ProbabilitiesDF,
    Purpose,
    TargetsDS,
    WeightsDS,
)
from krisi.utils.data import generate_synthetic_predictions_binary, shuffle_df_in_chunks

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric


class Model:
    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[WeightsDS] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        raise NotImplementedError


class RandomClassifier(Model):
    def __init__(self) -> None:
        self.name = "NS"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[WeightsDS] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
        predictions = preds_probs.iloc[:, 0]
        probabilities = preds_probs.iloc[:, 1:3]
        return predictions, probabilities


class RandomClassifierSmoothed(Model):
    def __init__(self, smoothing_window: Optional[int] = None) -> None:
        self.name = "NS-Smooth"
        self.smoothing_window = smoothing_window

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[WeightsDS] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        def get_avarage_change_length(ds: pd.Series) -> float:
            # This only works for binary classification
            indices = ds.diff().fillna(0).ne(0).cumsum()
            differences = indices.groupby(indices).size()
            return differences.mean()

        smoothing_window = (
            math.floor(get_avarage_change_length(y))
            if self.smoothing_window is None
            else self.smoothing_window
        )

        preds_probs = generate_synthetic_predictions_binary(
            y, sample_weight, smoothing_window=smoothing_window
        )
        predictions = preds_probs.iloc[:, 0]
        probabilities = preds_probs.iloc[:, 1:3]
        return predictions, probabilities


class RandomClassifierChunked(Model):
    def __init__(self, chunk_size: Union[float, int]) -> None:
        self.name = "NS"
        self.chunk_size = chunk_size

    def predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[WeightsDS] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        X = shuffle_df_in_chunks(X, self.chunk_size)
        predictions = X.iloc[:, 0]
        probabilities = X.iloc[:, 1:3]
        return predictions, probabilities


class PerfectModel(Model):
    def __init__(self) -> None:
        self.name = "PM"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: WeightsDS
    ) -> Tuple[pd.Series, pd.DataFrame]:
        predictions = y
        probabilities = pd.get_dummies(y, dtype=float)
        return predictions, probabilities


class WorstModel(Model):
    def __init__(self) -> None:
        self.name = "WM"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: WeightsDS
    ) -> Tuple[pd.Series, pd.DataFrame]:
        predictions = np.logical_xor(y, 1).astype(int)
        probabilities = pd.get_dummies(predictions, dtype=float)
        return predictions, probabilities


def calculate_benchmark(
    metric: Metric,
    models: List[Model],
    y: TargetsDS,
    predictions: PredictionsDS,
    probabilities: Optional[ProbabilitiesDF],
    sample_weight: Optional[WeightsDS],
) -> Metric:
    if metric.result is None:
        metric = metric.evaluate(y, predictions, probabilities, sample_weight)

    for model in models:
        benchmark_metric = deepcopy(metric.reset())
        if metric.purpose == Purpose.group or metric.purpose == Purpose.diagram:
            return metric

        if probabilities is not None:
            X = pd.concat([predictions, probabilities], axis="columns", copy=False)
        else:
            X = predictions.to_frame()
        benchmark_preds, benchmark_probs = model.predict(X, y, sample_weight)

        preds_or_probs = (
            benchmark_probs if metric.accepts_probabilities else benchmark_preds
        )

        benchmark_metric = benchmark_metric._evaluation(
            y, preds_or_probs, sample_weight=sample_weight
        )

        assert isinstance(benchmark_metric.result, (float, int))
        assert isinstance(metric.result, (float, int))

        if metric.purpose == Purpose.objective:
            comparison_result = metric.result - benchmark_metric.result
        elif metric.purpose == Purpose.loss:
            comparison_result = benchmark_metric.result - metric.result

        metric.comparison_result = pd.concat(
            [
                metric.comparison_result,
                pd.Series([comparison_result], index=[f"Δ {model.name}"]),
            ],
            axis=0,
        )

    return metric


def model_benchmarking(model: Model) -> Callable:
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
        benchmark_preds, benchmark_probs = model.predict(
            pd.concat([predictions, probabilities], axis="columns", copy=False),
            y,
            sample_weight,
        )

        for metric in all_metrics:
            if metric.purpose == Purpose.group or metric.purpose == Purpose.diagram:
                continue
            benchmark_metric = deepcopy(metric)
            benchmark_metric.result = None
            benchmark_metric.result_rolling = None

            if benchmark_metric.accepts_probabilities:
                benchmark_metric = benchmark_metric._evaluation(
                    y, benchmark_probs, sample_weight=sample_weight
                )
            else:
                benchmark_metric = benchmark_metric._evaluation(
                    y, benchmark_preds, sample_weight=sample_weight
                )

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
                    comparison_result = model_result - benchmark_result
                elif metric.purpose == Purpose.loss:
                    comparison_result = benchmark_result - model_result
            else:
                comparison_result = "Either benchmark or model result is None."
            metric.comparison_result = pd.concat(
                [
                    metric.comparison_result,
                    pd.Series([comparison_result], index=[f"Δ {model.name}"]),
                ],
                axis=0,
            )
        return all_metrics

    return postporcess_func


if __name__ == "__main__":
    RandomClassifierSmoothed().predict(
        pd.Series([0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1]),
    )
