from __future__ import annotations

import math
from copy import deepcopy
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

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


def zscore(arr: np.ndarray) -> np.ndarray:
    m = arr.mean()
    s = arr.std(ddof=0)
    return (arr - m) / s


def calculate_benchmark(
    metric: Metric,
    models: List[Model],
    y: TargetsDS,
    predictions: PredictionsDS,
    probabilities: Optional[ProbabilitiesDF],
    sample_weight: Optional[WeightsDS],
    num_benchmarking_iterations: int = 100,
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

        benchmark_metrics = [
            benchmark_metric._evaluation(
                y, preds_or_probs, sample_weight=sample_weight
            ).result
            for _ in range(num_benchmarking_iterations)
        ]

        mean_benchmark_metric = np.mean(benchmark_metrics)

        assert isinstance(mean_benchmark_metric, (float, int))
        assert isinstance(metric.result, (float, int))

        if metric.purpose == Purpose.objective:
            comparison_result = metric.result - mean_benchmark_metric
        elif metric.purpose == Purpose.loss:
            comparison_result = mean_benchmark_metric - metric.result

        benchmark_zscores = zscore(np.array(benchmark_metrics + [metric.result]))
        metric_zscore = benchmark_zscores[-1]

        metric.comparison_result = pd.concat(
            [
                metric.comparison_result,
                pd.Series([comparison_result], index=[f"Δ {model.name}"]),
                pd.Series([metric_zscore], index=[f"Δ {model.name}_zscore"]),
            ],
            axis=0,
        )

    return metric
