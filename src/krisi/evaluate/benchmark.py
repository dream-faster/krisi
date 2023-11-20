from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import numpy as np
import pandas as pd

from krisi.evaluate.type import (
    Model,
    PredictionsDS,
    ProbabilitiesDF,
    Purpose,
    TargetsDS,
    WeightsDS,
)

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric


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
    num_benchmark_iter: Optional[int],
) -> Metric:
    metric_result = (
        metric.evaluate(y, predictions, probabilities, sample_weight).result
        if metric.result is None
        else metric.result
    )
    benchmark_metric = metric.reset()

    for model in models:
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
            for _ in range(num_benchmark_iter)
        ]

        mean_benchmark_metric = np.mean(benchmark_metrics)

        assert isinstance(mean_benchmark_metric, (float, int))
        assert isinstance(metric_result, (float, int))

        if metric.purpose == Purpose.objective:
            comparison_result = metric_result - mean_benchmark_metric
        elif metric.purpose == Purpose.loss:
            comparison_result = mean_benchmark_metric - metric_result

        benchmark_zscores = zscore(np.array(benchmark_metrics + [metric_result]))
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
