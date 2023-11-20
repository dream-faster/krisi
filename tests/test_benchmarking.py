import numpy as np
import pandas as pd

from krisi.evaluate import score
from krisi.evaluate.benchmark import zscore
from krisi.evaluate.library.benchmarking_models import (
    PerfectModel,
    RandomClassifier,
    RandomClassifierChunked,
    WorstModel,
)
from krisi.evaluate.library.default_metrics_classification import (
    binary_classification_balanced_metrics,
    f_one_score_macro,
)
from krisi.sharedtypes import Task
from krisi.utils.data import (
    generate_synthetic_data,
    generate_synthetic_predictions_binary,
)


def test_benchmarking_random():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    predictions = preds_probs.iloc[:, 0]
    probabilities = preds_probs.iloc[:, 1:3]

    sc = score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=[f_one_score_macro],
        benchmark_models=RandomClassifier(),
    )
    sc.print()


def test_benchmarking_random_chunked():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)

    sample_weight = pd.Series([1.0] * len(y))
    X = generate_synthetic_predictions_binary(y, sample_weight)

    chunk_size = 2
    preds_probs = RandomClassifierChunked(chunk_size).predict(
        X, y, sample_weight=sample_weight
    )

    original_rolling = (
        (
            X.iloc[:, 0]
            .rolling(window=chunk_size, step=chunk_size, min_periods=chunk_size)
            .mean()
        )
        .sort_values()
        .reset_index(drop=True)
    )
    predictions_rolling = (
        (
            preds_probs[0]
            .rolling(window=chunk_size, step=chunk_size, min_periods=chunk_size)
            .mean()
        )
        .sort_values()
        .reset_index(drop=True)
    )

    assert (original_rolling == predictions_rolling).sum() > (len(y) / chunk_size) * 0.9


def test_benchmarking_random_all_metrics():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    predictions = preds_probs.iloc[:, 0]
    probabilities = preds_probs.iloc[:, 1:3]

    sc = score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=binary_classification_balanced_metrics,
        benchmark_models=RandomClassifierChunked(2),
    )
    sc.print()


def test_perfect_to_best():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    predictions = preds_probs.iloc[:, 0]
    probabilities = preds_probs.iloc[:, 1:3]

    sc = score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=binary_classification_balanced_metrics,
        benchmark_models=[PerfectModel(), WorstModel()],
    )
    sc.print()

    for metric in sc.get_all_metrics():
        if metric.result is not None and metric.comparison_result is not None:
            assert metric.comparison_result["Δ PM"] < metric.comparison_result["Δ WM"]


def test_zscore_function():
    scores = zscore(np.array([1, 2, 3, 4, 5]))

    assert scores[0] == -1.414213562373095


def test_benchmark_zscore():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)
    sample_weight = pd.Series([1.0] * len(y))
    preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
    predictions = preds_probs.iloc[:, 0]
    probabilities = preds_probs.iloc[:, 1:3]

    sc = score(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=binary_classification_balanced_metrics,
        benchmark_models=[PerfectModel(), WorstModel()],
    )
    sc.print()

    for metric in sc.get_all_metrics():
        if metric.result is not None and metric.comparison_result is not None:
            assert metric.comparison_result["Δ PM"] < metric.comparison_result["Δ WM"]
