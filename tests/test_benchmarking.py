import pandas as pd

from krisi.evaluate import score
from krisi.evaluate.group import Group
from krisi.evaluate.library.benchmarking import (
    PerfectModel,
    RandomClassifier,
    RandomClassifierChunked,
    WorstModel,
    model_benchmarking,
)
from krisi.evaluate.library.default_metrics_classification import (
    binary_classification_balanced_metrics,
    binary_classification_metrics_balanced_benchmarking,
    f_one_score_macro,
)
from krisi.sharedtypes import Task
from krisi.utils.data import (
    generate_synthetic_data,
    generate_synthetic_predictions_binary,
)


def test_benchmarking_random():
    groupped_metric = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=[f_one_score_macro],
        postprocess_funcs=[model_benchmarking(RandomClassifier())],
    )
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
        default_metrics=[groupped_metric],
    )
    sc.print()


def test_benchmarking_random_chunked():
    X, y = generate_synthetic_data(task=Task.classification, num_obs=1000)

    sample_weight = pd.Series([1.0] * len(y))
    X = generate_synthetic_predictions_binary(y, sample_weight)

    predictions = pd.concat(
        RandomClassifierChunked(5).predict(X, y, sample_weight=sample_weight),
        axis="columns",
        copy=False,
    )

    original_rolling = (
        (X.iloc[:, 0].rolling(window=5, step=5, min_periods=5).mean())
        .sort_values()
        .reset_index(drop=True)
    )
    predictions_rolling = (
        (predictions.iloc[:, 0].rolling(window=5, step=5, min_periods=5).mean())
        .sort_values()
        .reset_index(drop=True)
    )

    assert (original_rolling == predictions_rolling).sum() > len(y) / 5 * 0.9


def test_benchmarking_random_all_metrics():
    groupped_metric = binary_classification_metrics_balanced_benchmarking
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
        default_metrics=groupped_metric,
    )
    sc.print()


def test_perfect_to_best():
    benchmark = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=binary_classification_balanced_metrics,
        postprocess_funcs=[
            model_benchmarking(PerfectModel()),
            model_benchmarking(WorstModel()),
        ],
    )
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
        default_metrics=[benchmark],
    )
    sc.print()

    for metric in sc.get_all_metrics():
        if metric.result is not None and metric.comparison_result is not None:
            assert metric.comparison_result["Δ PM"] < metric.comparison_result["Δ WM"]
