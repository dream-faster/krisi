import pandas as pd

from krisi.evaluate import score
from krisi.evaluate.group import Group
from krisi.evaluate.library.benchmarking import RandomClassifier, model_benchmarking
from krisi.evaluate.library.default_metrics_classification import f_one_score_macro
from krisi.utils.data import (
    generate_synthetic_data,
    generate_synthetic_predictions_binary,
)
from krisi.utils.devutils.type import Task


def test_benchmarking_random():
    groupped_metric = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=[f_one_score_macro],
        postprocess_funcs=[model_benchmarking(RandomClassifier())],
    )
    X, y = generate_synthetic_data(task=Task.classification)
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
