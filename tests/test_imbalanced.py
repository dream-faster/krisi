import numpy as np
import pandas as pd

from krisi.evaluate import library, score
from krisi.sharedtypes import Task
from krisi.utils.data import (
    generate_synthetic_data,
    generate_synthetic_predictions_binary,
)


def test_benchmarking_random_all_metrics():
    groupped_metric = library.binary_classification_imbalanced_metrics
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

    assert sc["imbalance_ratio_y"].result == 0.0
    assert np.isclose(sc["imbalance_ratio_pred_y"].result, 0.0, atol=0.1)
