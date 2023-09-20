from typing import List

import numpy as np
import pandas as pd

from krisi import Metric, score


def get_metric_dict(metrics: List[Metric]) -> dict:
    return {metric.key: metric for metric in metrics}


def test_probabilities():
    # default_metrics = get_metric_dict(library.all_classification_metrics)["brier_score"]
    df = pd.read_csv("tests/sample_predictions_binary.csv", index_col=0)
    y = pd.read_csv("tests/sample_y_classification.csv", index_col=0).squeeze()

    sc = score(
        y[df.index],
        df["predictions_XGBClassifier"],
        df[["probabilities_XGBClassifier_0", "probabilities_XGBClassifier_1"]],
        # default_metrics=default_metrics,
    )

    assert np.isclose(sc["brier_score"].value, 0.37293859755314235, atol=0.1)
