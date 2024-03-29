import numpy as np
import pandas as pd
import pytest

from krisi import library
from krisi.evaluate import score


def test_spreading_comparions_results():
    sc = score(
        pd.Series(np.random.randint(2, size=100)),
        pd.Series(np.random.randint(2, size=100)),
        default_metrics=[library.ClassificationRegistry().f_one_score_macro],
        benchmark_models=library.ModelRegistry.RandomClassifierChunked(0.05),
    )

    metrics = sc.get_all_metrics(spread_comparisons=True)

    assert "f_one_score_macro-Δ NS" in [metric.key for metric in metrics]


def test_getting_no_skill_metric():
    sc = score(
        pd.Series(np.random.randint(2, size=100)),
        pd.Series(np.random.randint(2, size=100)),
        default_metrics=[library.ClassificationRegistry().f_one_score_macro],
        benchmark_models=library.ModelRegistry.RandomClassifierChunked(0.05),
    )

    metric = sc["f_one_score_macro-Δ NS"]

    assert "f_one_score_macro-Δ NS" == metric.key

    with pytest.raises(AssertionError):
        metric = sc["f_one_score_macro-Δ NS-too-many-dashes"]
