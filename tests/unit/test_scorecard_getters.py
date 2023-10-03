import numpy as np
import pandas as pd
import pytest

from krisi.evaluate import score
from krisi.evaluate.group import Group
from krisi.evaluate.library.benchmarking import RandomClassifier, model_benchmarking
from krisi.evaluate.library.default_metrics_classification import f_one_score_macro


def test_spreading_comparions_results():
    groupped_metric = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=[f_one_score_macro],
        postprocess_funcs=[model_benchmarking(RandomClassifier())],
    )
    sc = score(
        pd.Series(np.random.randint(100)),
        pd.Series(np.random.randint(100)),
        default_metrics=groupped_metric,
    )
    sc.evaluate()
    metrics = sc.get_all_metrics(spread_comparisons=True)

    assert "f_one_score_macro_benchmarking-Δ NS" in [metric.key for metric in metrics]


def test_getting_no_skill_metric():
    groupped_metric = Group[pd.Series](
        name="benchmarking",
        key="benchmarking",
        metrics=[f_one_score_macro],
        postprocess_funcs=[model_benchmarking(RandomClassifier())],
    )
    sc = score(
        pd.Series(np.random.randint(100)),
        pd.Series(np.random.randint(100)),
        default_metrics=groupped_metric,
    )
    sc.evaluate()
    metric = sc["f_one_score_macro_benchmarking-Δ NS"]

    assert "f_one_score_macro_benchmarking-Δ NS" == metric.key

    with pytest.raises(AssertionError):
        metric = sc["f_one_score_macro_benchmarking-Δ NS-too-many-dashes"]
