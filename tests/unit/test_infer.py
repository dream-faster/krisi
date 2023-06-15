import pandas as pd

from krisi.evaluate.assertions import infer_dataset_type
from krisi.evaluate.type import DatasetType


def test_recognize_multiclass_data():
    assert (
        infer_dataset_type(pd.Series([0, 1, 1, 2, 2, 0]))
        is DatasetType.classification_multiclass
    )
    assert (
        infer_dataset_type(pd.Series([2.0, 1.0, 1.0, 0.0]))
        is DatasetType.classification_multiclass
    )


def test_recognize_regression_data():
    assert infer_dataset_type(pd.Series([0.2, 0.3, 0.4, 0.5])) is DatasetType.regression


def test_recognize_binary_data():
    assert (
        infer_dataset_type(pd.Series([0, 1, 1, 0]))
        is DatasetType.classification_binary_balanced
    )
    assert (
        infer_dataset_type(pd.Series([0.0, 1.0, 1.0, 0.0]))
        is DatasetType.classification_binary_balanced
    )
