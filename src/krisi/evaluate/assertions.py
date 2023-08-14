from typing import Any, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from typing_extensions import get_args, get_origin

from krisi.evaluate.metric import Metric
from krisi.evaluate.type import DatasetType, Predictions, Targets, Weights
from krisi.utils.iterable_helpers import flatten, isiterable

valid_types = ["int64", "float64", "float32", "float", "int"]


def get_bad_types(iterable: Union[Targets, Predictions]) -> Iterable:
    return [x for x in iterable if type(x) not in valid_types]


def flatten_type(complex_type: Any) -> Union[List, Any]:
    types = get_args(complex_type)
    if len(types) == 0:
        return complex_type
    else:
        add_list = []
        if isinstance(get_origin(complex_type), type([])):
            add_list = [type([])]
        return add_list + [flatten_type(type_) for type_ in get_args(complex_type)]


def unpack_type(complex_type: Any) -> Tuple[Any]:
    return tuple(flatten(flatten_type(complex_type)))


def check_no_duplicate_metrics(metrics: List[Metric]):
    assert len(metrics) == len(
        set([metric.key for metric in metrics])
    ), f"Duplicate metric key found in {metrics}. Please remove the duplicates."


def check_valid_pred_target(
    y: Targets, predictions: Predictions, sample_weights: Optional[Weights]
):
    assert len(y) == len(
        predictions
    ), f"Target length {len(y)} should match predictions length {len(predictions)}"
    if sample_weights is not None:
        assert len(y) == len(
            sample_weights
        ), f"Target length {len(y)} should match sample_weights length {len(sample_weights)}"

    assert isinstance(
        y, unpack_type(Targets)
    ), f"Targets: {type(y)} are not of a valid type."
    assert isinstance(
        predictions, unpack_type(Predictions)
    ), f"Predictions: {type(predictions)} are not of a valid type."

    for iterable in [y, predictions]:
        assert isiterable(iterable), f"{iterable} is not an iterable or it is a string."
        dtype_name = ""
        if isinstance(iterable, pd.Series):
            dtype_name = iterable.dtype
        elif isinstance(iterable, np.ndarray):
            dtype_name = iterable.dtype.name
        else:
            types = set([type(obj) for obj in iterable])
            assert len(types) == 1, f"Inconsistent types in {iterable}"
            dtype_name = list(types)[0]

        assert (
            dtype_name in valid_types
        ), f"{iterable} contains at least one invalid type. {get_bad_types(iterable)}"


def infer_dataset_type(y: Targets) -> DatasetType:
    # TODO: Should work with booleans
    def infer_exact_type(is_classification: bool, y: Targets) -> DatasetType:
        if is_classification is False:
            return DatasetType.regression
        if len(set(y)) == 2:
            class_1 = y[0]
            no_of_class_1 = np.count_nonzero(y == class_1)
            ratio = no_of_class_1 / len(y)
            if ratio < 0.3 or ratio > 0.7:
                return DatasetType.classification_binary_imbalanced
            else:
                return DatasetType.classification_binary_balanced
        else:
            return DatasetType.classification_multiclass

    if isinstance(y, pd.Series):
        y = y.to_numpy()
    if isinstance(y, np.ndarray):
        if hasattr(y[0], "is_integer"):
            # Using this is faster, if available use this.
            return infer_exact_type(
                all([i.is_integer() for i in y if i is not None]), y
            )
        else:
            return infer_exact_type(np.all(np.mod(y, 1) == 0), y)
    else:
        return infer_exact_type(
            all([isinstance(i, int) for i in y if i is not None]), y
        )
