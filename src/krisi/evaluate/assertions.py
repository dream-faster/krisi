from typing import Iterable, Union

import numpy as np
import pandas as pd

from krisi.utils.iterable_helpers import isiterable

from .type import Predictions, Targets

valid_types = [
    "int64",
    "float64",
    "float32",
]


def get_bad_types(iterable: Union[Targets, Predictions]) -> Iterable:
    return [x for x in iterable if type(x) not in valid_types]


def check_valid_pred_target(y: Targets, predictions: Predictions):
    assert len(y) == len(
        predictions
    ), f"Target length {len(y)} should match predictions length {len(predictions)}"

    for iterable in [y, predictions]:
        assert isiterable(iterable), f"{iterable} is not an iterable or it is a string."
        dtype_name = ""
        if isinstance(iterable, pd.Series):
            dtype_name = iterable.dtype
        elif isinstance(iterable, np.ndarray):
            dtype_name = iterable.dtype.name

        assert (
            dtype_name in valid_types
        ), f"{iterable} contains at least one invalid type. {get_bad_types(iterable)}"


def is_dataset_classification_like(y: Targets):
    # TODO: Should work with booleans and also check for arrays of whole floats, eg.: [1.0,2.0,3.0]
    # TODO: Create multilabel heuristic to be passed on to ScoreCard
    if isinstance(y, pd.Series):
        return pd.api.types.is_integer_dtype(y)
    elif isinstance(y, np.ndarray):
        return all([i.is_integer() for i in y if i is not None])
    else:
        return all([isinstance(i, int) for i in y if i is not None])
