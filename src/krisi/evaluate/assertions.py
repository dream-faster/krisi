from typing import Any, Iterable, List, Tuple, Union, get_args, get_origin

import numpy as np
import pandas as pd

from krisi.utils.iterable_helpers import isiterable

from .type import Predictions, Targets

valid_types = ["int64", "float64", "float32", "float", "int"]


def get_bad_types(iterable: Union[Targets, Predictions]) -> Iterable:
    return [x for x in iterable if type(x) not in valid_types]


def flatten(xs):
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def flatten_type(complex_type: Union[Targets, Predictions]) -> Union[List, Any]:
    types = get_args(complex_type)
    if len(types) == 0:
        return complex_type
    else:
        add_list = []
        if get_origin(complex_type) == type([]):
            add_list = [type([])]
        return add_list + [flatten_type(type_) for type_ in get_args(complex_type)]


def unpack_type(complex_type: Union[Targets, Predictions]) -> Tuple[Any]:
    return tuple(flatten(flatten_type(complex_type)))


def check_valid_pred_target(y: Targets, predictions: Predictions):
    assert len(y) == len(
        predictions
    ), f"Target length {len(y)} should match predictions length {len(predictions)}"

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


def is_dataset_classification_like(y: Targets):
    # TODO: Should work with booleans and also check for arrays of whole floats, eg.: [1.0,2.0,3.0]
    # TODO: Create multilabel heuristic to be passed on to ScoreCard
    if isinstance(y, pd.Series):
        return pd.api.types.is_integer_dtype(y)
    elif isinstance(y, np.ndarray):
        return all([i.is_integer() for i in y if i is not None])
    else:
        return all([isinstance(i, int) for i in y if i is not None])
