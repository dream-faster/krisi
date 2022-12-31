from typing import Iterable, List, Tuple, Union

import numpy as np
import pandas as pd

from krisi.evaluate.type import Predictions, Targets
from krisi.utils.iterable_helpers import isiterable

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
