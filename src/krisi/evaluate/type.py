from enum import Enum
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

MResultGeneric = TypeVar("MResultGeneric", bound=Union[float, int, str, List, Tuple])


Predictions = Union[np.ndarray, pd.Series]
Targets = Union[np.ndarray, pd.Series]
MetricFunction = Callable[..., MResultGeneric]


class MetricCategories(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    unknown = "Unknown"


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


class CalculationTypes(Enum):
    single = "single"
    rolling = "rolling"
