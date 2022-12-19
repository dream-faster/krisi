from enum import Enum
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

Predictions = Union[np.ndarray, pd.Series]
Targets = Union[np.ndarray, pd.Series]
MetricFunction = Callable[..., Any]


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"
