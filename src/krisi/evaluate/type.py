from enum import Enum
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go

MetricResult = TypeVar(
    "MetricResult", bound=Union[float, int, str, List, Tuple, pd.DataFrame]
)


Predictions = Union[np.ndarray, pd.Series]
Targets = Union[np.ndarray, pd.Series]
MetricFunction = Callable[[Predictions, Targets], MetricResult]


class MetricCategories(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    unknown = "Unknown"


class SampleTypes(Enum):
    insample = "insample"
    outofsample = "outofsample"


class CalculationTypes(Enum):
    single = "single"
    rolling = "rolling"


class SaveModes(Enum):
    svg = "svg"
    html = "html"
    text = "text"
    obj = "obj"
    minimal = "minimal"
