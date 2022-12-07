import pandas as pd
import numpy as np
from typing import Union

ModelOverTime = pd.Series
InSamplePredictions = Union[np.ndarray, pd.Series]
OutSamplePredictions = Union[np.ndarray, pd.Series]
y = Union[np.ndarray, pd.Series]
X = Union[np.ndarray, pd.DataFrame]
