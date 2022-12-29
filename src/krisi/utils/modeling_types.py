from typing import Union

import numpy as np
import pandas as pd

ModelOverTime = pd.Series
InSamplePredictions = Union[np.ndarray, pd.Series]
OutSamplePredictions = Union[np.ndarray, pd.Series]
y = Union[np.ndarray, pd.Series]
X = Union[np.ndarray, pd.DataFrame]
