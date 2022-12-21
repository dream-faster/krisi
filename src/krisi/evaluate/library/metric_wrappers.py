from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox

from krisi.evaluate.type import Predictions, Targets


def ljung_box(
    y: Targets, predictions: Predictions
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    return acorr_ljungbox(predictions)
