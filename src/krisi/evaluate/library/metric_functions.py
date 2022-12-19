from typing import Any, Tuple, Union

import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from krisi.evaluate.type import Predictions, Targets


def ljung_box(
    y: Targets, predictions: Predictions
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    return acorr_ljungbox(predictions)
