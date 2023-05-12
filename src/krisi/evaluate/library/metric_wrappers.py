from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox

from krisi.evaluate.type import Predictions, ProbabilitiesDS, TargetsDS


def ljung_box(
    y: TargetsDS, predictions: Predictions
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    return acorr_ljungbox(predictions)


def create_one_hot(ds: Union[TargetsDS, ProbabilitiesDS]) -> pd.DataFrame:
    return pd.get_dummies(ds)


def brier_multi(targets: TargetsDS, probs: ProbabilitiesDS, **kwargs):
    # See https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    return np.mean(np.sum((probs - create_one_hot(targets)) ** 2, axis=1))
