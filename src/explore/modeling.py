from typing import Optional, Tuple, List, Union
import pandas as pd

import itertools
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults


def fit_arima_model(
    ds: pd.Series,
    order: Tuple[float, float, float] = (1, 0, 0),
    trend: Optional[Union[str, Tuple[float]]] = None,
) -> ARIMAResults:
    """
    order=(p,d,q).
        p is the autoregressive order
        q is the moving average order
        d is the number of times the series has been differenced
    """
    model = ARIMA(ds, order=order, trend=trend)
    result = model.fit()
    return result


def fit_multiple_arimas(
    ds: pd.Series,
    order_ranges: Tuple[List[float], List[float], List[float]] = (
        [0, 1, 2],
        [0, 1],
        [0, 1, 2],
    ),
) -> List[ARIMAResults]:
    cartesian_product = list(itertools.product(*order_ranges))
    model_results = [fit_arima_model(ds, order) for order in cartesian_product]

    return sorted(model_results, key=lambda x: x.aic)
