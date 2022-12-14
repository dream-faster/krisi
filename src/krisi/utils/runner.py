from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.utils.models import Model


def fit_model(
    model: Model, train: pd.DataFrame, test: Optional[pd.Series] = None
) -> None:

    if test is None:
        test = pd.Series([])

    model.fit(train, test)


def generate_univariate_predictions(
    model: Model,
    train: pd.DataFrame,
    len_test: int,
    target_col: str,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:

    insample_prediction = model.predict_in_sample(train[[target_col]])
    outsample_prediction = model.predict(pd.DataFrame([x for x in range(len_test)]))

    return insample_prediction, outsample_prediction
