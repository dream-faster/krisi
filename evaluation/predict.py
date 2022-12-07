from typing import Tuple, Union
import pandas as pd
import numpy as np

from models.base import Model


def generate_univariate_predictions(
    model: Model,
    df: pd.DataFrame,
    target_col: str,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:

    model.fit(df[[target_col]], df[target_col])
    outsample_prediction = model.predict(df[[target_col]])
    insample_prediction = model.predict_in_sample(df[[target_col]])

    return insample_prediction, outsample_prediction
