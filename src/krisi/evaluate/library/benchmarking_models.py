from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.utils.data import generate_synthetic_predictions_binary, shuffle_df_in_chunks

from ..type import Model, WeightsDS


class RandomClassifier(Model):
    def __init__(self) -> None:
        self.name = "NS"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[WeightsDS] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        preds_probs = generate_synthetic_predictions_binary(y, sample_weight)
        predictions = preds_probs.iloc[:, 0]
        probabilities = preds_probs.iloc[:, 1:3]
        return predictions, probabilities


class RandomClassifierSmoothed(Model):
    def __init__(self, smoothing_window: Optional[int] = None) -> None:
        self.name = "NS-Smooth"
        self.smoothing_window = smoothing_window

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: Optional[WeightsDS] = None
    ) -> Tuple[pd.Series, pd.DataFrame]:
        def get_avarage_change_length(ds: pd.Series) -> float:
            # This only works for binary classification
            indices = ds.diff().fillna(0).ne(0).cumsum()
            differences = indices.groupby(indices).size()
            return differences.mean()

        smoothing_window = (
            math.floor(get_avarage_change_length(y))
            if self.smoothing_window is None
            else self.smoothing_window
        )

        preds_probs = generate_synthetic_predictions_binary(
            y, sample_weight, smoothing_window=smoothing_window
        )
        predictions = preds_probs.iloc[:, 0]
        probabilities = preds_probs.iloc[:, 1:3]
        return predictions, probabilities


class RandomClassifierChunked(Model):
    def __init__(self, chunk_size: Union[float, int]) -> None:
        self.name = "NS"
        self.chunk_size = chunk_size

    def predict(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: Optional[WeightsDS] = None,
    ) -> Tuple[pd.Series, pd.DataFrame]:
        X = shuffle_df_in_chunks(X, self.chunk_size)
        predictions = X.iloc[:, 0]
        probabilities = X.iloc[:, 1:3]
        return predictions, probabilities


class PerfectModel(Model):
    def __init__(self) -> None:
        self.name = "PM"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: WeightsDS
    ) -> Tuple[pd.Series, pd.DataFrame]:
        predictions = y
        probabilities = pd.get_dummies(y, dtype=float)
        return predictions, probabilities


class WorstModel(Model):
    def __init__(self) -> None:
        self.name = "WM"

    def predict(
        self, X: pd.DataFrame, y: pd.Series, sample_weight: WeightsDS
    ) -> Tuple[pd.Series, pd.DataFrame]:
        predictions = np.logical_xor(y, 1).astype(int)
        probabilities = pd.get_dummies(predictions, dtype=float)
        return predictions, probabilities


benchmark_models = [
    RandomClassifier,
    RandomClassifierSmoothed,
    RandomClassifierChunked,
    PerfectModel,
    WorstModel,
]
