from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.naive import NaiveForecaster
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults

from all_types import InSamplePredictions, OutSamplePredictions, X, y


class Model(ABC):

    name: str = ""

    @abstractmethod
    def fit(self, X: X, y: y) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: X) -> OutSamplePredictions:
        raise NotImplementedError

    @abstractmethod
    def predict_in_sample(self, X: X) -> InSamplePredictions:
        raise NotImplementedError


 
 



def generate_univariate_predictions(
    model: Model,
    df: pd.DataFrame, 
    target_col: str,
) -> Tuple[Union[pd.Series, np.ndarray], Union[pd.Series, np.ndarray]]:

    model.fit(df[[target_col]], df[target_col]) 
    outsample_prediction = model.predict(df[[target_col]])
    insample_prediction = model.predict_in_sample(df[[target_col]])

    return insample_prediction, outsample_prediction


class StrategyTypes(Enum):
    mean = "mean"
    last = "last"
    drift = "drift"


@dataclass
class NaiveForecasterConfig:
    strategy: StrategyTypes
    fh: Optional[Union[int, List[int]]] = None  # Forecasting Horizon
    window_length: Optional[int] = None  # The window of the mean if strategy is 'mean'
    sp: int = 1  # Seasonal periodicity


class NaiveForecasterWrapper(Model):

    name: str = ""
    strategy_types = StrategyTypes

    def __init__(self, config: NaiveForecasterConfig) -> None:
        self.model = NaiveForecaster(
            strategy=config.strategy.value,
            sp=config.sp,
            window_length=config.window_length,
        )
        self.config = config

    def fit(self, X: X, y: y) -> None:
        self.model.fit(X)

    def predict_in_sample(self, X: X) -> InSamplePredictions:
        fh = ForecastingHorizon([x + 1 for x in range(len(X))], is_relative=False)
        fh = fh.to_relative(cutoff=len(X))
        return self.model.predict(fh=fh)

    def predict(self, X: X) -> OutSamplePredictions:
        fh = (
            self.config.fh
            if self.config.fh is not None
            else ForecastingHorizon([x + 1 for x in range(len(X))], is_relative=True)
        )
        return self.model.predict(fh=fh)

@dataclass
class ArimaConfig:
    """
    order=(p,d,q).
        p is the autoregressive order
        d is the number of times the series has been differenced
        q is the moving average order
    """

    order: Tuple[float, float, float]  # Forecasting Horizon
    trend: Optional[Union[str, Tuple[float]]] = None


class ArimaWrapper(Model):

    name: str = ""

    def __init__(self, config: ArimaConfig) -> None:
        self.config = config

    def fit(self, X: X, y: y) -> None:
        unfitted_model = ARIMA(X, order=self.config.order, trend=self.config.trend)
        self.model = unfitted_model.fit()

    def predict_in_sample(self, X: X) -> InSamplePredictions:
        return self.model.predict(end=len(X) - 1)

    def predict(self, X: X) -> OutSamplePredictions:
        if self.model:
            return self.model.forecast(len(X))
        else:
            raise ValueError("Model has to be fitted first")

default_naive_model = NaiveForecasterWrapper(
    NaiveForecasterConfig(strategy=NaiveForecasterWrapper.strategy_types.last)
)
default_arima_model = ArimaWrapper(ArimaConfig(order=(1, 0, 1)))
