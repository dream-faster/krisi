from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Union

from statsmodels.tsa.arima.model import ARIMA

from krisi.utils.modeling_types import InSamplePredictions, OutSamplePredictions, X, y


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


default_arima_model = ArimaWrapper(ArimaConfig(order=(1, 0, 1)))
