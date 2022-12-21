import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_squared_log_error,
)

from krisi.evaluate.library.metric_functions import ljung_box
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories, SampleTypes

predefined_default_metrics = [
    Metric[float](
        name="Mean Absolute Error",
        key="mae",
        category=MetricCategories.reg_err,
        func=mean_absolute_error,
    ),
    Metric[float](
        name="Mean Squared Error",
        key="mse",
        category=MetricCategories.reg_err,
        parameters={"squared": True},
        func=mean_squared_error,
    ),
    Metric[float](
        name="Root Mean Squared Error",
        key="rmse",
        category=MetricCategories.reg_err,
        parameters={"squared": False},
        func=mean_squared_error,
    ),
    Metric[float](
        name="Root Mean Squared Log Error",
        key="rmsle",
        category=MetricCategories.reg_err,
        parameters={"squared": False},
        func=mean_squared_log_error,
    ),
    Metric[float](
        name="Mean of the Residuals",
        key="residuals_mean",
        category=MetricCategories.residual,
        func=lambda y, pred: (y - pred).mean(),
    ),
    Metric[float](
        name="Standard Deviation of the Residuals",
        key="residuals_std",
        category=MetricCategories.residual,
        func=lambda y, pred: (y - pred).std(),
    ),
    Metric[pd.DataFrame](
        name="Ljung Box Statistics",
        key="ljung_box_statistics",
        category=MetricCategories.residual,
        func=ljung_box,
        info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
        restrict_to_sample=SampleTypes.insample,
    ),
]
