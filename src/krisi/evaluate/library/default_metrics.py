import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from krisi.evaluate.library.diagrams import display_time_series
from krisi.evaluate.library.metric_wrappers import ljung_box
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories, SampleTypes

predefined_default_metrics = [
    Metric[float](
        name="Mean Absolute Error",
        key="mae",
        category=MetricCategories.reg_err,
        info="(Mean absolute error) represents the difference between the original and predicted values extracted by averaged the absolute difference over the data set.",
        func=mean_absolute_error,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Mean Absolute Percentage Error",
        key="mape",
        category=MetricCategories.reg_err,
        func=mean_absolute_percentage_error,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Mean Squared Error",
        key="mse",
        category=MetricCategories.reg_err,
        info="(Mean Squared Error) represents the difference between the original and predicted values extracted by squared the average difference over the data set.",
        parameters={"squared": True},
        func=mean_squared_error,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Root Mean Squared Error",
        key="rmse",
        category=MetricCategories.reg_err,
        info="(Root Mean Squared Error) is the error rate by the square root of Mean Squared Error.",
        parameters={"squared": False},
        func=mean_squared_error,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Root Mean Squared Log Error",
        key="rmsle",
        category=MetricCategories.reg_err,
        parameters={"squared": False},
        func=mean_squared_log_error,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="R-squared",
        key="r2",
        category=MetricCategories.reg_err,
        info="(Coefficient of determination) represents the coefficient of how well the values fit compared to the original values. The value from 0 to 1 interpreted as percentages. The higher the value is, the better the model is.",
        func=r2_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Mean of the Residuals",
        key="residuals_mean",
        category=MetricCategories.residual,
        func=lambda y, pred: (y - pred).mean(),
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Standard Deviation of the Residuals",
        key="residuals_std",
        category=MetricCategories.residual,
        func=lambda y, pred: (y - pred).std(),
        plot_func=display_time_series,
    ),
    Metric[pd.DataFrame](
        name="Ljung Box Statistics",
        key="ljung_box_statistics",
        category=MetricCategories.residual,
        func=ljung_box,
        info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
        restrict_to_sample=SampleTypes.insample,
        # plot_func=display_time_series
    ),
]
