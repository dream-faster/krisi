from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_squared_log_error,
    r2_score,
)

from krisi.evaluate.library.diagrams import (
    display_acf_plot,
    display_density_plot,
    display_single_value,
    display_time_series,
)
from krisi.evaluate.library.metric_wrappers import ljung_box
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import ComputationalComplexity, MetricCategories, SampleTypes

mae = Metric[float](
    name="Mean Absolute Error",
    key="mae",
    category=MetricCategories.reg_err,
    info="(Mean absolute error) represents the difference between the original and predicted values extracted by averaged the absolute difference over the data set.",
    func=mean_absolute_error,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)

mape = Metric[float](
    name="Mean Absolute Percentage Error",
    key="mape",
    category=MetricCategories.reg_err,
    func=mean_absolute_percentage_error,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)

smape = Metric[float](
    name="Symmetric Mean Absolute Percentage Error",
    key="smape",
    category=MetricCategories.reg_err,
    func=lambda y_true, y_pred: np.mean(
        np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)), axis=0
    ),
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)

mse = Metric[float](
    name="Mean Squared Error",
    key="mse",
    category=MetricCategories.reg_err,
    info="(Mean Squared Error) represents the difference between the original and predicted values extracted by squared the average difference over the data set.",
    parameters={"squared": True},
    func=mean_squared_error,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
rmse = Metric[float](
    name="Root Mean Squared Error",
    key="rmse",
    category=MetricCategories.reg_err,
    info="(Root Mean Squared Error) is the error rate by the square root of Mean Squared Error.",
    parameters={"squared": False},
    func=mean_squared_error,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
rmsle = Metric[float](
    name="Root Mean Squared Log Error",
    key="rmsle",
    category=MetricCategories.reg_err,
    parameters={"squared": False},
    func=mean_squared_log_error,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
r_two = Metric[float](
    name="R-squared",
    key="r_two",
    category=MetricCategories.reg_err,
    info="(Coefficient of determination) represents the coefficient of how well the values fit compared to the original values. The value from 0 to 1 interpreted as percentages. The higher the value is, the better the model is.",
    func=r2_score,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
residuals = Metric[List[float]](
    name="Residuals",
    key="residuals",
    category=MetricCategories.residual,
    func=lambda y, pred: y - pred,
    plot_func_rolling=display_time_series,
    plot_funcs=[display_acf_plot, display_density_plot, display_time_series],
)
residuals_mean = Metric[float](
    name="Mean of the Residuals",
    key="residuals_mean",
    category=MetricCategories.residual,
    func=lambda y, pred: (y - pred).mean(),
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
residuals_std = Metric[float](
    name="Standard Deviation of the Residuals",
    key="residuals_std",
    category=MetricCategories.residual,
    func=lambda y, pred: (y - pred).std(),
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
ljung_box_statistics = Metric[pd.DataFrame](
    name="Ljung Box Statistics",
    key="ljung_box_statistics",
    category=MetricCategories.residual,
    func=ljung_box,
    info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
    restrict_to_sample=SampleTypes.insample,
    comp_complexity=ComputationalComplexity.high,
)

all_regression_metrics = [
    mae,
    mape,
    smape,
    mse,
    rmse,
    rmsle,
    r_two,
    residuals,
    residuals_mean,
    residuals_std,
    ljung_box_statistics,
]

minimal_regression_metrics = [
    mae,
    mape,
    mse,
    rmse,
    rmsle,
    r_two,
]

low_computation_regression_mterics = [
    metric
    for metric in all_regression_metrics
    if metric.comp_complexity is not ComputationalComplexity.high
]
