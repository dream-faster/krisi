from sklearn.metrics import mean_absolute_error, mean_squared_error

from krisi.evaluate.library.metric_functions import ljung_box
from krisi.evaluate.metric import Metric, MetricCategories
from krisi.evaluate.type import SampleTypes

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
        hyperparameters={"squared": True},
        func=mean_squared_error,
    ),
    Metric[float](
        name="Root Mean Squared Error",
        key="rmse",
        category=MetricCategories.reg_err,
        hyperparameters={"squared": False},
        func=mean_squared_error,
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
    Metric(
        name="Ljung Box Statistics",
        key="ljung_box_statistics",
        category=MetricCategories.residual,
        func=ljung_box,
        info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
        restrict_to_sample=SampleTypes.insample,
    ),
]

# ljung_box_score: Metric[float] = Metric(
#     name="ljung-box-score",
#     category=MetricCategories.residual,
#     info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
# )
# mae: Metric[float] = Metric(
#     name="Mean Absolute Error", category=MetricCategories.reg_err, info="Mean Absolute Error"
# )
# mse: Metric[float] = Metric(
#     name="Mean Squared Error", category=MetricCategories.reg_err, info="Mean Squared Error"
# )
# rmse: Metric[float] = Metric(
#     name="Root Mean Squared Error",
#     category=MetricCategories.reg_err,
#     info="Root Mean Squared Error",
# )
# aic: Metric[float] = Metric(
#     name="Akaike Information Criterion", category=MetricCategories.entropy
# )
# bic: Metric[float] = Metric(
#     name="Bayesian Information Criterion", category=MetricCategories.entropy
# )
# pacf_res: Metric[Tuple[float, float]] = Metric(
#     name="Partial Autocorrelation", category=MetricCategories.residual
# )
# acf_res: Metric[Tuple[float, float]] = Metric(
#     name="Autocorrelation", category=MetricCategories.residual
# )
# residuals_mean: Metric[float] = Metric(
#     name="Residual Mean", category=MetricCategories.residual
# )
# residuals_std: Metric[float] = Metric(
#     name="Residual Standard Deviation", category=MetricCategories.residual
# )
