from sklearn.metrics import mean_absolute_error, mean_squared_error

from krisi.evaluate import MCats, Metric, SampleTypes
from krisi.evaluate.library.metric_functions import ljung_box

default_metrics = [
    Metric[float](
        name="mae",
        full_name="Mean Absolute Error",
        category=MCats.reg_err,
        func=mean_absolute_error,
    ),
    Metric[float](
        name="mse",
        full_name="Mean Squared Error",
        category=MCats.reg_err,
        hyperparameters={"squared": True},
        func=mean_squared_error,
    ),
    Metric[float](
        name="rmse",
        full_name="Root Mean Squared Error",
        category=MCats.reg_err,
        hyperparameters={"squared": False},
        func=mean_squared_error,
    ),
    Metric[float](
        name="residuals_mean",
        full_name="Mean of the Residuals",
        category=MCats.residual,
        func=lambda y, pred: (y - pred).mean(),
    ),
    Metric[float](
        name="residuals_std",
        full_name="Standard Deviation of the Residuals",
        category=MCats.residual,
        func=lambda y, pred: (y - pred).std(),
    ),
    Metric(
        name="ljung-box-statistics",
        full_name="Ljung Box Statistics",
        category=MCats.residual,
        func=ljung_box,
        info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
        restrict_to_sample=SampleTypes.insample,
    ),
]

# ljung_box_score: Metric[float] = Metric(
#     name="ljung-box-score",
#     category=MCats.residual,
#     info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
# )
# mae: Metric[float] = Metric(
#     name="Mean Absolute Error", category=MCats.reg_err, info="Mean Absolute Error"
# )
# mse: Metric[float] = Metric(
#     name="Mean Squared Error", category=MCats.reg_err, info="Mean Squared Error"
# )
# rmse: Metric[float] = Metric(
#     name="Root Mean Squared Error",
#     category=MCats.reg_err,
#     info="Root Mean Squared Error",
# )
# aic: Metric[float] = Metric(
#     name="Akaike Information Criterion", category=MCats.entropy
# )
# bic: Metric[float] = Metric(
#     name="Bayesian Information Criterion", category=MCats.entropy
# )
# pacf_res: Metric[Tuple[float, float]] = Metric(
#     name="Partial Autocorrelation", category=MCats.residual
# )
# acf_res: Metric[Tuple[float, float]] = Metric(
#     name="Autocorrelation", category=MCats.residual
# )
# residuals_mean: Metric[float] = Metric(
#     name="Residual Mean", category=MCats.residual
# )
# residuals_std: Metric[float] = Metric(
#     name="Residual Standard Deviation", category=MCats.residual
# )
