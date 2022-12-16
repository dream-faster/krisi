import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Tuple

from rich import print

from krisi.evaluate.metric import MCats, Metric
from krisi.utils.iterable_helpers import map_newdict_on_olddict
from krisi.utils.printing import get_summary


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """
    ljung_box_score: Metric[float] = Metric(
        name="ljung-box-score",
        category=MCats.residual,
        info="If p is larger than our significance level then we cannot dismiss the null-hypothesis that the residuals are a random walk.",
    )
    mae: Metric[float] = Metric(
        name="Mean Absolute Error", category=MCats.reg_err, info="Mean Absolute Error"
    )
    mse: Metric[float] = Metric(
        name="Mean Squared Error", category=MCats.reg_err, info="Mean Squared Error"
    )
    rmse: Metric[float] = Metric(
        name="Root Mean Squared Error",
        category=MCats.reg_err,
        info="Root Mean Squared Error",
    )
    aic: Metric[float] = Metric(
        name="Akaike Information Criterion", category=MCats.entropy
    )
    bic: Metric[float] = Metric(
        name="Bayesian Information Criterion", category=MCats.entropy
    )
    pacf_res: Metric[Tuple[float, float]] = Metric(
        name="Partial Autocorrelation", category=MCats.residual
    )
    acf_res: Metric[Tuple[float, float]] = Metric(
        name="Autocorrelation", category=MCats.residual
    )
    residuals_mean: Metric[float] = Metric(
        name="Residual Mean", category=MCats.residual
    )
    residuals_std: Metric[float] = Metric(
        name="Residual Standard Deviation", category=MCats.residual
    )
    # hearst_exponent: Metric[float] = Metric(category=MCats.entropy)

    def __init__(
        self, model_name: str, dataset_name: str, sample_type: SampleTypes
    ) -> None:
        self.__dict__["model_name"] = model_name
        self.__dict__["dataset_name"] = dataset_name
        self.__dict__["sample_type"] = sample_type

    def __setattr__(self, key: str, item: Any) -> None:
        metric = getattr(self, key, None)

        if metric is None:
            logging.warning(
                "No such metric exists, creating a new one.\nConsider using set_new_metric to also define category and information regarding the metric."
            )
            if isinstance(item, Metric):
                self.__dict__[key] = item
            else:
                self.__dict__[key] = Metric(name=key, result=item)
        else:
            if isinstance(item, dict):
                metric_dict = map_newdict_on_olddict(
                    vars(metric), item, exclude=["name"]
                )
                self.__dict__[key] = Metric(**metric_dict)
            elif isinstance(item, Metric):
                metric_dict = map_newdict_on_olddict(
                    vars(metric), vars(item), exclude=["name"]
                )
                self.__dict__[key] = Metric(**metric_dict)
            else:
                metric["result"] = item
                self.__dict__[key] = metric

    def set_new_metric(self, metric_key: str, metric: Metric) -> None:
        self.__dict__[metric_key] = metric

    def __setitem__(self, key: str, item: Any) -> None:
        self.__setattr__(key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown metric")

    def __delitem__(self, key: str) -> None:
        setattr(self, key, None)

    def __str__(self) -> str:
        print(get_summary(self, categories=[el.value for el in MCats]))
        return ""

    def __repr__(self) -> str:
        print(get_summary(self, repr=True, categories=[el.value for el in MCats]))
        return ""
