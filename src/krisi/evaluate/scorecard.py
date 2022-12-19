import logging
from dataclasses import dataclass
from typing import Any, List, Tuple

from rich import print
from rich.pretty import Pretty

from krisi.evaluate.library.default_metrics import default_metrics
from krisi.evaluate.metric import MCats, Metric
from krisi.evaluate.type import SampleTypes
from krisi.utils.iterable_helpers import map_newdict_on_olddict
from krisi.utils.printing import get_summary


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """
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

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        sample_type: SampleTypes,
        default_metrics: List[Metric] = default_metrics,
    ) -> None:
        self.__dict__["model_name"] = model_name
        self.__dict__["dataset_name"] = dataset_name
        self.__dict__["sample_type"] = sample_type

        for metric in default_metrics:
            self.__dict__[metric.name] = metric

    def __setattr__(self, key: str, item: Any) -> None:
        metric = getattr(self, key, None)

        if metric is None:
            if isinstance(item, dict):
                self.__dict__[key] = Metric(**item)
            elif isinstance(item, Metric):
                self.__dict__[key] = item
            else:
                self.__dict__[key] = Metric(
                    name=key, result=item, category=MCats.unknown
                )
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

    def __setitem__(self, key: str, item: Any) -> None:
        self.__setattr__(key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown metric")

    def __delitem__(self, key: str) -> None:
        setattr(self, key, None)

    def __str__(self) -> str:
        print(Pretty(self.__dict__))
        return ""

    def __repr__(self) -> str:
        print(Pretty(self.__dict__))
        return ""

    def print_summary(self, with_info: bool = False) -> None:
        print(
            get_summary(
                self,
                repr=True,
                categories=[el.value for el in MCats],
                with_info=with_info,
            )
        )
