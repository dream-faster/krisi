import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

import numpy as np

from krisi.utils.printing import bold, get_term_size, iterative_length


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


class MCats(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    unknown = "Unknown Category"


T = TypeVar("T", bound=Union[float, int, str, List, Tuple])


@dataclass
class Metric(Generic[T]):
    name: str
    category: MCats = MCats.unknown
    metric_result: Optional[T] = None
    hyperparameters: Optional[Any] = None
    info: str = ""

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown Field")

    def __str__(self) -> str:
        return print_metric(self)

    def __repr__(self) -> str:
        return print_metric(self, repr=True)


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
    residuals_mean: Metric[Tuple[float, float]] = Metric(
        name="Residual Mean", category=MCats.residual
    )
    residuals_std: Metric[Tuple[float, float]] = Metric(
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
                self.__dict__[key] = Metric(name=key, metric_result=item)
        else:
            if isinstance(item, dict):
                for key_, value_ in item.items():
                    if key_ in metric.__dict__:
                        metric[key_] = value_
                self.__dict__[key] = metric
            elif isinstance(item, Metric):
                self.__dict__[key] = item
            else:
                metric["metric_result"] = item
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
        return print_summary(self)

    def __repr__(self) -> str:
        return print_summary(self, repr=True)


def group_by_categories(flat_list: List[dict[str, Any]]) -> dict:
    categories = dict()
    for category in MCats:
        categories[category] = list(
            filter(
                lambda x: x["category"] == category
                if hasattr(x, "category")
                else False,
                flat_list,
            )
        )
    return categories


def print_summary(obj: ScoreCard, repr: bool = False) -> str:
    divider_len: int = get_term_size()
    full_str = ""
    full_str += f"\n\nResult of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    full_str += f"\n{'―'*divider_len}"

    categories = group_by_categories(list(vars(obj).values()))

    full_str += f"\n{'name':^30s}| {'result':^15s}| {'hyperparams':^15s}"

    for category, metrics in categories.items():
        full_str += f"\n\n\n{category.value:>15s}"
        full_str += f"\n{'.'*divider_len:>15s}"
        for metric in metrics:
            full_str += f"\n{str(metric):>15s}"

    return full_str


def handle_iterable_printing(obj: Any) -> str:
    if isinstance(obj, (str, float, int)):
        return str(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, np.ndarray):
        return f"Shape: {str(obj.shape)}"
    else:
        return f"Shape: {str(len(obj))}"


def print_metric(obj: Metric, repr: bool = False) -> str:
    hyperparams = ""
    if obj.hyperparameters is not None:
        hyperparams += "".join(
            [f"{key} - {value}" for key, value in obj.hyperparameters.items()]
        )

    return f"{obj.name:>30s}: {handle_iterable_printing(obj.metric_result):^15.5s}{hyperparams:>15s}"
