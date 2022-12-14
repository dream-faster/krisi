import logging
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

from krisi.utils.printing import bold, iterative_length


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


class MetricCats(Enum):
    residual = "Residual Diagnostics"
    classification_error = "Forecast Errors - Classification"
    regression_error = "Forecast Errors - Regression"
    unknown = "Unknown Category"


T = TypeVar("T", bound=Union[float, int, str, List])


@dataclass
class MetricSummary(Generic[T]):
    category: MetricCats = MetricCats.unknown
    metric_result: Optional[T] = None
    hyperparameters: Optional[Any] = None
    info: str = ""

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown Field")


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """

    # ljung_box_score: MetricSummary[float] = MetricSummary(
    #     category=MetricCats.residual, info="If p is larger than our significance level"
    # )
    mse: MetricSummary[float] = MetricSummary(
        category=MetricCats.regression_error, info="Mean Squared Error"
    )
    hearst_exponent: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    pacf_res: Optional[Tuple[float, float]] = None
    acf_res: Optional[Tuple[float, float]] = None

    def __init__(
        self, model_name: str, dataset_name: str, sample_type: SampleTypes
    ) -> None:
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.sample_type = sample_type

    def __setattr__(self, key: str, item: Any) -> None:
        metric = getattr(self, key, None)
        if metric is None:
            logging.warning(
                "No such metric exists, creating a new one.\nConsider using set_new_metric to also define category and information regarding the metric."
            )
            self.__dict__[key] = MetricSummary(item)
        else:
            metric["metric_result"] = item
            self.__dict__[key] = metric

    def set_new_metric(self, metric_key: str, metric: MetricSummary) -> None:
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


def print_summary(obj: ScoreCard, repr: bool = False) -> str:
    title = f"\n\nResult of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    title += f"\n{'―'*len(title)}"
    return (
        f'{" "*4}\n'.join(
            [title]
            + [
                f"{str(key):>15s}: {f'Iterable of shape: {str(iterative_length(value))}' if isinstance(value, Iterable) and not isinstance(value, str) else str(value):<15s}"
                for key, value in vars(obj).items()
                if value is not None
            ]
        )
        + "\n\n"
    )
