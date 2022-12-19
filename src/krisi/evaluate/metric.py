from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

from krisi.evaluate.type import MetricFunction, Predictions, SampleTypes, Targets
from krisi.utils.iterable_helpers import string_to_id
from krisi.utils.printing import print_metric


class MetricCategories(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    unknown = "Unknown"


T = TypeVar("T", bound=Union[float, int, str, List, Tuple])


@dataclass
class Metric(Generic[T]):
    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[T] = None
    hyperparameters: dict = field(default_factory=dict)
    func: MetricFunction = lambda x, y: None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None

    def __post_init__(self):
        if self.key is "":
            self.key = string_to_id(self.name)

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown Field")

    def __str__(self) -> str:
        return print_metric(self)

    def __repr__(self) -> str:
        return print_metric(self, repr=True)

    def evaluate(self, y: Targets, prediction: Predictions) -> None:
        if self.result is not None:
            raise ValueError("This metric already contains a result.")
        else:
            self.result = self.func(y, prediction, **self.hyperparameters)
