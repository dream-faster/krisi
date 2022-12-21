from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

from krisi.evaluate.type import (
    MetricCategories,
    MetricFunction,
    MResultGeneric,
    Predictions,
    SampleTypes,
    Targets,
)
from krisi.utils.iterable_helpers import string_to_id
from krisi.utils.printing import print_metric


@dataclass
class Metric(Generic[MResultGeneric]):
    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[MResultGeneric, List[MResultGeneric]]] = None
    parameters: dict = field(default_factory=dict)
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
        self.__set_result(self.func(y, prediction, **self.parameters))

    def evaluate_over_time(self, y: Targets, prediction: Predictions) -> None:
        self.__set_result([self.func(y[:i], prediction[:i]) for i in range(len(y))])

    def __set_result(self, result: Union[MResultGeneric, List[MResultGeneric]]):
        if self.result is not None:
            raise ValueError("This metric already contains a result.")
        else:
            self.result = result
