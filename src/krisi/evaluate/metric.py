from dataclasses import dataclass
from enum import Enum
from typing import Any, Generic, List, Optional, Tuple, TypeVar, Union

from krisi.utils.printing import print_metric


class MCats(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"


T = TypeVar("T", bound=Union[float, int, str, List, Tuple])


@dataclass
class Metric(Generic[T]):
    name: str
    category: Optional[MCats] = None
    result: Optional[T] = None
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