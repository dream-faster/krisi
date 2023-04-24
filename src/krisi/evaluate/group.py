from dataclasses import dataclass
from typing import List, Optional

from krisi.evaluate.assertions import check_valid_pred_target

from .metric import Metric
from .type import MetricFunction, Predictions, Targets


@dataclass
class Group(Metric):
    def __init__(
        self, name: str, key: str, metrics: List[Metric], func: MetricFunction
    ) -> None:
        self.metrics = metrics
        self.group_func = func
        self.key = key
        self.name = name

    def evaluate(self, y: Targets, predictions: Predictions) -> List[Metric]:
        check_valid_pred_target(y, predictions)

        results = self.group_func(y, predictions)

        return [metric.evaluate_in_group(results) for metric in self.metrics]

    def evaluate_over_time(
        self, y: Targets, predictions: Predictions, window: Optional[int] = None
    ) -> List[Metric]:
        check_valid_pred_target(y, predictions)

        results = self.group_func(y, predictions)

        return [
            metric.evaluate_over_time_in_group(results, window=window)
            for metric in self.metrics
        ]

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
