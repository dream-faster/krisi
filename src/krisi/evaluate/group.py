from dataclasses import dataclass
from typing import Any, List, Optional

from .metric import Metric
from .type import MetricFunction, PredictionsDS, TargetsDS


@dataclass
class Group(Metric):
    def __init__(
        self, name: str, key: str, metrics: List[Metric], func: MetricFunction
    ) -> None:
        self.metrics = metrics
        self.group_func = func
        self.key = key
        self.name = name

    def evaluate(self, y: TargetsDS, predictions: PredictionsDS) -> List[Metric]:
        results = self.group_func(y, predictions)

        return [metric.evaluation_(results) for metric in self.metrics]

    def evaluate_over_time(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        rolling_args: Optional[dict[str, Any]],
    ) -> List[Metric]:
        results = self.group_func(y, predictions)

        return [
            metric.rolling_evaluation_(results, rolling_args=rolling_args)
            for metric in self.metrics
        ]

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
