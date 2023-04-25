from dataclasses import dataclass
from typing import List, Optional

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
        self, y: TargetsDS, predictions: PredictionsDS, window: Optional[int] = None
    ) -> List[Metric]:
        results = self.group_func(y, predictions)

        return [
            metric.rolling_evaluation_(results, window=window)
            for metric in self.metrics
        ]

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
