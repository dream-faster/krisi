from dataclasses import dataclass
from typing import Any, Dict, Generic, List

from .metric import Metric
from .type import (
    MetricFunction,
    MetricResult,
    PredictionsDS,
    ProbabilitiesDS,
    TargetsDS,
)


@dataclass
class Group(Metric, Generic[MetricResult]):
    def __init__(
        self, name: str, key: str, metrics: List[Metric], func: MetricFunction
    ) -> None:
        self.metrics = metrics
        self.group_func = func
        self.key = key
        self.name = name

    def evaluate(
        self, y: TargetsDS, predictions: PredictionsDS, probabilities: ProbabilitiesDS
    ) -> List[Metric]:
        if self.accepts_probabilities and probabilities is not None:
            results = self.group_func(y, predictions, probabilities)
        else:
            results = self.group_func(y, predictions)
        return [metric._evaluation(results) for metric in self.metrics]

    def evaluate_over_time(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDS,
        rolling_args: Dict[str, Any],
    ) -> List[Metric]:
        if self.accepts_probabilities and probabilities is not None:
            results = self.group_func(y, predictions, probabilities)
        else:
            results = self.group_func(y, predictions)
        return [
            metric._rolling_evaluation(results, rolling_args=rolling_args)
            for metric in self.metrics
        ]

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
