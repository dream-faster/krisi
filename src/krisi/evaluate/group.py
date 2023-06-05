from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from krisi.utils.iterable_helpers import wrap_in_list

from .metric import Metric, PostProcessFunction
from .type import (
    MetricFunction,
    MetricResult,
    PredictionsDS,
    ProbabilitiesDF,
    TargetsDS,
)


def deepcopy_and_evaluate(
    postprocess_func: Metric, metric: Metric, rolling: bool
) -> Metric:
    postprocess_metric = deepcopy(postprocess_func)
    postprocess_metric.key = f"{postprocess_metric.key}_{metric.key}"
    postprocess_metric.name = f"{postprocess_metric.name} of {metric.name}"
    postprocess_metric._evaluation(metric.result_rolling if rolling else metric.result)
    return postprocess_metric


@dataclass
class Group(Metric, Generic[MetricResult]):
    def __init__(
        self,
        name: str,
        key: str,
        metrics: List[Metric],
        preprocess_func: Optional[MetricFunction] = None,
        postprocess_funcs: Optional[
            Union[List[Metric], Metric, List[PostProcessFunction], PostProcessFunction]
        ] = None,
    ) -> None:
        self.preprocess_func = preprocess_func
        self.postprocess_funcs = (
            wrap_in_list(postprocess_funcs) if postprocess_funcs else None
        )
        self.key = key
        self.name = name
        self.metrics = deepcopy(metrics)

        for metric in self.metrics:
            metric.key = f"{metric.key}_{self.key}"

        if self.postprocess_funcs is not None:
            for postprocess_func in self.postprocess_funcs:
                if isinstance(postprocess_func, Metric):
                    postprocess_func.key = f"{self.key}_{postprocess_func.key}"

    def _preprocess(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
    ) -> Union[
        Tuple[Any, None], Tuple[TargetsDS, PredictionsDS, Optional[ProbabilitiesDF]]
    ]:
        if self.preprocess_func is not None:
            if self.accepts_probabilities and probabilities is not None:
                results = self.preprocess_func(y, predictions, probabilities)
            else:
                results = self.preprocess_func(y, predictions)
        else:
            results = (y, predictions, probabilities)

        if not isinstance(results, (list, tuple)):
            results = (results,)
        return results

    def _postprocess(self, all_metrics: List[Metric], rolling: bool) -> List[Metric]:
        if self.postprocess_funcs is not None:
            all_metrics_ = []
            for metric in all_metrics:
                for postprocess_func in self.postprocess_funcs:
                    if isinstance(postprocess_func, Metric):
                        all_metrics_.append(
                            deepcopy_and_evaluate(postprocess_func, metric, rolling)
                        )
                    else:
                        all_metrics_.append(postprocess_func(metric))

            all_metrics = all_metrics_
        return all_metrics

    def evaluate(
        self, y: TargetsDS, predictions: PredictionsDS, probabilities: ProbabilitiesDF
    ) -> List[Metric]:
        results = self._preprocess(y, predictions, probabilities)
        all_metrics = [metric._evaluation(*results) for metric in self.metrics]
        return self._postprocess(all_metrics, rolling=False)

    def evaluate_over_time(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        rolling_args: Dict[str, Any],
    ) -> List[Metric]:
        results = self._preprocess(y, predictions, probabilities)
        all_metrics = [
            metric._rolling_evaluation(*results, rolling_args=rolling_args)
            for metric in self.metrics
        ]
        return self._postprocess(all_metrics, rolling=True)

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
