from copy import deepcopy
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

from krisi.utils.iterable_helpers import wrap_in_list

from .metric import Metric, PostProcessFunction
from .type import (
    Calculation,
    MetricFunction,
    MetricResult,
    PredictionsDS,
    ProbabilitiesDF,
    TargetsDS,
    WeightsDS,
)


def deepcopy_and_evaluate(
    postprocess_func: Metric, metric: Metric, sample_weight: WeightsDS, rolling: bool
) -> Metric:
    postprocess_metric = deepcopy(postprocess_func)
    postprocess_metric.key = f"{postprocess_metric.key}_{metric.key}"
    postprocess_metric.name = f"{postprocess_metric.name} of {metric.name}"
    postprocess_metric._evaluation(
        metric.result_rolling if rolling else metric.result, sample_weight=sample_weight
    )
    return postprocess_metric


class Group(Metric, Generic[MetricResult]):
    def __init__(
        self,
        name: str,
        key: str,
        metrics: List[Metric],
        calculation: Union[str, Calculation] = Calculation.both,
        preprocess_func: Optional[MetricFunction] = None,
        postprocess_funcs: Optional[
            Union[List[Metric], Metric, List[PostProcessFunction], PostProcessFunction]
        ] = None,
    ) -> None:
        self.calculation = Calculation.from_str(calculation)
        self.preprocess_func = preprocess_func
        self.postprocess_funcs = (
            wrap_in_list(postprocess_funcs) if postprocess_funcs else None
        )
        self.key = key
        self.name = name
        self.metrics = deepcopy(metrics)

        for metric in self.metrics:
            metric.key = f"{metric.key}_{self.key}"

    def _preprocess(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weights: WeightsDS,
    ) -> Tuple[Tuple, dict]:
        unnamed_results, named_results = [], dict()
        if self.preprocess_func is not None:
            if self.accepts_probabilities and probabilities is not None:
                unnamed_results = self.preprocess_func(
                    y, predictions, probabilities, sample_weights=sample_weights
                )
            else:
                unnamed_results = self.preprocess_func(
                    y, predictions, sample_weights=sample_weights
                )
        else:
            named_results = dict(
                y=y, predictions=predictions, probabilities=probabilities
            )

        if not isinstance(unnamed_results, (list, tuple)):
            unnamed_results = (unnamed_results,)
        return unnamed_results, named_results

    def _postprocess(
        self, all_metrics: List[Metric], sample_weight: WeightsDS, rolling: bool
    ) -> List[Metric]:
        if self.postprocess_funcs is not None:
            all_metrics_ = []
            for metric in all_metrics:
                for postprocess_func in self.postprocess_funcs:
                    if isinstance(postprocess_func, Metric):
                        all_metrics_.append(
                            deepcopy_and_evaluate(
                                postprocess_func, metric, sample_weight, rolling
                            )
                        )
                    else:
                        all_metrics_.append(
                            postprocess_func(metric, sample_weight, rolling)
                        )

            all_metrics = all_metrics_
        return all_metrics

    def evaluate(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weight: WeightsDS,
    ) -> List[Metric]:
        if self.calculation == Calculation.rolling:
            return []
        args, kwargs = self._preprocess(y, predictions, probabilities, sample_weight)
        all_metrics = [
            metric._evaluation(*args, **kwargs, sample_weight=sample_weight)
            for metric in self.metrics
        ]
        return self._postprocess(
            all_metrics, sample_weight=sample_weight, rolling=False
        )

    def evaluate_over_time(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weight: WeightsDS,
        rolling_args: Dict[str, Any],
    ) -> List[Metric]:
        if self.calculation == Calculation.single:
            return []
        args, kwargs = self._preprocess(y, predictions, probabilities, sample_weight)
        all_metrics = [
            metric._rolling_evaluation(
                *args, **kwargs, sample_weight=sample_weight, rolling_args=rolling_args
            )
            for metric in self.metrics
        ]
        return self._postprocess(all_metrics, sample_weight=sample_weight, rolling=True)

    def __str__(self) -> str:
        return "\n".join([metric.__str__() for metric in self.metrics])

    def __repr__(self) -> str:
        return "\n".join([metric.__repr__() for metric in self.metrics])
