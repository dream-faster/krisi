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
    Purpose,
    TargetsDS,
    WeightsDS,
)


def deepcopy_and_evaluate(
    postprocess_func: Metric, metric: Metric, sample_weight: WeightsDS, rolling: bool
) -> Metric:
    postprocess_metric = deepcopy(postprocess_func)
    postprocess_metric._evaluation(
        metric.result_rolling if rolling else metric.result, sample_weight=sample_weight
    )
    return postprocess_metric


class Group(Metric, Generic[MetricResult]):
    metrics: List[Metric]

    def __init__(
        self,
        name: str,
        key: str,
        metrics: List[Metric],
        calculation: Union[str, Calculation] = Calculation.both,
        preprocess_func: Optional[MetricFunction] = None,
        postprocess_funcs: Optional[
            Union[List[PostProcessFunction], PostProcessFunction]
        ] = None,
        purpose: Purpose = Purpose.group,
        append_key: bool = True,
    ) -> None:
        self.calculation = Calculation.from_str(calculation)
        self.preprocess_func = preprocess_func
        self.postprocess_funcs = (
            wrap_in_list(postprocess_funcs) if postprocess_funcs else None
        )
        self.key = key
        self.name = name
        self.metrics = deepcopy(metrics)
        self.purpose = purpose
        if append_key:
            for metric in self.metrics:
                metric.key = f"{metric.key}_{self.key}"

    def _preprocess(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weights: WeightsDS,
    ) -> Optional[Tuple]:
        results = (
            self.preprocess_func(
                y, predictions, probabilities, sample_weights=sample_weights
            )
            if self.preprocess_func is not None
            else None
        )

        if not isinstance(results, (list, tuple)) and results is not None:
            results = (results,)
        return results

    def _postprocess(
        self,
        all_metrics: List[Metric],
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weight: WeightsDS,
        rolling: bool,
    ) -> List[Metric]:
        if self.postprocess_funcs is not None:
            all_metrics_ = []

            for postprocess_func in self.postprocess_funcs:
                all_metrics_ += postprocess_func(
                    all_metrics,
                    y,
                    predictions,
                    probabilities,
                    sample_weight=sample_weight,
                    rolling=rolling,
                )

            all_metrics = all_metrics_
        self.metrics = all_metrics

        return self.metrics

    def get_all_metrics(self) -> List[Metric]:
        return self.metrics

    def __handle_args_to_pass_in(
        self,
        result: Any,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        accept_probabilities: bool,
    ):
        if result is None:
            if accept_probabilities:
                if probabilities is None:
                    return None
                else:
                    return [y, probabilities]
            else:
                return [y, predictions]
        else:
            return result

    def evaluate_rolling_properties(self):
        for metric in self.metrics:
            metric.evaluate_rolling_properties()

    def evaluate(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: ProbabilitiesDF,
        sample_weight: WeightsDS,
    ) -> List[Metric]:
        if self.calculation == Calculation.rolling:
            return []
        results = self._preprocess(y, predictions, probabilities, sample_weight)

        for metric in self.metrics:
            args_to_pass = self.__handle_args_to_pass_in(
                results,
                y,
                predictions,
                probabilities,
                metric.accepts_probabilities,
            )
            if args_to_pass is not None:
                metric._evaluation(
                    *args_to_pass,
                    sample_weight=sample_weight,
                )

        return self._postprocess(
            self.metrics,
            y,
            predictions,
            probabilities,
            sample_weight=sample_weight,
            rolling=False,
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
        results = self._preprocess(y, predictions, probabilities, sample_weight)

        for metric in self.metrics:
            args_to_pass = self.__handle_args_to_pass_in(
                results,
                y,
                predictions,
                probabilities,
                metric.accepts_probabilities,
            )
            if args_to_pass is not None:
                metric._rolling_evaluation(
                    *args_to_pass,
                    sample_weight=sample_weight,
                    rolling_args=rolling_args,
                )

        return self._postprocess(
            self.metrics,
            y,
            predictions,
            probabilities,
            sample_weight=sample_weight,
            rolling=True,
        )

    def __str__(self) -> str:
        return " - ".join([metric.key for metric in self.metrics])
