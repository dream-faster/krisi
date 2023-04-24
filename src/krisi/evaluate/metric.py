import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Generic, List, Optional, Union

from krisi.evaluate.assertions import check_valid_pred_target
from krisi.evaluate.type import (
    ComputationalComplexity,
    MetricCategories,
    MetricFunction,
    MetricResult,
    Predictions,
    SampleTypes,
    Targets,
)
from krisi.report.console import print_metric
from krisi.report.type import InteractiveFigure, PlotFunction, plotly_interactive
from krisi.utils.iterable_helpers import isiterable, string_to_id


@dataclass
class Metric(Generic[MetricResult]):
    """
    Class representing a metric.

    Parameters
    ----------
    name: str
        The name of the metric.
    key: str
        The key used to reference the metric.
    category: MetricCategories
        The category of the metric.
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]]
        The result of the evaluated `Metric`, by default None
    result_rolling: Optional[Union[Exception, MetricResult, List[MetricResult]]]
        The result of the evaluated `Metric` over time, by default None
    parameters: dict
        The paramaters that are passed into the evaluation function (param: `func`), by default field(default_factory=dict)
    func: Callable
        The function used to compute the metric.
    plot_funcs: List[Callable]
        List of functions used to plot the metric.
    plot_func_rolling: Callable
        Function used to plot the rolling metric value.
    info: str
        Additional information about the metric.
    restrict_to_sample: Optional[SampleTypes]
        Weather the `Metric` should only be evaluated on insample or out of sample data, by default None
    comp_complexity: Optional[ComputationalComplexity]
        How resource intensive the calculation is, by default None
    """

    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    result_rolling: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    parameters: dict = field(default_factory=dict)
    func: Optional[MetricFunction] = None
    plot_funcs: Optional[List[PlotFunction]] = None
    plot_func_rolling: Optional[PlotFunction] = None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None
    comp_complexity: Optional[ComputationalComplexity] = None
    func_group: Optional[Callable] = None

    def __post_init__(self):
        if self.key == "":
            self.key = string_to_id(self.name)

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown Field")

    def __str__(self) -> str:
        return print_metric(self)

    def __repr__(self) -> str:
        return print_metric(self, repr=True)

    def evaluate(self, y: Targets, predictions: Predictions) -> None:
        assert self.func is not None, "`func` has to be set to calculate Metric result."
        check_valid_pred_target(y, predictions)

        try:
            result = self.func(y, predictions, **self.parameters)
        except Exception as e:
            result = e
        self.__safe_set(result, key="result")

    def evaluate_in_group(self, *args) -> "Metric":
        assert (
            self.func_group is not None
        ), "Group Function has to be set to be able to `evaluate_in_group`."
        try:
            result = self.func_group(*args, **self.parameters)
        except Exception as e:
            result = e
        self.__safe_set(result, key="result")

        return self

    def evaluate_over_time_in_group(
        self, *args, window: Optional[int] = None
    ) -> "Metric":
        assert (
            self.func_group is not None
        ), "Group Function has to be set to be able to `evaluate_in_group`."
        try:
            if window is not None:
                result_rolling = [
                    self.func_group(
                        *[el[i : i + window] for el in args],
                        **self.parameters,
                    )
                    for i in range(0, len(args[0]) - 1, window)
                ]
            else:
                # expanding
                result_rolling = [
                    self.func_group(*[el[: i + 1] for el in args], **self.parameters)
                    for i in range(len(args[0]) - 1)
                ]
        except Exception as e:
            result_rolling = e

        self.__safe_set(result_rolling, key="result_rolling")
        return self

    def evaluate_over_time(
        self, y: Targets, predictions: Predictions, window: Optional[int] = None
    ) -> None:
        try:
            if window is not None:
                result_rolling = [
                    self.func(
                        y[i : i + window],
                        predictions[i : i + window],
                        **self.parameters,
                    )
                    for i in range(0, len(y) - 1, window)
                ]
            else:
                # expanding
                result_rolling = [
                    self.func(y[: i + 1], predictions[: i + 1], **self.parameters)
                    for i in range(len(y) - 1)
                ]
        except Exception as e:
            result_rolling = e

        self.__safe_set(result_rolling, key="result_rolling")

    def is_evaluated(self, rolling: bool = False):
        if rolling:
            return self.result_rolling is not None
        else:
            return self.result is not None

    def get_diagram_over_time(self) -> Optional[List[InteractiveFigure]]:
        return create_diagram_rolling(self)

    def get_diagrams(
        self,
    ) -> Optional[List[InteractiveFigure]]:
        return create_diagram(self)

    def print(
        self,
        mode: Union[str, List[str]] = "dummy",
        with_info: bool = False,
        input_analysis: bool = True,
        title: Optional[str] = None,
    ) -> None:
        pass

    def __safe_set(
        self, result: Union[Exception, MetricResult, List[MetricResult]], key: str
    ):
        if self.__dict__[key] is not None:
            raise ValueError("This metric already contains a result.")
        else:
            self.__dict__[key] = result


def create_diagram_rolling(obj: Metric) -> Optional[List[InteractiveFigure]]:
    if obj.plot_func_rolling is None:
        logging.info("No plot_func_rolling (Plotting Function Rolling) specified")
        return None
    elif isinstance(obj.result_rolling, Exception) or obj.result_rolling is None:
        return None
    elif isiterable(obj.result_rolling):
        return [
            InteractiveFigure(
                f"{obj.key}_{obj.plot_func_rolling.__name__}",
                get_figure=plotly_interactive(
                    obj.plot_func_rolling, obj.result_rolling, title=obj.name
                ),
                category=obj.category,
            )
        ]
    else:
        return None


def create_diagram(obj: Metric) -> Optional[List[InteractiveFigure]]:
    if obj.plot_funcs is None:
        logging.info("No plot_func (Plotting Function) specified")
        return None
    elif isinstance(obj.result, Exception) or obj.result is None:
        return None
    else:
        return [
            InteractiveFigure(
                f"{obj.key}_{plot_func.__name__}",
                get_figure=plotly_interactive(plot_func, obj.result, title=obj.name),
                title=f"{obj.name} - {plot_func.__name__}",
                category=obj.category,
            )
            for plot_func in obj.plot_funcs
        ]
