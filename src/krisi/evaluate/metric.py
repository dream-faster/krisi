import logging
from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Union

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
    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    result_rolling: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    parameters: dict = field(default_factory=dict)
    func: MetricFunction = lambda x, y: None
    plot_funcs: Optional[List[PlotFunction]] = None
    plot_func_rolling: Optional[PlotFunction] = None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None
    comp_complexity: Optional[ComputationalComplexity] = None

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
        check_valid_pred_target(y, predictions)

        try:
            result = self.func(y, predictions, **self.parameters)
        except Exception as e:
            result = e
        self.__safe_set(result, key="result")

    def evaluate_over_time(
        self, y: Targets, predictions: Predictions, window: Optional[int] = None
    ) -> None:
        try:
            if window:
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

    def get_diagram_over_time(self) -> Optional[List[InteractiveFigure]]:
        return create_diagram_rolling(self)

    def get_diagrams(
        self,
    ) -> Optional[List[InteractiveFigure]]:
        return create_diagram(self)

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
