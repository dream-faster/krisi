from dataclasses import dataclass, field
from typing import Any, Generic, List, Optional, Union

from krisi.report.type import InteractiveFigure, PlotFunction, plotly_interactive
from krisi.utils.iterable_helpers import isiterable, string_to_id
from krisi.utils.printing import print_metric

from .assertions import check_valid_pred_target
from .type import (
    MetricCategories,
    MetricFunction,
    MetricResult,
    Predictions,
    SampleTypes,
    Targets,
)


@dataclass
class Metric(Generic[MetricResult]):
    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    result_over_time: Optional[
        Union[Exception, MetricResult, List[MetricResult]]
    ] = None
    parameters: dict = field(default_factory=dict)
    func: MetricFunction = lambda x, y: None
    plot_func: Optional[PlotFunction] = None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None

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
                result_over_time = [
                    self.func(
                        y[i : i + window],
                        predictions[i : i + window],
                        **self.parameters,
                    )
                    for i in range(0, len(y) - 1, window)
                ]
            else:
                # expanding
                result_over_time = [
                    self.func(y[: i + 1], predictions[: i + 1], **self.parameters)
                    for i in range(len(y) - 1)
                ]
        except Exception as e:
            result_over_time = e

        self.__safe_set(result_over_time, key="result_over_time")

    def get_diagram_over_time(self) -> Optional[InteractiveFigure]:
        return create_diagram(self)

    def __safe_set(
        self, result: Union[Exception, MetricResult, List[MetricResult]], key: str
    ):
        if self.__dict__[key] is not None:
            raise ValueError("This metric already contains a result.")
        else:
            self.__dict__[key] = result


def create_diagram(obj: Metric) -> Optional[InteractiveFigure]:

    if isinstance(obj.result_over_time, Exception) or obj.result_over_time is None:
        return None
    elif isiterable(obj.result_over_time) and obj.plot_func is not None:
        return InteractiveFigure(
            obj.key,
            get_figure=plotly_interactive(
                obj.plot_func, obj.result_over_time, name=obj.name
            ),
        )
    else:
        return None
