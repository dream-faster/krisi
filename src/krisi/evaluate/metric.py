import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, Union

import pandas as pd

from krisi.evaluate.type import (
    ComputationalComplexity,
    MetricCategories,
    MetricFunction,
    MetricResult,
    Predictions,
    PredictionsDS,
    SampleTypes,
    Targets,
    TargetsDS,
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
    disable_rolling: bool
        Whether the metric should not be evaluated on a rolling basis, by default False
        If this is switched on, the metric will still be evaluated when `evaluate_over_time` is called
        but not on a rolling basis.
    """

    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    result_rolling: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    parameters: dict = field(default_factory=dict)
    func: Optional[MetricFunction] = None
    plot_funcs: Optional[List[Tuple[PlotFunction, Optional[Dict[str, Any]]]]] = None
    plot_func_rolling: Optional[Tuple[PlotFunction, Optional[Dict[str, Any]]]] = None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None
    comp_complexity: Optional[ComputationalComplexity] = None
    disable_rolling: bool = False

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
        print(print_metric(self, repr=True))
        return super().__repr__()

    def _evaluation(self, *args) -> "Metric":
        try:
            result = self.func(*args, **self.parameters)
        except Exception as e:
            result = e
        self.__safe_set(result, key="result")
        return self

    def evaluate(self, y: Targets, predictions: Predictions) -> None:
        assert (
            self.func is not None
        ), "`func` has to be set on Metric to calculate result."

        self._evaluation(y, predictions)

    def _rolling_evaluation(self, *args, rolling_args: dict) -> "Metric":
        if self.disable_rolling:
            self._evaluation(*args)
        else:
            _df = pd.concat(args, axis="columns")
            try:
                df_rolled = (
                    _df.expanding()
                    if rolling_args["window"] is None
                    else _df.rolling(**rolling_args)
                )

                result_rolling = [
                    self.func(*single_window.values.T, **self.parameters)
                    for single_window in df_rolled
                ]
            except Exception as e:
                result_rolling = e
            self.__safe_set(result_rolling, key="result_rolling")

        return self

    def evaluate_over_time(
        self, y: TargetsDS, predictions: PredictionsDS, rolling_args: dict
    ) -> None:
        self._rolling_evaluation(y, predictions, rolling_args=rolling_args)

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
                f"{obj.key}_{obj.plot_func_rolling[0].__name__}",
                get_figure=plotly_interactive(
                    obj.plot_func_rolling[0], obj.result_rolling, title=obj.name
                ),
                title=f"{obj.name} - {obj.plot_func_rolling[0].__name__}",
                category=obj.category,
                plot_args=merge_default_dict(obj.plot_func_rolling[1]),
            )
        ]
    else:
        return None


def merge_default_dict(
    new_dict: Optional[Dict[str, Any]] = None,
    default_dict: Dict[str, Any] = dict(width=900.0, height=600.0),
) -> Dict[str, Any]:
    if new_dict is None:
        return default_dict
    else:
        return {**default_dict, **new_dict}


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
                plot_args=merge_default_dict(plot_args),
            )
            for plot_func, plot_args in obj.plot_funcs
        ]
