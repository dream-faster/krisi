import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, List, Optional, Tuple, Union

import pandas as pd

from krisi.evaluate.type import (
    Calculation,
    ComputationalComplexity,
    MetricCategories,
    MetricFunction,
    MetricResult,
    PredictionsDS,
    ProbabilitiesDF,
    SampleTypes,
    TargetsDS,
    WeightsDS,
)
from krisi.report.console import print_metric
from krisi.report.type import InteractiveFigure, PlotDefinition, plotly_interactive
from krisi.utils.iterable_helpers import (
    filter_nan,
    isiterable,
    string_to_id,
    wrap_in_list,
)
from krisi.utils.state import RunType, get_global_state


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
    plot_funcs_rolling: Callable
        Function used to plot the rolling metric value.
    info: str
        Additional information about the metric.
    restrict_to_sample: Optional[SampleTypes]
        Weather the `Metric` should only be evaluated on insample or out of sample data, by default None
    comp_complexity: Optional[ComputationalComplexity]
        How resource intensive the calculation is, by default None
    calculation: Calculation
        Whether the metric should be evaluated only when calculating rolling, single or both, by default both
    accepts_probabilities: bool
        Whether the metric accepts probabilities as input, by default False
    supports_multiclass: bool
        Whether the metric supports multiclass classification, by default False
    """

    name: str
    key: str = ""
    category: Optional[MetricCategories] = None
    result: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    result_rolling: Optional[Union[Exception, MetricResult, List[MetricResult]]] = None
    parameters: dict = field(default_factory=dict)
    func: Optional[MetricFunction] = None
    plot_funcs: Optional[Union[List[PlotDefinition], PlotDefinition]] = None
    plot_funcs_rolling: Optional[Union[List[PlotDefinition], PlotDefinition]] = None
    info: str = ""
    restrict_to_sample: Optional[SampleTypes] = None
    comp_complexity: Optional[ComputationalComplexity] = None
    calculation: Union[str, Calculation] = Calculation.both
    accepts_probabilities: bool = False
    supports_multiclass: bool = False
    diagnostics: Optional[Dict[str, Any]] = field(default_factory=dict)
    _from_group: bool = False

    def __post_init__(self):
        if self.key == "":
            self.key = string_to_id(self.name)
        if not isinstance(self.plot_funcs, list) and self.plot_funcs is not None:
            self.plot_funcs = wrap_in_list(self.plot_funcs)
        if (
            not isinstance(self.plot_funcs_rolling, list)
            and self.plot_funcs_rolling is not None
        ):
            self.plot_funcs_rolling = wrap_in_list(self.plot_funcs_rolling)
        self.calculation = Calculation.from_str(self.calculation)

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown Field")

    def __str__(self) -> str:
        return print_metric(self)

    def __repr__(self) -> str:
        print(print_metric(self, repr=True))
        return super().__repr__()

    def _evaluation(self, *args, **kwargs) -> "Metric":
        if self.calculation == Calculation.rolling:
            return self
        if self._from_group:
            return self
        if self.func is None:
            raise ValueError("`func` has to be set on Metric to calculate result.")
        try:
            result = self.func(*args, **kwargs, **self.parameters)
        except Exception as e:
            result = e
            if get_global_state().run_type == RunType.test:
                raise e
        self.__safe_set(result, key="result")
        return self

    def evaluate(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: Optional[ProbabilitiesDF] = None,
        sample_weight: Optional[WeightsDS] = None,
    ) -> None:
        assert (
            self.func is not None
        ), "`func` has to be set on Metric to calculate result."
        if self.accepts_probabilities:
            if probabilities is not None:
                self._evaluation(
                    y, predictions, probabilities, sample_weight=sample_weight
                )
            else:
                self.__safe_set(
                    ValueError(
                        "Metric requires probabilities, but None were provided."
                    ),
                    key="result",
                )
        else:
            self._evaluation(y, predictions, sample_weight=sample_weight)
        if sample_weight is not None:
            self.__dict__["diagnostics"] = dict(used_sample_weight=True)

    @staticmethod
    def __handle_window(
        window,
    ) -> Tuple[TargetsDS, PredictionsDS, Optional[ProbabilitiesDF], dict]:
        y = window["y"]
        pred = window["predictions"]
        if len(window.columns) > 2:
            sample_weight = (
                window.pop("sample_weight") if "sample_weight" in window else None
            )
            prob = window.iloc[:, 2:]

            return y, pred, prob, dict(sample_weight=sample_weight)
        else:
            return y, pred, None, dict(sample_weight=None)

    @staticmethod
    def __calc_window(window: pd.DataFrame, func: MetricFunction, parameters: dict):
        y, pred, prob, sample_weight = Metric.__handle_window(window)

        return func(
            *filter_nan([y, pred, prob]),
            **sample_weight,
            **parameters,
        )

    def _rolling_evaluation(self, *args, rolling_args: dict) -> "Metric":
        if self._from_group:
            return self
        if self.calculation == Calculation.single:
            return self
        if self.func is None:
            raise ValueError("`func` has to be set on Metric to calculate result.")
        else:
            _df = pd.concat(args, axis="columns")
            if "sample_weight" in _df:
                self.__dict__["diagnostics"] = dict(used_sample_weight=True)
            try:
                df_rolled = (
                    _df.expanding()
                    if "window" in rolling_args and rolling_args["window"] is None
                    else _df.rolling(**rolling_args)
                )

                result_rolling = [
                    Metric.__calc_window(single_window, self.func, self.parameters)
                    for single_window in df_rolled
                    if len(single_window) > 0
                ]

            except Exception as e:
                result_rolling = e
                if get_global_state().run_type == RunType.test:
                    raise e
            self.__safe_set(result_rolling, key="result_rolling")

        return self

    def evaluate_over_time(
        self,
        y: TargetsDS,
        predictions: PredictionsDS,
        probabilities: Optional[ProbabilitiesDF] = None,
        sample_weight: Optional[WeightsDS] = None,
        rolling_args: dict = dict(),
    ) -> None:
        if self.accepts_probabilities and probabilities is not None:
            self._rolling_evaluation(
                y, predictions, probabilities, sample_weight, rolling_args=rolling_args
            )
        else:
            self._rolling_evaluation(
                y, predictions, sample_weight, rolling_args=rolling_args
            )

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
    if obj.plot_funcs_rolling is None:
        logging.info("No plot_funcs_rolling (Plotting Function Rolling) specified")
        return None
    elif isinstance(obj.result_rolling, Exception) or obj.result_rolling is None:
        return None
    elif isiterable(obj.result_rolling) and len(obj.result_rolling) > 0:
        if isinstance(obj.result_rolling[0], (pd.DataFrame, pd.Series)):
            return None
        return [
            InteractiveFigure(
                f"{obj.key}_{plot_funcs_rolling.__name__}",
                get_figure=plotly_interactive(
                    plot_funcs_rolling, obj.result_rolling, title=obj.name
                ),
                title=f"{obj.name} - {plot_funcs_rolling.__name__}",
                category=obj.category,
                plot_args=merge_default_dict(plot_args),
            )
            for plot_funcs_rolling, plot_args in obj.plot_funcs_rolling
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


PostProcessFunction = Callable[[Metric], Metric]
