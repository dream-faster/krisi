from __future__ import annotations

import datetime
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Union

import pandas as pd
from rich import print
from typing_extensions import Literal

from krisi.evaluate.assertions import (
    check_no_duplicate_metrics,
    check_valid_pred_target,
    infer_dataset_type,
)
from krisi.evaluate.group import Group
from krisi.evaluate.library import get_default_metrics_for_dataset_type
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import (
    DatasetType,
    MetricCategories,
    PathConst,
    Predictions,
    PredictionsDS,
    PrintMode,
    Probabilities,
    ProbabilitiesDF,
    SampleTypes,
    SaveModes,
    ScoreCardMetadata,
    Targets,
    TargetsDS,
    Weights,
    WeightsDS,
)
from krisi.evaluate.utils import (
    convert_to_series,
    ensure_df,
    get_save_path,
    handle_unnamed,
    rename_probs_columns,
)
from krisi.report.console import (
    get_large_metric_summary,
    get_minimal_summary,
    get_summary,
)
from krisi.report.report import create_report_from_scorecard
from krisi.report.type import DisplayModes, InteractiveFigure
from krisi.utils.environment import is_notebook
from krisi.utils.io import ensure_path, save_console, save_minimal_summary, save_object
from krisi.utils.iterable_helpers import (
    empty_if_none,
    flatten,
    map_newdict_on_olddict,
    remove_nans,
    replace_if_none,
    strip_builtin_functions,
    wrap_in_list,
)
from krisi.utils.state import RunType, get_global_state, set_global_state

logger = logging.getLogger("krisi")


class ScoreCard:
    """
    ScoreCard Object.

    Krisi's main object holding and evaluating metrics.
    Stores, evaluates and generates vizualisations of predefined
    and custom metrics for regression and classification.

    Parameters
    -------
    y: Targets
        True Targets to which the metrics are evaluated to.
    predictions: Predictions
        The single point predictions to which the metrics are evaluated to.
    model_name: Optional[str]
        The name of the model that the predictions were generated by. Used for identifying scorecards.
    model_description: str
        A description of the model that the predictions were generated by. Used for reporting.
    dataset_name: Optional[str]
        The name of the dataset from which the `y` (targets) orginate from. Used for reporting.
    dataset_description: str = ""
        A description of the dataset from which the `y` (targets) orginate from. Used for reporting.
    project_name: Optional[str]
        The name of the project. Used for reporting and saving to a directory (eg.: multiple scorecards)
    project_description: str = ""
        A description of the project. Used for reporting.
    dataset_type: Optional[Union[DatasetType, str]]
        Whether the task was a binar/multi-label classifiction of regression. If set to `None` it will infer from the target.
    sample_type: SampleTypes = SampleTypes.outofsample
        Whether we should evaluate it on insample or out of sample.

            - `SampleTypes.outofsample`
            - `SampleTypes.insample`
    default_metrics: Optional[List[Metric]]
        Default metrics that get evaluated. See `library`.
    custom_metrics: Optional[List[Metric]]
        Custom metrics that get evaluated. If specified it will evaluate these after `default_metric`
        See `library`.
    rolling_args : Dict[str, Any], optional
        Arguments to be passed onto `pd.DataFrame.rolling`.
        Default:

        - The window size of the rolling metric evaluation. If `None` evaluation over time will be on expanding window basis, by default `None`.
        - The step size of the rolling metric evaluation, by default `1`.

    Examples
    --------
    >>> from krisi import ScoreCard
    ... y_pred, y_true = [0, 2, 1, 3], [0, 1, 2, 3]
    ... sc = ScoreCard()
    ... sc.evaluate(y_pred, y_true, defaults=True) # Calculate predefined metrics
    ... sc["own_metric"] = (y_pred - y_true).mean() # Add a metric result directly
    ... sc.print()
    """

    y: TargetsDS
    predictions: PredictionsDS
    probabilities: Optional[ProbabilitiesDF]
    sample_weight: Optional[WeightsDS]
    sample_type: SampleTypes
    default_metrics_keys: List[str]
    custom_metrics_keys: List[str]
    dataset_type: DatasetType
    metadata: ScoreCardMetadata
    rolling_args: Dict[str, Any]

    def __init__(
        self,
        y: Targets,
        predictions: Predictions,
        probabilities: Optional[Probabilities] = None,
        sample_weight: Optional[Weights] = None,
        model_name: Optional[str] = None,
        model_description: str = "",
        dataset_name: Optional[str] = None,
        dataset_description: str = "",
        project_name: Optional[str] = None,
        project_description: str = "",
        dataset_type: Optional[Union[DatasetType, str]] = None,
        sample_type: Union[str, SampleTypes] = SampleTypes.outofsample,
        default_metrics: Optional[Union[List[Metric], Metric]] = None,
        custom_metrics: Optional[Union[List[Metric], Metric]] = None,
        rolling_args: Optional[Dict[str, Any]] = None,
        raise_exceptions: bool = False,
    ) -> None:
        if raise_exceptions:
            state = get_global_state()
            state.run_type = RunType.test
            set_global_state(state)

        sample_type = SampleTypes.from_str(sample_type)
        dataset_type = DatasetType.from_str(dataset_type) if dataset_type else None
        check_valid_pred_target(y, predictions, sample_weight)
        default_metrics = (
            wrap_in_list(default_metrics) if default_metrics is not None else None
        )
        custom_metrics = (
            wrap_in_list(custom_metrics) if custom_metrics is not None else None
        )

        check_no_duplicate_metrics(
            empty_if_none(default_metrics) + empty_if_none(custom_metrics)
        )
        self.__dict__["y"] = convert_to_series(y, "y")
        self.__dict__["predictions"] = convert_to_series(predictions, "predictions")
        self.__dict__["probabilities"] = (
            rename_probs_columns(ensure_df(probabilities, "probabilities"))
            if probabilities is not None
            else None
        )
        self.__dict__["sample_weight"] = (
            convert_to_series(
                sample_weight,  # pd.Series([1.0] * len(y))
                "sample_weight",
            )
            if sample_weight is not None
            else None
        )

        self.__dict__["sample_type"] = sample_type
        if rolling_args is None:
            window_size = len(y) // 20  # 5% of the data
            rolling_args = dict(
                window=window_size,
                step=window_size,
                min_periods=window_size,
                closed="left",
            )
            logger.info(f"rolling_args not specified, using default: {rolling_args}")

        self.__dict__["rolling_args"] = (
            rolling_args if rolling_args is not None else dict(window=len(y) // 100)
        )
        self.__dict__["dataset_type"] = (
            infer_dataset_type(y) if dataset_type is None else dataset_type
        )
        model_name_, dataset_name_, project_name_ = handle_unnamed(
            y, predictions, model_name, dataset_name, project_name
        )

        self.__dict__["metadata"] = ScoreCardMetadata(
            get_save_path(project_name_),
            project_name_,
            project_description,
            model_name_,
            model_description,
            dataset_name_,
            dataset_description,
        )

        if default_metrics is None:
            self.__dict__["dataset_type"] = infer_dataset_type(y)
            default_metrics = get_default_metrics_for_dataset_type(self.dataset_type)

        self.__dict__["default_metrics_keys"] = [
            metric.key for metric in default_metrics
        ]
        if custom_metrics is None:
            custom_metrics = []

        self.__dict__["custom_metrics_keys"] = [metric.key for metric in custom_metrics]

        for metric in default_metrics:
            self.__dict__[metric.key] = deepcopy(metric)

        for metric in custom_metrics:
            self.__dict__[metric.key] = deepcopy(metric)

    def __setitem__(self, key: str, item: Any) -> None:
        self.__setattr__(key, item)

    def __getitem__(self, key: Union[str, List[str]]) -> Union[ScoreCard, Metric]:
        if isinstance(key, List):
            scorecard_copy = deepcopy(self)
            all_keys = list(scorecard_copy.__dict__.keys())
            for k in all_keys:
                if k not in key:
                    if isinstance(scorecard_copy.__dict__[k], Metric):
                        del scorecard_copy.__dict__[k]

            return scorecard_copy
        elif isinstance(key, str):
            return deepcopy(getattr(self, key, Metric("Unknown Metric")))

    def __delitem__(self, key: str) -> None:
        del self[key]

    def __setattr__(self, key: str, item: Any) -> None:
        """
        Defines Dictionary like behaviour and ensures that a Metric can be
        added as a:

            - `Metric` object,
            - Dictionary,
            - Direct result (float, int or a List of float or int). Gets wrapped in a `Metric` object

        Parameters
        ----------
        key : string
            The key to which the object will be assigned to.
        item : Dictionary, Metric, Float, Int or List of Float or Int, or pd.Series
            The result that gets stored with the key. Depending on the type of object it will result in
            different behaviours:

                - If `Metric` or `Dict` it will store the object on `ScoreCard[key]`
                - If (`Float`, `Int`, `List of Float`, `List of Int`, `pd.Series`) it will check if a `Metric` on `key`
                already exists on `ScoreCard`. If yes, it will assign item to the result field of the
                existing `Metric`. If not it will wrap the item in a new `Metric` first then assign it to
                `ScoreCard[key]`.

        Examples
        --------
        >>> from krisi import ScoreCard
        ... sc = ScoreCard()
        ... sc['metric_result'] = 0.53 # Direct result assignment as a Dictionary
        Metric(result=0.53, key='metric_result', category=None, parameters=None, info="", ...)

        >>> sc.another_metric_result = 1 # Direct object assignment
        Metric(result=1, key='another_metric_result', category=None, parameters=None, info="", ...)

        >>> from krisi.evaluate.metric import Metric
        ... from krisi.evaluate.type import MetricCategories
        ... sc.full_metric = Metric("My own metric", category=MetricCategories.class_err, info="A fictious metric with metadata", func: lambda y, y_hat: (y - y_hat)/2)
        Metric("My own metric", key="my_own_metric", category=MetricCategories.class_err, info="A fictious metric with metadata", ...)

        >>> sc.metric_as_dictionary = {name: "My other metric", info: "A Metric created with a dictionary", func: lambda y, y_hat: y - y_hat}
        Metric("My other metric", key="my_other_metric", info="A Metric created with a dictionary", ...)

        """
        metric = getattr(self, key, None)

        if metric is None:
            if isinstance(item, dict):
                self.__dict__[key] = Metric(key=key, **item)
            elif isinstance(item, Metric):
                item["key"] = key
                self.__dict__[key] = item
            else:
                self.__dict__[key] = Metric(
                    name=key, key=key, result=item, category=MetricCategories.unknown
                )
        else:
            if isinstance(item, dict):
                metric_dict = map_newdict_on_olddict(
                    strip_builtin_functions(vars(metric)), item, exclude=["name"]
                )
                self.__dict__[key] = Metric(**metric_dict)
            elif isinstance(item, Metric):
                metric_dict = map_newdict_on_olddict(
                    strip_builtin_functions(vars(metric)),
                    strip_builtin_functions(vars(item)),
                    exclude=["name"],
                )
                self.__dict__[key] = Metric(**metric_dict)
            else:
                metric["result"] = item
                self.__dict__[key] = metric

    def get_ds(self, name_as_index: bool = False) -> pd.Series:
        """
        Returns a `pd.Series` where each index is the name of a `Metric` and
        the value is the corresponding `result`

        Returns
        -------
        pd.Series
        """
        metrics = [
            metric
            for metric in self.get_all_metrics()
            if not isinstance(metric.result, Iterable)
        ]

        return pd.Series(
            [metric["result"] for metric in metrics],
            index=[metric.name if name_as_index else metric.key for metric in metrics],
        )

    def get_default_metrics(self) -> List[Metric]:
        """
        Returns a List of Predefined Metrics according to task type:
        regression, classification, multi-label classification.

        Returns
        -------
        List of Metrics
        """
        return [
            self.__dict__[key]
            for key in self.default_metrics_keys
            if key in self.__dict__
        ]

    def get_custom_metrics(self) -> List[Metric]:
        """
        Returns a List of Custom ``Metric``s defined by the user on initalization
        of the ``ScoreCard``

        Returns
        -------
        List of Metrics
        """
        predifined_custom_metrics = [
            self.__dict__[key]
            for key in self.custom_metrics_keys
            if key in self.__dict__
        ]
        modified_custom_metrics = [
            value
            for key, value in self.__dict__.items()
            if key not in self.custom_metrics_keys + self.default_metrics_keys
            and hasattr(value, "key")
        ]

        return predifined_custom_metrics + modified_custom_metrics

    def get_all_metrics(
        self, defaults: bool = True, only_evaluated: bool = False
    ) -> List[Metric]:
        """
        Helper function that returns both `default_metrics` and `custom_metrics`.

        Parameters
        -------
        only_evaluated: bool
            Only return `Metric`s that were already evaluated
        defauls: bool
            Whether `default_metrics` should be return or not.

        Returns
        -------
        List of Metrics
        """
        if defaults:
            metrics = self.get_default_metrics() + self.get_custom_metrics()
        else:
            metrics = self.get_custom_metrics()

        if only_evaluated:
            return [metric for metric in metrics if metric.is_evaluated()]
        else:
            return metrics

    def __evaluate(
        self,
        func_key_evaluate: Literal["evaluate", "evaluate_over_time"],
        result_key: Literal["result", "result_rolling"],
        defaults: bool = True,
        rolling_args: Optional[Dict[str, Any]] = dict(),
    ):
        for metric in self.get_all_metrics(defaults=defaults):
            if metric.restrict_to_sample is not self.sample_type:
                getattr(metric, func_key_evaluate)(
                    self.y,
                    self.predictions,
                    self.probabilities,
                    self.sample_weight,
                    **rolling_args,
                )

    def evaluate_rolling_properties(self):
        for metric in self.get_all_metrics(defaults=True):
            metric.evaluate_rolling_properties()

    def evaluate(self, defaults: bool = True) -> None:
        """
        Evaluates `Metric`s present on the `ScoreCard`

        Parameters
        ----------
        defaults: boolean
            Wether the default `Metric`s should be evaluated or not.

        Returns
        -------
        None
        """
        self.__evaluate("evaluate", "result", defaults=defaults)

    def evaluate_over_time(self, defaults: bool = True) -> None:
        """
        Evaluates `Metric`s present on the `ScoreCard` over time, either with expanding
        or fixed sized window. Assigns list of results to `results_over_time`.

        Parameters
        ----------
        y: Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
            The true labels to compare values to.
        predictions: Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
            The predicted values. Integers or whole floats if classification, else floats.
        defaults: bool
            Wether the default ``Metric``s should be evaluated or not. Default value = True
        window: int
            Size of window. If number is provided then evaluation happens on a fixed window size,
            otherwise it evaluates it on an expanding window basis.

        Returns
        -------
        None
        """
        self.__evaluate(
            "evaluate_over_time",
            "result_rolling",
            defaults=defaults,
            rolling_args={"rolling_args": self.rolling_args},
        )
        self.evaluate_rolling_properties()

    def print(
        self,
        mode: Union[str, PrintMode, List[PrintMode], List[str]] = PrintMode.extended,
        with_info: bool = False,
        with_parameters: bool = True,
        with_diagnostics: bool = False,
        input_analysis: bool = True,
        title: Optional[str] = None,
        frame_or_series: bool = True,
    ) -> None:
        """
        Prints the ScoreCard to the console.

        Parameters
        ----------
        mode: Union[str, PrintMode, List[PrintMode], List[str]] = PrintMode.extended
            - `PrintMode.extended` or `'extended'` prints the full ScoreCard, with targets, predictions, residuals, etc.
            - `PrintMode.minimal` or `'minimal'` prints the name and the matching result of each metric in the ScoreCard, without fancy formatting
            - `PrintMode.minimal_table` or `'minimal_table'` creates a table format of just the metric name and the accompanying result
        with_info: bool
            Wether descriptions of each metrics should be printed or not
        input_analysis: bool = True
            Wether it should print analysis about the raw `targets` and `predictions`
        title: Optional[str] = None
            Title of the table when mode = `'minimal_table'`
        """

        modes = [PrintMode.from_str(mode_) for mode_ in wrap_in_list(mode)]
        if title is None:
            title = self.metadata.project_name
        for mode in modes:
            if mode is PrintMode.extended:
                to_display = get_summary(
                    self,
                    repr=True,
                    categories=[el.value for el in MetricCategories],
                    with_info=with_info,
                    with_parameters=with_parameters,
                    with_diagnostics=with_diagnostics,
                    input_analysis=input_analysis,
                )

            elif mode is PrintMode.minimal:
                to_display = get_minimal_summary(self, dataframe=frame_or_series)
            elif mode is PrintMode.minimal_table:
                to_display = get_large_metric_summary(
                    self,
                    title,
                    with_info=with_info,
                    with_parameters=with_parameters,
                    with_diagnostics=with_diagnostics,
                )
            if isinstance(to_display, (pd.DataFrame, pd.Series)) and is_notebook():
                from IPython.display import display

                display(to_display)
            else:
                print(to_display)

    def cleanup_group(self) -> None:
        for metric in self.get_all_metrics(defaults=True):
            if isinstance(metric, Group):
                group = metric
                for metric_in_group in group.get_all_metrics():
                    if metric_in_group.key not in self.__dict__:
                        metric_in_group._from_group = True
                        self[metric_in_group.key] = metric_in_group
                    else:
                        self[metric_in_group.key] = {
                            f"result_{group.key}": metric_in_group.__dict__["result"],
                            f"result_rolling_{group.key}": metric_in_group.__dict__[
                                "result_rolling"
                            ],
                            "_from_group": True,
                        }

    def save(
        self,
        with_info: bool = False,
        with_parameters: bool = True,
        with_diagnostics: bool = False,
        save_modes: List[Union[SaveModes, str]] = [
            SaveModes.minimal,
            SaveModes.obj,
            SaveModes.text,
        ],
        override_base_path: Optional[Path] = None,
        timestamp_no_overwrite: bool = False,
    ) -> ScoreCard:
        if override_base_path is None:
            if timestamp_no_overwrite is False:
                dir_model_name = self.metadata.model_name
            else:
                dir_model_name = datetime.datetime.now().strftime("%H-%M-%S-%f")

        path = replace_if_none(
            override_base_path,
            os.path.join(
                self.metadata.save_path,
                os.path.join(
                    PathConst.default_save_output_path,
                    Path(dir_model_name),
                ),
            ),
        )
        ensure_path(path)

        if SaveModes.minimal in save_modes or SaveModes.minimal.value in save_modes:
            save_minimal_summary(self, path)
        if SaveModes.obj in save_modes or SaveModes.obj.value in save_modes:
            save_object(self, path)
        save_console(
            self,
            path,
            with_info,
            with_parameters=with_parameters,
            with_diagnostics=with_diagnostics,
            save_modes=save_modes,
        )

        return self

    def get_diagram_dictionary(self) -> Dict[str, InteractiveFigure]:
        diagrams = flatten(
            remove_nans([metric.get_diagrams() for metric in self.get_all_metrics()])
        )

        return {diagram.id: diagram for diagram in diagrams}

    def generate_report(
        self,
        display_modes: Union[str, List[str], DisplayModes, List[DisplayModes]] = [
            DisplayModes.interactive
        ],
        html_template_url: Path = PathConst.html_report_template_url,
        css_template_url: Path = PathConst.css_report_template_url,
        author: str = "",
        report_title: Optional[str] = None,
        override_base_path: Optional[Path] = None,
    ) -> None:
        save_path = replace_if_none(override_base_path, self.metadata.save_path)
        ensure_path(save_path)

        display_modes = [
            DisplayModes.from_str(mode) for mode in wrap_in_list(display_modes)
        ]
        report = create_report_from_scorecard(
            self,
            display_modes,
            html_template_url,
            css_template_url,
            author=author,
            report_title=replace_if_none(
                report_title, f"Report on {self.metadata.project_name}"
            ),
            save_path=save_path,
        )
        report.generate_launch()

    def __repr__(self):
        md = self.__dict__["metadata"]
        return (
            super().__repr__()[:-1]
            + f"\n{md.model_name:>40s} | {md.dataset_name} \n{md.project_name:>40s} | {self.dataset_type.value}\n\n{get_minimal_summary(self, dataframe=False)}>"
        )

    def __union(
        self,
        other: ScoreCard,
        function: Callable,
    ) -> ScoreCard:
        copied_scorecard = deepcopy(self)

        for key, value in copied_scorecard.__dict__.items():
            if isinstance(value, Metric) and key in other.__dict__.keys():
                if value.result is not None and other[key].result is not None:
                    copied_scorecard.__dict__[key].__dict__["result"] = function(
                        value.result, other[key].result
                    )
                    copied_scorecard.__dict__[key].__dict__["result_rolling"] = None
                else:
                    copied_scorecard.__dict__[key].__dict__["result"] = None
                    copied_scorecard.__dict__[key].__dict__["result_rolling"] = None

        return copied_scorecard

    def subtract(self, other: ScoreCard) -> ScoreCard:
        return self.__union(other, lambda x, y: x - y)

    def subtract_abs(self, other: ScoreCard) -> ScoreCard:
        return self.__union(other, lambda x, y: abs(x - y))

    def add(self, other: ScoreCard) -> ScoreCard:
        return self.__union(other, lambda x, y: x + y)

    def multiply(self, other: ScoreCard) -> ScoreCard:
        return self.__union(other, lambda x, y: x * y)

    def divide(self, other: ScoreCard) -> ScoreCard:
        return self.__union(other, lambda x, y: x / y)


def get_rolling_diagrams(obj: ScoreCard) -> List[List[InteractiveFigure]]:
    return [
        diagram
        for diagram in [
            metric.get_diagram_over_time() for metric in obj.get_all_metrics()
        ]
        if diagram is not None
    ]
