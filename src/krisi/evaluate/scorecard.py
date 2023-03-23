import datetime
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import pandas as pd
from rich import print
from rich.pretty import Pretty

from krisi.evaluate.assertions import is_dataset_classification_like
from krisi.evaluate.library.default_metrics_classification import (
    all_classification_metrics,
)
from krisi.evaluate.library.default_metrics_regression import all_regression_metrics
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import (
    MetricCategories,
    PathConst,
    Predictions,
    SampleTypes,
    SaveModes,
    ScoreCardMetadata,
    Targets,
)
from krisi.evaluate.utils import handle_unnamed
from krisi.report.console import get_minimal_summary, get_summary
from krisi.report.report import create_report_from_scorecard
from krisi.report.type import DisplayModes, InteractiveFigure
from krisi.utils.io import save_console, save_minimal_summary, save_object
from krisi.utils.iterable_helpers import (
    flatten,
    map_newdict_on_olddict,
    remove_nans,
    strip_builtin_functions,
)


@dataclass
class ScoreCard:
    """ScoreCard Object.

    Krisi's main object holding and evaluating metrics.
    Stores, evaluates and generates vizualisations of predefined
    and custom metrics for regression and classification.

    Examples
    --------
    >>> from krisi import ScoreCard
    ... y_pred, y_true = [0, 2, 1, 3], [0, 1, 2, 3]
    ... sc = ScoreCard()
    ... sc.evaluate(y_pred, y_true, defaults=True) # Calculate predefined metrics
    ... sc["own_metric"] = (y_pred - y_true).mean() # Add a metric result directly
    ... sc.print_summary(extended=True)
    """

    y: Targets
    predictions: Predictions
    sample_type: SampleTypes
    default_metrics_keys: List[str]
    custom_metrics_keys: List[str]
    classification: bool  # TODO: Support multilabel classification
    metadata: ScoreCardMetadata

    def __init__(
        self,
        y: Targets,
        predictions: Predictions,
        model_name: Optional[str] = None,
        model_description: str = "",
        dataset_name: Optional[str] = None,
        dataset_description: str = "",
        project_name: Optional[str] = None,
        project_description: str = "",
        classification: Optional[bool] = None,
        sample_type: SampleTypes = SampleTypes.outofsample,
        default_metrics: Optional[List[Metric]] = None,
        custom_metrics: Optional[List[Metric]] = None,
    ) -> None:
        self.__dict__["y"] = y
        self.__dict__["predictions"] = predictions
        self.__dict__["sample_type"] = sample_type
        self.__dict__["classification"] = (
            is_dataset_classification_like(y)
            if classification is None
            else classification
        )
        model_name_, dataset_name_, project_name_ = handle_unnamed(
            y, predictions, model_name, dataset_name, project_name
        )

        self.__dict__["metadata"] = ScoreCardMetadata(
            project_name_,
            project_description,
            model_name_,
            model_description,
            dataset_name_,
            dataset_description,
        )

        if default_metrics is None:
            default_metrics = (
                all_classification_metrics
                if self.classification
                else all_regression_metrics
            )

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

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key, "Unknown metric")

    def __delitem__(self, key: str) -> None:
        setattr(self, key, None)

    def __str__(self) -> str:
        print(Pretty(self.__dict__))
        return ""

    def __repr__(self) -> str:
        print(Pretty(self.__dict__))
        return ""

    def __setattr__(self, key: str, item: Any) -> None:
        """Defines Dictionary like behaviour and ensures that a Metric can be
        added as a
            - Metric object,
            - Dictionary,
            - Direct result (float, int or a List of float or int). Gets wrapped in a ``Metric`` object

        See examples for behaviour.

        Parameters
        ----------
        key : string
            The key to which the object will be assign to on this object

        item : Dictionary, Metric, Float, Int or List of Float or Int, or pd.Series
            Always ignored, exists for compatibility.

        Returns
        -------
        None

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
        ... sc.full_metric = Metric("My own metric", category=MetricCategories.class_err, info="A fictious metric with metadata")
        Metric("My own metric", key="my_own_metric", func: lambda y, y_hat: (y - y_hat)/2, category=MetricCategories.class_err, info="A fictious metric with metadata", ...)

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

    def get_ds(self) -> pd.Series:
        metrics = [
            metric
            for metric in self.get_all_metrics()
            if not isinstance(metric.result, Iterable)
        ]

        return pd.Series(
            [metric.result for metric in metrics],
            index=[metric.name for metric in metrics],
        )

    def get_default_metrics(self) -> List[Metric]:
        """Returns a List of Predefined Metrics according to task type:
        regression, classification, multi-label classification.


        Returns
        -------
        List of Metrics
        """
        return [self.__dict__[key] for key in self.default_metrics_keys]

    def get_custom_metrics(self) -> List[Metric]:
        """Returns a List of Custom ``Metric``s defined by the user on initalization
        of the ``ScoreCard``

        Returns
        -------
        List of Metrics
        """
        predifined_custom_metrics = [
            self.__dict__[key] for key in self.custom_metrics_keys
        ]
        modified_custom_metrics = [
            value
            for key, value in self.__dict__.items()
            if key not in self.custom_metrics_keys + self.default_metrics_keys
            and hasattr(value, "key")
        ]

        return predifined_custom_metrics + modified_custom_metrics

    def get_all_metrics(self, defaults: bool = True) -> List[Metric]:
        if defaults:
            return self.get_default_metrics() + self.get_custom_metrics()
        else:
            return self.get_custom_metrics()

    def evaluate(self, defaults: bool = True) -> "ScoreCard":
        """Evaluates ``Metric``s present on the ``ScoreCard``

        Parameters
        ----------
        y: Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
        The true labels to compare values to

        predictions: Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
        The predicted values. Integers or whole floats if classification, else floats.

        defaults: boolean
        Wether the default ``Metric``s should be evaluated or not. Default value = True

        Returns
        -------
        self: ScoreCard
        """

        for metric in self.get_all_metrics(defaults=defaults):
            if metric.restrict_to_sample is not self.sample_type:
                metric.evaluate(self.y, self.predictions)

        return self

    def evaluate_over_time(
        self,
        defaults: bool = True,
        window: Optional[int] = None,
    ) -> "ScoreCard":
        """Evaluates ``Metric``s present on the ``ScoreCard`` over time, either with expanding
        or fixed sized window. Assigns list of results to ``results_over_time``.

        Parameters
        ----------
        y: Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
        The true labels to compare values to

        predictions: Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
        The predicted values. Integers or whole floats if classification, else floats.

        defaults: boolean
        Wether the default ``Metric``s should be evaluated or not. Default value = True

        window: int - Optional
        Size of window. If number is provided then evaluation happens on a fixed window size,
        otherwise it evaluates it on an expanding window basis.

        Returns
        -------
        self: ScoreCard
        """
        for metric in self.get_all_metrics(defaults=defaults):
            if metric.restrict_to_sample is not self.sample_type:
                metric.evaluate_over_time(self.y, self.predictions, window=window)
        return self

    def print_summary(
        self,
        with_info: bool = False,
        extended: bool = True,
        input_analysis: bool = True,
    ) -> None:
        if extended:
            summary = get_summary(
                self,
                repr=True,
                categories=[el.value for el in MetricCategories],
                with_info=with_info,
                input_analysis=input_analysis,
            )
        else:
            summary = get_minimal_summary(self)

        print(summary)

    def save(
        self,
        path: Path = PathConst.default_eval_output_path,
        with_info: bool = False,
        save_modes: List[Union[SaveModes, str]] = [
            SaveModes.minimal,
            SaveModes.obj,
            SaveModes.text,
        ],
    ) -> "ScoreCard":
        import os

        if self.metadata.project_name:
            path = Path(os.path.join(path, Path(f"{self.metadata.project_name}")))
        path = Path(
            os.path.join(
                path,
                Path(
                    f"{datetime.datetime.now().strftime('%H-%M-%S')}_{self.metadata.model_name}_{self.metadata.dataset_name}"
                ),
            )
        )

        if not os.path.exists(path):
            os.makedirs(path)

        if SaveModes.minimal in save_modes or SaveModes.minimal.value in save_modes:
            save_minimal_summary(self, path)
        if SaveModes.obj in save_modes or SaveModes.obj.value in save_modes:
            save_object(self, path)
        save_console(self, path, with_info, save_modes)

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
    ) -> None:
        report = create_report_from_scorecard(
            self,
            display_modes,
            html_template_url,
            css_template_url,
            author=author,
        )
        report.generate_launch()


def get_rolling_diagrams(obj: "ScoreCard") -> List[List[InteractiveFigure]]:
    return [
        diagram
        for diagram in [
            metric.get_diagram_over_time() for metric in obj.get_all_metrics()
        ]
        if diagram is not None
    ]
