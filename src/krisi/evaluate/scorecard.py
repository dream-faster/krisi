import datetime
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List, Optional, Union

from rich import print
from rich.layout import Layout
from rich.panel import Panel
from rich.pretty import Pretty

from krisi.evaluate.library.default_metrics_classification import (
    predefined_classification_metrics,
)
from krisi.evaluate.library.default_metrics_regression import (
    predefined_regression_metrics,
)
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import (
    MetricCategories,
    PathConst,
    Predictions,
    SampleTypes,
    SaveModes,
    Targets,
)
from krisi.report.type import InteractiveFigure
from krisi.utils.iterable_helpers import map_newdict_on_olddict, strip_builtin_functions
from krisi.utils.printing import (
    get_minimal_summary,
    get_summary,
    save_console,
    save_minimal_summary,
    save_object,
)


@dataclass
class ScoreCard:
    """ScoreCard Object.

    Krisi's main object holding and evaluating metrics.
    Stores, evaluates and generates vizualisations of predefined
    and custom metrics for regression and classification.

    Examples
    --------
    >>> import numpy as np
    ... from krisi.evaluate import SampleTypes, ScoreCard
    ... y_pred = [0, 2, 1, 3]
    ... y_true = [0, 1, 2, 3]
    ... sc.evaluate(y_pred, y_true, defaults=True) # Calculate predefined metrics
    ... accuracy_score(y_true, y_pred, normalize=False)
    ... sc["own_metric"] = (target - predictions).mean() # Add a metric result directly
    ... sc.print_summary(extended=True)
    """

    model_name: Optional[str]
    dataset_name: Optional[str]
    project_name: Optional[str]
    sample_type: SampleTypes
    default_metrics_keys: List[str]
    custom_metrics_keys: List[str]
    classification: Optional[bool]  # TODO: Support multilabel classification

    def __init__(
        self,
        model_name: str,
        dataset_name: Optional[str] = None,
        project_name: Optional[str] = None,
        classification: Optional[bool] = None,
        sample_type: SampleTypes = SampleTypes.outofsample,
        default_metrics: List[Metric] = [],
        custom_metrics: List[Metric] = [],
    ) -> None:
        self.__dict__["model_name"] = model_name
        self.__dict__["dataset_name"] = dataset_name
        self.__dict__["project_name"] = project_name
        self.__dict__["sample_type"] = sample_type

        if classification:
            default_metrics = predefined_classification_metrics
        else:
            default_metrics = predefined_regression_metrics

        self.__dict__["default_metrics_keys"] = [
            metric.key for metric in default_metrics
        ]
        self.__dict__["custom_metrics_keys"] = [metric.key for metric in custom_metrics]

        for metric in default_metrics:
            self.__dict__[metric.key] = deepcopy(metric)
        for metric in custom_metrics:
            self.__dict__[metric.key] = deepcopy(metric)

    def __setattr__(self, key: str, item: Any) -> None:
        """Defines Dictionary like behaviour and ensures that a Metric can be
        added as a Metric object, a Dictionary, or as a float, int or a List of float or int

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

    def evaluate(
        self, y: Targets, predictions: Predictions, defaults: bool = True
    ) -> "ScoreCard":
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
                metric.evaluate(y, predictions)

        return self

    def evaluate_over_time(
        self,
        y: Targets,
        predictions: Predictions,
        defaults: bool = True,
        window: Optional[int] = None,
    ) -> "ScoreCard":
        for metric in self.get_all_metrics(defaults=defaults):
            if metric.restrict_to_sample is not self.sample_type:
                metric.evaluate_over_time(y, predictions, window=window)
        return self

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

    def print_summary(
        self, with_info: bool = False, extended: bool = True
    ) -> "ScoreCard":
        if extended:
            summary = get_summary(
                self,
                repr=True,
                categories=[el.value for el in MetricCategories],
                with_info=with_info,
            )
        else:
            summary = get_minimal_summary(self)

        print(summary)
        return self

    def save(
        self,
        path: str = PathConst.default_eval_output_path,
        with_info: bool = False,
        save_modes: List[Union[SaveModes, str]] = [
            SaveModes.minimal,
            SaveModes.obj,
            SaveModes.text,
        ],
    ) -> "ScoreCard":
        if self.project_name:
            path += f"{self.project_name}/"
        path += f"{datetime.datetime.now().strftime('%H:%M:%S')}_{self.model_name}_{self.dataset_name}"
        import os

        if not os.path.exists(path):
            os.makedirs(path)

        if SaveModes.minimal in save_modes or SaveModes.minimal.value in save_modes:
            save_minimal_summary(self, path)
        if SaveModes.obj in save_modes or SaveModes.minimal.obj in save_modes:
            save_object(self, path)
        save_console(self, path, with_info, save_modes)

        return self

    def get_rolling_diagrams(self) -> List[InteractiveFigure]:
        return [
            diagram
            for diagram in [
                metric.get_diagram_over_time() for metric in self.get_all_metrics()
            ]
            if diagram is not None
        ]
