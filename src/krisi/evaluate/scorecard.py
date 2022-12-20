from copy import deepcopy
from dataclasses import dataclass
from typing import Any, List

from rich import print
from rich.pretty import Pretty

from krisi.evaluate.library.default_metrics import predefined_default_metrics
from krisi.evaluate.metric import Metric, MetricCategories
from krisi.evaluate.type import SampleTypes
from krisi.utils.iterable_helpers import map_newdict_on_olddict
from krisi.utils.printing import get_summary


def strip_builtin_functions(dict_to_strip: dict) -> dict:
    return {key_: value_ for key_, value_ in dict_to_strip.items() if key_[:2] != "__"}


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes
    default_metrics_keys: List[str]

    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        sample_type: SampleTypes,
        default_metrics: List[Metric] = predefined_default_metrics,
    ) -> None:
        self.__dict__["model_name"] = model_name
        self.__dict__["dataset_name"] = dataset_name
        self.__dict__["sample_type"] = sample_type
        self.__dict__["default_metrics_keys"] = [
            metric.key for metric in default_metrics
        ]

        for metric in default_metrics:
            self.__dict__[metric.key] = deepcopy(metric)

    def __setattr__(self, key: str, item: Any) -> None:
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
        return [self.__dict__[key] for key in self.default_metrics_keys]

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

    def print_summary(self, with_info: bool = False) -> "ScoreCard":
        print(
            get_summary(
                self,
                repr=True,
                categories=[el.value for el in MetricCategories],
                with_info=with_info,
            )
        )
        return self
