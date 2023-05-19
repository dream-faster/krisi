import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

MetricResult = TypeVar(
    "MetricResult", bound=Union[float, int, str, List, Tuple, pd.DataFrame, pd.Series]
)


Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
Probabilities = Union[np.ndarray, pd.Series, pd.DataFrame, List[Union[int, float]]]
PredictionsDS = pd.Series
ProbabilitiesDF = pd.DataFrame
TargetsDS = pd.Series
MetricFunctionProb = Callable[[PredictionsDS, TargetsDS, ProbabilitiesDF], MetricResult]
MetricFunctionNoProb = Callable[
    [PredictionsDS, TargetsDS, ProbabilitiesDF], MetricResult
]
MetricFunction = Union[MetricFunctionNoProb, MetricFunctionProb]


class MetricCategories(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    unknown = "Unknown"


class SampleTypes(Enum):
    insample = "insample"
    outofsample = "outofsample"


class Calculation(Enum):
    single = "single"
    rolling = "rolling"
    both = "both"

    @staticmethod
    def from_str(value: Union[str, "Calculation"]) -> "Calculation":
        if isinstance(value, Calculation):
            return value
        for calc in Calculation:
            if calc.value == value:
                return calc
        else:
            raise ValueError(f"Unknown Calculation: {value}")


class SaveModes(Enum):
    svg = "svg"
    html = "html"
    text = "text"
    obj = "obj"
    minimal = "minimal"


class PathConst:
    default_eval_output_path: Path = Path("output/")
    default_analyse_output_path: Path = Path("analyse/")
    default_save_output_path: Path = Path("scorecards/")
    html_report_template_url: Path = Path("library/pdf_layouts/scorecard/report.html")
    css_report_template_url: Path = Path("library/pdf_layouts/scorecard/report.css")


class ComputationalComplexity(Enum):
    low = "low"
    medium = "medium"
    high = "high"


@dataclass
class ScoreCardMetadata:
    save_path: Path
    project_name: str = ""
    project_description: str = ""
    model_name: str = ""
    model_description: str = ""
    dataset_name: str = ""
    dataset_description: str = ""

    def __setattr__(self, key: str, item: Any) -> None:
        if key == "project_name":
            self.save_path = Path(
                os.path.join(PathConst.default_eval_output_path, item)
            )

        super().__setattr__(key, item)


class PrintMode(Enum):
    extended = "extended"
    minimal = "minimal"
    minimal_table = "minimal_table"

    @staticmethod
    def from_str(value: Union[str, "PrintMode"]) -> "PrintMode":
        if isinstance(value, PrintMode):
            return value
        for strategy in PrintMode:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown PrintMode: {value}")


class NamingPrefixes:
    model = "Model_"
    dataset = "Dataset_"
    project = "Project_"


class DatasetType(Enum):
    regression = "regression"
    classification_binary = "classification_binary"
    classification_multiclass = "classification_multiclass"
