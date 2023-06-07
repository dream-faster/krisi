from __future__ import annotations

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
Weights = Union[np.ndarray, pd.Series, pd.DataFrame, List[Union[int, float]]]
PredictionsDS = pd.Series
ProbabilitiesDF = pd.DataFrame
TargetsDS = pd.Series
WeightsDS = pd.Series
MetricFunctionProb = Callable[[PredictionsDS, TargetsDS, ProbabilitiesDF], MetricResult]
MetricFunctionProbWeights = Callable[
    [PredictionsDS, TargetsDS, ProbabilitiesDF, WeightsDS], MetricResult
]
MetricFunctionNoProb = Callable[
    [PredictionsDS, TargetsDS, ProbabilitiesDF], MetricResult
]
MetricFunctionNoProbWeights = Callable[
    [PredictionsDS, TargetsDS, ProbabilitiesDF, WeightsDS], MetricResult
]
MetricFunction = Union[
    MetricFunctionNoProb,
    MetricFunctionProb,
    MetricFunctionProbWeights,
    MetricFunctionNoProbWeights,
]


class MetricCategories(Enum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    stats = "General Statistics"
    unknown = "Unknown"


class SampleTypes(Enum):
    insample = "insample"
    outofsample = "outofsample"


class Calculation(Enum):
    single = "single"
    rolling = "rolling"
    both = "both"

    @staticmethod
    def from_str(value: Union[str, Calculation]) -> Calculation:
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
    def from_str(value: Union[str, PrintMode]) -> PrintMode:
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

    @staticmethod
    def from_str(value: Union[str, DatasetType]) -> DatasetType:
        if isinstance(value, DatasetType):
            return value
        for strategy in DatasetType:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown DatasetType: {value}")

    def is_classification(self):
        return self in [
            DatasetType.classification_binary,
            DatasetType.classification_multiclass,
        ]
