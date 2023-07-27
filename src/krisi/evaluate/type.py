from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

from krisi.utils.enums import ParsableEnum

MetricResult = TypeVar(
    "MetricResult", bound=Union[float, int, str, List, Tuple, pd.DataFrame, pd.Series]
)


Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
Probabilities = Union[np.ndarray, pd.Series, pd.DataFrame, List[Union[int, float]]]
Weights = Union[np.ndarray, pd.Series, List[Union[int, float]]]
PredictionsDS = pd.Series
ProbabilitiesDF = pd.DataFrame
TargetsDS = pd.Series
WeightsDS = pd.Series
MetricFunctionProb = Callable[[PredictionsDS, TargetsDS, ProbabilitiesDF], MetricResult]
MetricFunctionProbWeights = Callable[
    [PredictionsDS, TargetsDS, ProbabilitiesDF, WeightsDS], MetricResult
]
MetricFunctionNoProb = Callable[[PredictionsDS, TargetsDS], MetricResult]
MetricFunctionNoProbWeights = Callable[
    [PredictionsDS, TargetsDS, WeightsDS], MetricResult
]
MetricFunction = Union[
    MetricFunctionNoProb,
    MetricFunctionProb,
    MetricFunctionProbWeights,
    MetricFunctionNoProbWeights,
]


class MetricCategories(ParsableEnum):
    residual = "Residual Diagnostics"
    entropy = "Information Entropy"
    class_err = "Forecast Errors - Classification"
    reg_err = "Forecast Errors - Regression"
    stats = "General Statistics"
    unknown = "Unknown"


class SampleTypes(ParsableEnum):
    insample = "insample"
    validation = "validation"
    outofsample = "outofsample"


class Calculation(ParsableEnum):
    single = "single"
    rolling = "rolling"
    both = "both"


class SaveModes(ParsableEnum):
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


class ComputationalComplexity(ParsableEnum):
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


class PrintMode(ParsableEnum):
    extended = "extended"
    minimal = "minimal"
    minimal_table = "minimal_table"


class NamingPrefixes:
    model = "Model_"
    dataset = "Dataset_"
    project = "Project_"


class DatasetType(ParsableEnum):
    regression = "regression"
    classification_binary_balanced = "classification_binary_balanced"
    classification_binary_imbalanced = "classification_binary_imbalanced"
    classification_multiclass = "classification_multiclass"

    def is_classification(self):
        return self in [
            DatasetType.classification_binary_balanced,
            DatasetType.classification_binary_imbalanced,
            DatasetType.classification_multiclass,
        ]


class Purpose(ParsableEnum):
    objective = "objective"
    loss = "loss"
    diagram = "diagram"
    group = "group"
