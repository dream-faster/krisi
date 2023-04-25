from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, List, Tuple, TypeVar, Union

import numpy as np
import pandas as pd

MetricResult = TypeVar(
    "MetricResult", bound=Union[float, int, str, List, Tuple, pd.DataFrame]
)


Predictions = Union[np.ndarray, pd.Series, List[Union[int, float]]]
Targets = Union[np.ndarray, pd.Series, List[Union[int, float]]]
PredictionsDS = pd.Series
TargetsDS = pd.Series
MetricFunction = Callable[[PredictionsDS, TargetsDS], MetricResult]


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


class SaveModes(Enum):
    svg = "svg"
    html = "html"
    text = "text"
    obj = "obj"
    minimal = "minimal"


class PathConst:
    default_eval_output_path: Path = Path("output/")
    html_report_template_url: Path = Path("library/pdf_layouts/scorecard/report.html")
    css_report_template_url: Path = Path("library/pdf_layouts/scorecard/report.css")


class ComputationalComplexity(Enum):
    low = "low"
    medium = "medium"
    high = "high"


@dataclass
class ScoreCardMetadata:
    project_name: str = ""
    project_description: str = ""
    model_name: str = ""
    model_description: str = ""
    dataset_name: str = ""
    dataset_description: str = ""


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
