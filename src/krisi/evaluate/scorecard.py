from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


@dataclass
class ScoreCard(dict):
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """
    ljung_box_score: Optional[float] = None
    hearst_exponent: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None

    def update(self, key: str, value: Any) -> None:
        # if hasattr(self, key):
        setattr(self, key, value)


