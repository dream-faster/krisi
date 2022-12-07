from dataclasses import dataclass
from enum import Enum
from typing import Optional, Any


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """
    ljung_box_score: Optional[int] = None

    def update(self, key: str, value: Any) -> None:
        if hasattr(self, key):
            setattr(self, key, value)
