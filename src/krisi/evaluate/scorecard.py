import os
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any, List, Optional, Tuple

from krisi.utils.printing import bold


class SampleTypes(Enum):
    insample = "insample"
    outsample = "outsample"


def iterative_length(obj: Iterable) -> List[int]:
    object_shape = []
    num_obj = 0
    for el in obj:
        if isinstance(el, Iterable) and not isinstance(el, str):
            object_shape.append(iterative_length(el))
        else:
            num_obj += 1
    if num_obj > 0:
        object_shape.append(num_obj)
    return object_shape


@dataclass
class ScoreCard:
    """Default Identifiers"""

    model_name: str
    dataset_name: str
    sample_type: SampleTypes

    """ Score Metrics """

    ljung_box_score: Optional[float] = None
    hearst_exponent: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    pacf_res: Optional[Tuple[float, float]] = None
    acf_res: Optional[Tuple[float, float]] = None

    def __init__(self, model_name: str, dataset_name: str, sample_type: SampleTypes):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.sample_type = sample_type

    def __setitem__(self, key: str, item: Any) -> None:
        setattr(self, key, item)

    def __getitem__(self, key: str):
        return getattr(self, key, "Unknown metric")

    def __delitem__(self, key: str) -> None:
        setattr(self, key, None)

    def __str__(self) -> str:
        title = f"\n\nResult of {bold(self.model_name)} on {bold(self.dataset_name)} tested on {bold(self.sample_type.value)}"
        title += f"\n{'―'*len(title)}"
        return (
            f'{" "*4}\n'.join(
                [title]
                + [
                    f"{str(key):>15s}: {f'Iterable of shape: {str(iterative_length(value))}' if isinstance(value, Iterable) and not isinstance(value, str) else str(value):<15s}"
                    for key, value in vars(self).items()
                    if value is not None
                ]
            )
            + "\n\n"
        )
