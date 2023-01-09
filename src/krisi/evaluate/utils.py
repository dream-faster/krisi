import uuid
from typing import Optional, Tuple

import pandas as pd

from .type import (
    MetricCategories,
    PathConst,
    Predictions,
    SampleTypes,
    SaveModes,
    Targets,
)


def handle_unnamed(
    y: Targets,
    model_name: Optional[str],
    dataset_name: Optional[str],
    project_name: Optional[str],
) -> Tuple[str, str, str]:
    if dataset_name is None:
        if isinstance(y, pd.Series) and y.name is not None:
            dataset_name = str(y.name)
        else:
            dataset_name = f"Dataset:{str(uuid.uuid4()).split('-')[0]}"

    if model_name is None:
        model_name = f"Model:{str(uuid.uuid4()).split('-')[0]}"

    if project_name is None:
        project_name = f"Project:{str(uuid.uuid4()).split('-')[0]}"

    return model_name, dataset_name, project_name
