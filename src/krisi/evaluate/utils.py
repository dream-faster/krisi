import datetime
import uuid
from typing import Optional, Tuple

import pandas as pd

from krisi.evaluate.type import Predictions, Targets


def handle_unnamed(
    y: Targets,
    predictions: Predictions,
    model_name: Optional[str],
    dataset_name: Optional[str],
    project_name: Optional[str],
) -> Tuple[str, str, str]:
    time = datetime.datetime.now()
    display_time = time.strftime("%Y%m%d-%H%M%S")

    if dataset_name is None:
        if isinstance(y, pd.Series) and y.name is not None:
            dataset_name = str(y.name)
        else:
            dataset_name = f"Dataset_{display_time+str(uuid.uuid4()).split('-')[0]}"

    if model_name is None:
        if isinstance(predictions, pd.Series) and predictions.name is not None:
            model_name = str(predictions.name)
        else:
            model_name = f"Model_{display_time+str(uuid.uuid4()).split('-')[0]}"

    if project_name is None:
        project_name = f"Project_{display_time+str(uuid.uuid4()).split('-')[0]}"

    return model_name, dataset_name, project_name
