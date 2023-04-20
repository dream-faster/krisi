import datetime
import uuid
from typing import TYPE_CHECKING, List, Optional, Tuple

import pandas as pd

from krisi.evaluate.type import Predictions, Targets

if TYPE_CHECKING:
    from krisi.evaluate.scorecard import ScoreCard


def handle_empty_metrics_to_display(
    scorecard: "ScoreCard", sort_by: Optional[str], metric_keys: Optional[List[str]]
) -> Tuple[List[str], Optional[str]]:
    if metric_keys is None:
        metric_keys = [
            metric.key
            for metric in scorecard.get_all_metrics(only_evaluated=True)
            if isinstance(metric.result, (float, int))
        ]

    if sort_by is not None:
        if sort_by not in metric_keys:
            metric_keys.insert(0, sort_by)
        else:
            metric_keys.remove(sort_by)
            metric_keys.insert(0, sort_by)

    else:
        sort_by = metric_keys[0]
    return metric_keys, sort_by


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
