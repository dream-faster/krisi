import datetime
import os
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from krisi.evaluate.type import (
    NamingPrefixes,
    PathConst,
    Predictions,
    Probabilities,
    Targets,
)
from krisi.utils.iterable_helpers import is_int

if TYPE_CHECKING:
    from krisi.evaluate.scorecard import ScoreCard, ScoreCardMetadata


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
            dataset_name = f"{NamingPrefixes.dataset}{display_time+str(uuid.uuid4()).split('-')[0]}"

    if model_name is None:
        if isinstance(predictions, pd.Series) and predictions.name is not None:
            model_name = str(predictions.name)
        else:
            model_name = (
                f"{NamingPrefixes.model}{display_time+str(uuid.uuid4()).split('-')[0]}"
            )

    if project_name is None:
        project_name = (
            f"{NamingPrefixes.dataset}{display_time+str(uuid.uuid4()).split('-')[0]}"
        )

    return model_name, dataset_name, project_name


def get_save_path(project_name: str) -> Path:
    return Path(os.path.join(PathConst.default_eval_output_path, project_name))


def last_model_name(metadata: "ScoreCardMetadata") -> Path:
    if metadata.model_name[:6] == NamingPrefixes.model and is_int(
        metadata.model_name[7]
    ):
        path = os.path.join(metadata.save_path, PathConst.default_save_output_path)
        if os.path.exists(path):
            listdir = os.listdir(
                os.path.join(metadata.save_path, PathConst.default_save_output_path)
            )
            dir_model_name = listdir[-1]
        else:
            dir_model_name = metadata.model_name
    else:
        dir_model_name = metadata.model_name

    return Path(dir_model_name)


def convert_to_series(
    data: Union[List[float], List[int], pd.Series, np.ndarray], name: str
) -> pd.Series:
    """Converts a list[floats or ints] or a numpy array to a pandas Series.

    Parameters
    ----------
    data : Union[List[float], List[int], pd.Series, np.ndarray]
        The data to convert.

    Returns
    -------
    pd.Series
        The converted data.
    """
    if isinstance(data, pd.Series):
        return data.rename(name)
    return pd.Series(data, name=name)


def ensure_df(data: Probabilities, name: str) -> pd.DataFrame:
    """Converts a Probabilities to a pandas DataFrame.

    Parameters
    ----------
    data : Probabilities
        The data to convert.

    Returns
    -------
    pd.DataFrame
        The converted data.
    """
    if isinstance(data, pd.DataFrame):
        return data
    elif isinstance(data, pd.Series):
        return data.to_frame(name)
    elif isinstance(data, np.ndarray):
        df = pd.DataFrame(data)
        df.index.name = name
        return df
    elif isinstance(data, list):
        df = pd.DataFrame(data)
        df.index.name = name
        return df
    else:
        raise ValueError(f"Data type {type(data)} not supported.")


def rename_probs_columns(df: pd.DataFrame) -> pd.DataFrame:
    if all(
        [
            col if isinstance(col, int) else is_int(col.split("_")[-1])
            for col in df.columns
        ]
    ):
        return df.rename(columns={col: int(col.split("_")[-1]) for col in df.columns})
    else:
        return df.rename(columns={col: i for i, col in enumerate(df.columns)})
