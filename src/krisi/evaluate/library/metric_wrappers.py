import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from statsmodels.stats.diagnostic import acorr_ljungbox

from krisi.evaluate.type import Predictions, PredictionsDS, ProbabilitiesDF, TargetsDS

logger = logging.getLogger("krisi")


def ljung_box(
    y: TargetsDS, predictions: Predictions
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    return acorr_ljungbox(predictions)


def create_one_hot(ds: Union[TargetsDS, ProbabilitiesDF]) -> pd.DataFrame:
    return pd.get_dummies(ds)


def brier_multi(targets: TargetsDS, probs: ProbabilitiesDF, **kwargs):
    # See https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    return np.mean(np.sum((probs - create_one_hot(targets)) ** 2, axis=1))


def decide_class_label(
    probs: ProbabilitiesDF, label_index: Optional[int] = None
) -> Tuple[Union[str, int], bool]:
    if label_index in probs.columns:
        return label_index, True
    elif str(label_index) in probs.columns:
        return str(label_index), True
    elif label_index is None and any([(i or str(i)) in probs.columns for i in [0, 1]]):
        for i in [0, 1]:
            if i in probs.columns:
                logger.info(f"Found an integer {i} column, setting it as pos_label")
                return i, True
            elif str(i) in probs.columns:
                logger.info(f"Found an string {i} column, setting it as pos_label")
                return str(i), True
    else:
        logger.info(
            "pos_label undefined and no integer class found in columns -> setting pos_label to iloc.[:,0]"
        )
        return 0, False


def brier_score_wrap(
    y: TargetsDS, preds: PredictionsDS, probs: ProbabilitiesDF, **kwargs
) -> float:
    if len(probs.columns) > 2:
        raise ValueError(
            "Brier score only supports binary classification. Use brier_score_multi instead."
        )
    if len(probs.columns) == 2:
        label_index = kwargs.get("pos_label", None)

        label_index, found_in_df = decide_class_label(probs, label_index)

        logger.info(
            f"Brier score only supports binary classification. Using column named {label_index} of probabilities."
        )

        if found_in_df:
            probs = probs[[label_index]]
        else:
            probs = probs.iloc[:, label_index]

    return brier_score_loss(
        y_true=y,
        y_prob=probs,
        **kwargs,
    )
