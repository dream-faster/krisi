import logging
from typing import Any, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from statsmodels.stats.diagnostic import acorr_ljungbox

from krisi.evaluate.type import (
    Predictions,
    PredictionsDS,
    ProbabilitiesDF,
    TargetsDS,
    WeightsDS,
)

logger = logging.getLogger("krisi")


def ljung_box(
    y: TargetsDS, predictions: Predictions, **kwargs
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    return acorr_ljungbox(predictions)


def create_one_hot(ds: Union[TargetsDS, ProbabilitiesDF]) -> pd.DataFrame:
    return pd.get_dummies(ds)


def brier_multi(targets: TargetsDS, probs: ProbabilitiesDF, **kwargs):
    # See https://stats.stackexchange.com/questions/403544/how-to-compute-the-brier-score-for-more-than-two-classes
    return np.mean(
        np.sum(
            (probs - create_one_hot(targets)) ** 2,
            axis=1,
        )
    )


def wrap_brier_score(
    y: TargetsDS, preds: PredictionsDS, probs: ProbabilitiesDF, **kwargs
) -> float:
    return brier_score_loss(
        y_true=y,
        y_prob=probs,
        **kwargs,
    )


def wrap_roc_auc(
    y: TargetsDS, preds: PredictionsDS, probs: ProbabilitiesDF, **kwargs
) -> float:
    return roc_auc_score(
        y_true=y,
        y_score=probs,
        labels=list(probs.columns),
        **kwargs,
    )


def bennet_s(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: WeightsDS,
    **kwargs,
) -> float:
    """Bennett, Alpert, & Goldstein S score.
    Notes
    -------
    See https://stats.stackexchange.com/a/217404/315248

    Returns
    -------
    float
        float
    """
    tn, fp, fn, tp = confusion_matrix(y, preds, sample_weight=sample_weight)

    p_0 = (tn + tp) / (tn + fp + fn + tp)

    S_Score = (2 * p_0) - 1
    return S_Score
