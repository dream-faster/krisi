import logging
from typing import Any, Optional, Tuple, Union

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


def wrap_brier_score(y: TargetsDS, probs: ProbabilitiesDF, **kwargs) -> float:
    return brier_score_loss(
        y_true=y,
        y_prob=probs.iloc[:, 1],
        **kwargs,
    )


def wrap_roc_auc(
    y: TargetsDS,
    probs: ProbabilitiesDF,
    sample_weight: Optional[None] = None,
    **kwargs,
) -> float:
    probs = probs.rename(columns={col: i for i, col in enumerate(probs.columns)})
    prob_columns = list(probs.columns)
    if len(prob_columns) == 2:
        probs = probs.iloc[:, 1]

    return roc_auc_score(
        y_true=y,
        y_score=probs,
        labels=prob_columns,
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
    tn, fp, fn, tp = confusion_matrix(y, preds, sample_weight=sample_weight).ravel()

    p_0 = (tn + tp) / (tn + fp + fn + tp)

    S_Score = (2 * p_0) - 1
    return S_Score


def y_label_imbalance_ratio(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: Optional[WeightsDS] = None,
    pos_label: int = 1,
    **kwargs,
) -> float:
    label_frequencies = y.value_counts(normalize=True).dropna()
    if len(label_frequencies) == 2:
        return min(
            label_frequencies[pos_label], 1 - label_frequencies[pos_label]
        )  # always returns the smaller of the two
    elif len(label_frequencies) == 1:
        logger.warning("Only one class in target, returning 0")
        return 0.0
    else:
        raise ValueError(
            f"Imbalance only works on binary classification problems, got more than 2 labels: {label_frequencies}"
        )


def pred_y_imbalance_ratio(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: Optional[WeightsDS] = None,
    pos_label: int = 1,
    **kwargs,
) -> float:
    prediction_frequencies = preds.value_counts(normalize=True).dropna()
    target_frequencies = y.value_counts(normalize=True).dropna()
    if len(prediction_frequencies) == 1 or len(target_frequencies) == 1:
        logger.warning("Only one class in either predictions or target, returning 0")
        return 0.0
    elif len(prediction_frequencies) != 2 or len(target_frequencies) != 2:
        raise ValueError(
            f"pred_y_imbalance_ratio only works on binary classification problems, got more than 2 labels for either preds: {prediction_frequencies} or target: {target_frequencies}"
        )

    return abs(
        min(target_frequencies[pos_label], 1 - target_frequencies[pos_label])
        - min(prediction_frequencies[pos_label], 1 - prediction_frequencies[pos_label])
    )
