import logging
from typing import Any, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)

from krisi.evaluate.type import (
    Predictions,
    PredictionsDS,
    ProbabilitiesDF,
    TargetsDS,
    WeightsDS,
)

logger = logging.getLogger("krisi")


def wrap_brier_score(
    y: TargetsDS, probs: ProbabilitiesDF, sample_weight: Optional[WeightsDS], **kwargs
) -> float:
    return brier_score_loss(
        y_true=y,
        y_prob=probs.iloc[:, 1],
        sample_weight=sample_weight,
        **kwargs,
    )


def wrap_roc_auc(
    y: TargetsDS,
    probs: ProbabilitiesDF,
    sample_weight: Optional[WeightsDS],
    **kwargs,
) -> float:
    probs = probs.rename(columns={col: i for i, col in enumerate(probs.columns)})
    prob_columns = list(probs.columns)
    if len(prob_columns) == 2:
        probs = probs.iloc[:, 1]
    if len(np.unique(y)) == 1:
        return "ROC AUC only works with more than one class."

    return roc_auc_score(
        y_true=y,
        y_score=probs,
        labels=prob_columns,
        sample_weight=sample_weight,
        **kwargs,
    )


def wrap_avg_precision(
    y: TargetsDS,
    probs: ProbabilitiesDF,
    sample_weight: Optional[WeightsDS],
    **kwargs,
) -> float:
    probs = probs.rename(columns={col: i for i, col in enumerate(probs.columns)})
    prob_columns = list(probs.columns)
    if len(prob_columns) == 2:
        probs = probs.iloc[:, 1]

    return average_precision_score(
        y_true=y,
        y_score=probs,
        sample_weight=sample_weight,
        **kwargs,
    )


def bennet_s(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: Optional[WeightsDS],
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


""" v Diagnostics Wrappers v """


def y_label_imbalance_ratio(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: Optional[WeightsDS],
    pos_label: int = 1,
    **kwargs,
) -> float:
    label_frequencies = y.value_counts(normalize=True).dropna()
    perfect_balance = 0.5
    if len(label_frequencies) == 2:
        return abs(label_frequencies[pos_label] - perfect_balance)

    elif len(label_frequencies) == 1:
        return 0.0
    else:
        raise ValueError(
            f"Imbalance only works on binary classification problems, got more than 2 labels: {label_frequencies}"
        )


def pred_y_imbalance_ratio(
    y: TargetsDS,
    preds: PredictionsDS,
    sample_weight: Optional[WeightsDS],
    pos_label: int = 1,
    **kwargs,
) -> float:
    perfect_balance = 0.5
    prediction_frequencies = preds.value_counts(normalize=True).dropna()
    target_frequencies = y.value_counts(normalize=True).dropna()
    if len(prediction_frequencies) == 1 or len(target_frequencies) == 1:
        return 0.0
    elif len(prediction_frequencies) != 2 or len(target_frequencies) != 2:
        raise ValueError(
            f"pred_y_imbalance_ratio only works on binary classification problems, got more than 2 labels for either preds: {prediction_frequencies} or target: {target_frequencies}"
        )

    return abs(
        abs(target_frequencies[pos_label] - perfect_balance)
        - abs(prediction_frequencies[pos_label] - perfect_balance)
    )


def ljung_box(
    y: TargetsDS, predictions: Predictions, sample_weight: Optional[WeightsDS], **kwargs
) -> Union[pd.DataFrame, Tuple[Any, Any], Tuple[Any, Any, Any, Any]]:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    return acorr_ljungbox(predictions, **kwargs)
