import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from krisi.evaluate.library.diagrams import (
    callibration_plot,
    display_single_value,
    display_time_series,
)
from krisi.evaluate.library.metric_wrappers import brier_multi
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories

accuracy = Metric[float](
    name="Accuracy",
    key="accuracy",
    category=MetricCategories.class_err,
    info="In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
    func=accuracy_score,
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
)
""" ~ """

recall = Metric[float](
    name="Recall",
    key="recall",
    category=MetricCategories.class_err,
    info="The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html",
    parameters={"average": "binary"},
    func=recall_score,
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
)
"""~"""
precision = Metric[float](
    name="Precision",
    key="precision",
    category=MetricCategories.class_err,
    info="The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html",
    parameters={"average": "binary"},
    func=precision_score,
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
)
"""~"""
f_one_score = Metric[float](
    name="F1 Score",
    key="f_one_score",
    category=MetricCategories.class_err,
    info="The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
    parameters={"average": "macro"},
    func=f1_score,
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
    supports_multiclass=True,
)
"""~"""
matthew_corr = Metric[float](
    name="Matthew Correlation Coefficient",
    key="matthew_corr",
    category=MetricCategories.class_err,
    info="The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.",
    func=matthews_corrcoef,
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
)
"""~"""
brier_score = Metric[float](
    name="Brier Score",
    key="brier_score",
    category=MetricCategories.class_err,
    info="The smaller the Brier score loss, the better, hence the naming with “loss”. The Brier score measures the mean squared difference between the predicted probability and the actual outcome. The Brier score always takes on a value between zero and one, since this is the largest possible difference between a predicted probability (which must be between zero and one) and the actual outcome (which can take on values of only 0 and 1). It can be decomposed as the sum of refinement loss and calibration loss.",
    func=lambda y, pred, prob, **kwargs: brier_score_loss(
        y_true=y, y_prob=prob, **kwargs
    ),
    parameters=dict(pos_label=1),
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=True,
)
"""~"""
callibration = Metric[float](
    name="Callibration Plot",
    key="callibration",
    category=MetricCategories.class_err,
    info="Used to plot the callibration of a model with it probabilities.",
    func=lambda y, pred, prob: pd.concat(
        [y, prob.iloc[:, 0].rename("probs")], axis="columns"
    ),
    plot_funcs=[(callibration_plot, dict(width=1500.0, bin_size=0.1))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=True,
    supports_multiclass=True,
)
"""~"""

brier_score_multi = Metric[float](
    name="Brier Score Multilabel",
    key="brier_score_multi",
    category=MetricCategories.class_err,
    info="Multilabel calculation of the Brier score loss. The smaller the Brier score loss, the better, hence the naming with “loss”. The Brier score measures the mean squared difference between the predicted probability and the actual outcome. The Brier score always takes on a value between zero and one, since this is the largest possible difference between a predicted probability (which must be between zero and one) and the actual outcome (which can take on values of only 0 and 1). It can be decomposed as the sum of refinement loss and calibration loss.",
    func=lambda y, pred, prob, **kwargs: brier_multi(y, prob, **kwargs),
    plot_funcs=[(display_single_value, dict(width=500.0))],
    plot_func_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=True,
    supports_multiclass=True,
)
"""~"""
all_classification_metrics = [
    accuracy,
    recall,
    precision,
    f_one_score,
    matthew_corr,
    brier_score,
    brier_score_multi,
    callibration,
]
"""~"""
minimal_classification_metrics = [accuracy, f_one_score]
"""~"""
multilabel_classification = [
    metric for metric in all_classification_metrics if metric.supports_multiclass
]
"""~"""
