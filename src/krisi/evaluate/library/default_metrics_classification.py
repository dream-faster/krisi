import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from krisi.evaluate.group import Group
from krisi.evaluate.library.diagrams import (
    callibration_plot,
    display_density_plot,
    display_single_value,
    display_time_series,
)
from krisi.evaluate.library.metric_wrappers import (
    bennet_s,
    brier_multi,
    wrap_brier_score,
    wrap_roc_auc,
)
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import Calculation, MetricCategories

accuracy_binary = Metric[float](
    name="Accuracy",
    key="accuracy",
    category=MetricCategories.class_err,
    info="In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
    func=accuracy_score,
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
)
""" ~ """

recall_binary, recall_macro = [
    Metric[float](
        name=f"Recall ({mode})",
        key=f"recall_{mode}",
        category=MetricCategories.class_err,
        info="The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html",
        parameters={"average": mode},
        func=recall_score,
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
    )
    for mode in ["binary", "macro"]
]
"""~"""
precision_binary, precision_macro = [
    Metric[float](
        name=f"Precision ({mode})",
        key=f"precision_{mode}",
        category=MetricCategories.class_err,
        info="The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html",
        parameters={"average": mode},
        func=precision_score,
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
    )
    for mode in ["binary", "macro"]
]
"""~"""
f_one_score_binary, f_one_score_macro, f_one_score_micro, f_one_score_weighted = [
    Metric[float](
        name=f"F1 Score ({mode})",
        key=f"f_one_score_{mode}",
        category=MetricCategories.class_err,
        info="The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
        parameters={"average": mode},
        func=f1_score,
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
        supports_multiclass=True,
    )
    for mode in ["binary", "macro", "micro", "weighted"]
]
"""~"""
matthew_corr = Metric[float](
    name="Matthew Correlation Coefficient",
    key="matthew_corr",
    category=MetricCategories.class_err,
    info="The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes. The MCC is in essence a correlation coefficient value between -1 and +1. A coefficient of +1 represents a perfect prediction, 0 an average random prediction and -1 an inverse prediction. The statistic is also known as the phi coefficient.",
    func=matthews_corrcoef,
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
)
"""~"""
brier_score = Metric[float](
    name="Brier Score",
    key="brier_score",
    category=MetricCategories.class_err,
    info="The smaller the Brier score loss, the better, hence the naming with “loss”. The Brier score measures the mean squared difference between the predicted probability and the actual outcome. The Brier score always takes on a value between zero and one, since this is the largest possible difference between a predicted probability (which must be between zero and one) and the actual outcome (which can take on values of only 0 and 1). It can be decomposed as the sum of refinement loss and calibration loss.",
    func=wrap_brier_score,
    parameters=dict(pos_label=1),
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=True,
)
"""~"""
calibration = Metric[float](
    name="Calibration Plot",
    key="calibration",
    category=MetricCategories.class_err,
    info="Used to plot the calibration of a model with it probabilities.",
    func=lambda y, pred, prob, **kwargs: pd.concat(
        [y, prob.iloc[:, 0].rename("probs")], axis="columns"
    ),
    plot_funcs=[(callibration_plot, dict(width=1500.0, bin_size=0.1))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
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
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=True,
    supports_multiclass=True,
)
"""~"""
s_score = Metric[float](
    name="S Score",
    key="s_score",
    category=MetricCategories.class_err,
    info="the 𝑆 score first proposed by Bennett, Alpert, & Goldstein (1954). It assumes a 50% probability of classifying any given item into its correct class 'by chance' alone and discounts this from the final score. It also uses all cells of the confusion matrix (unlike 𝐹1 which ignores 𝑑 or the number of 'true negatives')",
    func=bennet_s,
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
    accepts_probabilities=False,
    supports_multiclass=False,
)
"""~"""


cross_entropy = Metric[float](
    name="Cross Entropy",
    key="cross_entropy",
    category=MetricCategories.class_err,
    info="Log loss, aka logistic loss or cross-entropy loss. This is the loss function used in (multinomial) logistic regression and extensions of it such as neural networks, defined as the negative log-likelihood of a logistic model that returns y_pred probabilities for its training data y_true.",
    func=lambda y, pred, prob, **kwargs: log_loss(
        y, prob, labels=list(prob.columns), **kwargs
    ),
    plot_funcs=[
        (display_single_value, dict(width=750.0)),
        (display_density_plot, dict(width=1500.0)),
    ],
    plot_funcs_rolling=[
        (display_density_plot, dict(width=1500.0)),
        (display_time_series, dict(width=1500.0)),
    ],
    accepts_probabilities=True,
    supports_multiclass=True,
)
"""~"""

roc_auc_binary_micro, roc_auc_binary_macro = [
    Metric[float](
        name=f"ROC AUC ({mode}))",
        key=f"roc_auc_binary_{mode}",
        category=MetricCategories.class_err,
        info="Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters). https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
        func=wrap_roc_auc,
        parameters={"average": mode},
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
        accepts_probabilities=True,
        supports_multiclass=True,
    )
    for mode in ["micro", "macro"]
]
"""~"""

roc_auc_multi_weighted, roc_auc_multi_micro, roc_auc_multi_macro = [
    Metric[float](
        name=f"ROC AUC ({mode}))",
        key=f"roc_auc_{mode}",
        category=MetricCategories.class_err,
        info="Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters). https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
        func=wrap_roc_auc,
        parameters={"average": mode, "multi_class": "ovr"},
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
        accepts_probabilities=True,
        supports_multiclass=True,
    )
    for mode in ["weighted", "micro", "macro"]
]
"""~"""

roc_auc_multi_weighted, roc_auc_multi_micro, roc_auc_multi_macro = [
    Metric[float](
        name=f"ROC AUC ({mode}))",
        key=f"roc_auc_{mode}",
        category=MetricCategories.class_err,
        info="Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores. Note: this implementation can be used with binary, multiclass and multilabel classification, but some restrictions apply (see Parameters). https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html",
        func=wrap_roc_auc,
        parameters={"average": mode, "multi_class": "ovr"},
        plot_funcs=[(display_single_value, dict(width=750.0))],
        plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
        accepts_probabilities=True,
        supports_multiclass=True,
    )
    for mode in ["weighted", "micro", "macro"]
]
# """~"""
# ndcg = Metric[float](
#     name="NDCG Score",
#     key="ndcg",
#     category=MetricCategories.class_err,
#     info="Compute Normalized Discounted Cumulative Gain. Sum the true scores ranked in the order induced by the predicted scores, after applying a logarithmic discount. Then divide by the best possible score (Ideal DCG, obtained for a perfect ranking) to obtain a score between 0 and 1. This ranking metric returns a high value if true labels are ranked high by y_score.",
#     func=ndcg_score,
#     parameters={},
#     plot_funcs=[(display_single_value, dict(width=750.0))],
#     plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
#     accepts_probabilities=False,
#     supports_multiclass=True,
# )

standard_deviation = Metric[float](
    name="Standard Deviation",
    key="std",
    category=MetricCategories.stats,
    info="Standard Deviation",
    func=lambda rolling_res, **kwargs: pd.Series(rolling_res).std(),
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
)
median = Metric[float](
    name="Median",
    key="median",
    category=MetricCategories.stats,
    info="Median",
    func=lambda rolling_res, **kwargs: pd.Series(rolling_res).median(),
    plot_funcs=[(display_single_value, dict(width=750.0))],
    plot_funcs_rolling=(display_time_series, dict(width=1500.0)),
)

consistency_group = Group[pd.Series](
    name="Consistency",
    key="consistency",
    metrics=[f_one_score_weighted, matthew_corr],
    postprocess_funcs=[standard_deviation, median],
    calculation=Calculation.rolling,
)

binary_classification_metrics = [
    accuracy_binary,
    recall_binary,
    precision_binary,
    f_one_score_binary,
    f_one_score_macro,
    f_one_score_micro,
    matthew_corr,
    brier_score,
    calibration,
    roc_auc_binary_macro,
    roc_auc_binary_micro,
    cross_entropy,
    s_score,
]
"""~"""
minimal_binary_classification_metrics = [accuracy_binary, f_one_score_binary]
"""~"""
multiclass_classification_metrics = [
    cross_entropy,
    recall_macro,
    precision_macro,
    f_one_score_weighted,
    f_one_score_micro,
    brier_score_multi,
    roc_auc_multi_macro,
    roc_auc_multi_weighted,
    matthew_corr,
    # ndcg,
    consistency_group,
    # s_score,
]
"""~"""
minimal_multiclass_classification_metrics = [
    f_one_score_weighted,
    roc_auc_multi_weighted,
]
