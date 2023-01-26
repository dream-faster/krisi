from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from krisi.evaluate.library.diagrams import (
    display_acf_plot,
    display_density_plot,
    display_single_value,
    display_time_series,
)
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories

accuracy = Metric[float](
    name="Accuracy",
    key="accuracy",
    category=MetricCategories.class_err,
    info="In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
    func=accuracy_score,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
recall = Metric[float](
    name="Recall",
    key="recall",
    category=MetricCategories.class_err,
    info="The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html",
    parameters={"average": "binary"},
    func=recall_score,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
precision = Metric[float](
    name="Precision",
    key="precision",
    category=MetricCategories.class_err,
    info="The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html",
    parameters={"average": "binary"},
    func=precision_score,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)
f_one_score = Metric[float](
    name="F1 Score",
    key="f_one_score",
    category=MetricCategories.class_err,
    info="The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
    parameters={"average": "binary"},
    func=f1_score,
    plot_funcs=[display_single_value],
    plot_func_rolling=display_time_series,
)


all_classification_metrics = [accuracy, recall, precision, f_one_score]
minimal_classification_metrics = [accuracy, f_one_score]
