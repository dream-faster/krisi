import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from krisi.evaluate.library.diagrams import display_time_series
from krisi.evaluate.library.metric_wrappers import ljung_box
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories, SampleTypes

predefined_classification_metrics = [
    Metric[float](
        name="Accuracy",
        key="accuracy",
        category=MetricCategories.class_err,
        info="In multilabel classification, this function computes subset accuracy: the set of labels predicted for a sample must exactly match the corresponding set of labels in y_true. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html",
        func=accuracy_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Recall",
        key="recall",
        category=MetricCategories.class_err,
        info="The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html",
        parameters={"average": "binary"},
        func=recall_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Precision",
        key="precision",
        category=MetricCategories.class_err,
        info="The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\nThe best value is 1 and the worst value is 0. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html",
        parameters={"average": "binary"},
        func=precision_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="F1 Score",
        key="f1_score",
        category=MetricCategories.class_err,
        info="The F1 score can be interpreted as a harmonic mean of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html",
        parameters={"average": "binary"},
        func=f1_score,
        plot_func=display_time_series,
    ),
]
