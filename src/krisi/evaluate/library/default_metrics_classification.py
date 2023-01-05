import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

from krisi.evaluate.library.diagrams import display_time_series
from krisi.evaluate.library.metric_wrappers import ljung_box
from krisi.evaluate.metric import Metric
from krisi.evaluate.type import MetricCategories, SampleTypes

predefined_classification_metrics = [
    Metric[float](
        name="Accuracy",
        key="accuracy",
        category=MetricCategories.class_err,
        info="",
        func=accuracy_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Recall",
        key="recall",
        category=MetricCategories.class_err,
        info="The recall is the ratio tp / (tp + fn) where tp is the number of true positives and fn the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples.\nThe best value is 1 and the worst value is 0.",
        parameters={"average": "binary"},
        func=recall_score,
        plot_func=display_time_series,
    ),
    Metric[float](
        name="Precision",
        key="precision",
        category=MetricCategories.class_err,
        info="The precision is the ratio tp / (tp + fp) where tp is the number of true positives and fp the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.\nThe best value is 1 and the worst value is 0.",
        parameters={"average": "binary"},
        func=precision_score,
        plot_func=display_time_series,
    ),
]
