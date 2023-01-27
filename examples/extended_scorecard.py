import numpy as np

from krisi.evaluate import Metric, MetricCategories, SampleTypes, ScoreCard

target, predictions = np.random.rand(100), np.random.rand(100)

sc = ScoreCard(
    target,
    predictions,
    "<your_model_name>",
    dataset_name="<your_dataset_name>",
    sample_type=SampleTypes.insample,
)

""" Predefined metrics """
sc.evaluate(defaults=True)

""" Adding a new metric """
calculated_metric_example = (target - predictions).mean()

sc["own_metric"] = calculated_metric_example
# or
sc.another_metric = calculated_metric_example * 2.0
# or
sc["yet_another_metric"] = Metric(
    name="A new, own Metric",
    category=MetricCategories.residual,
    result=calculated_metric_example * 3.0,
    parameters={"hyper_1": 5.0},
)

# Updating a metric
sc.yet_another_metric = dict(info="Giving description to a metric")

""" Print scorecard summary """
sc.print_summary(with_info=True, extended=True)
