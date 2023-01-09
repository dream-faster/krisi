import numpy as np

from krisi.evaluate import SampleTypes, ScoreCard

target, predictions = np.random.rand(1000), np.random.rand(1000)
sc = ScoreCard(
    target,
    predictions,
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    sample_type=SampleTypes.insample,
)

""" A predefined metrics """
sc.evaluate(defaults=True)

""" Adding a new metric """
sc["own_metric"] = (target - predictions).mean()

""" Print scorecard summary """
sc.print_summary(extended=True)
