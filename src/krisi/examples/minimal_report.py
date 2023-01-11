import numpy as np

from krisi.evaluate import SampleTypes, ScoreCard

target, predictions = np.random.rand(10), np.random.rand(10)
sc = ScoreCard(
    target,
    predictions,
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    sample_type=SampleTypes.insample,
)

""" A predefined metrics """
sc.evaluate(defaults=True)
sc.generate_report()
