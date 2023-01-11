import numpy as np

from krisi.evaluate import SampleTypes, ScoreCard

target, predictions = np.random.rand(100), np.random.rand(100)
sc = ScoreCard(
    target,
    predictions,
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    project_name="Example Project",
    sample_type=SampleTypes.insample,
)

""" A predefined metrics """
sc.evaluate(defaults=True)
sc.generate_report()
