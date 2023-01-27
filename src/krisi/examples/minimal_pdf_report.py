import numpy as np

from krisi.evaluate import ScoreCard
from krisi.report.type import DisplayModes

target, predictions = np.random.rand(100), np.random.rand(100)
sc = ScoreCard(
    target,
    predictions,
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    project_name="Example Project",
)

""" A predefined metrics """
sc.evaluate(defaults=True)
sc.generate_report(
    display_modes=[DisplayModes.pdf],
)
