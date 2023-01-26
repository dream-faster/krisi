import numpy as np

from krisi.evaluate import SampleTypes, ScoreCard
from krisi.report.type import DisplayModes

target, predictions = np.random.rand(100), np.random.rand(100)
sc = ScoreCard(
    target,
    predictions,
    model_name="<your_model_name>",
    model_description="You can explain main features of the project to be displayed in the report",
    dataset_name="<your_dataset_name>",
    dataset_description="You can explain main features of the dataset to be displayed in the report",
    project_name="Example Project",
    project_description="The extra information about what the project is about will be displayed when creating a report",
    sample_type=SampleTypes.insample,
)

""" A predefined metrics """
sc.evaluate(defaults=True)
sc.generate_report(
    display_modes=[DisplayModes.interactive],
)
