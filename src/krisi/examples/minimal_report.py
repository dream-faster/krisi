import numpy as np

from krisi.evaluate import SampleTypes, ScoreCard
from krisi.report.type import DisplayModes

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
sc.generate_report(
    html_template_url="library/scorecard/index.html",
    css_template_url="library/scorecard/main.css",
    display_modes=[DisplayModes.interactive],
)
