"""
Interactive, HTML report
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import os

import numpy as np

from krisi import score
from krisi.evaluate.type import Calculation
from krisi.report.type import DisplayModes

sc = score(
    y=np.random.random(1000),
    predictions=np.random.random(1000),
    model_name="ARIMA",
    model_description="You can explain main features of the project to be displayed in the report",
    dataset_name="APPLE Stocks",
    dataset_description="You can explain main features of the dataset to be displayed in the report",
    project_name="Quantitative Finance Project",
    project_description="The extra information about what the project is about will be displayed when creating a report",
    calculation=Calculation.rolling,
)

if "PYTEST_CURRENT_TEST" in os.environ:
    print("Not testing Dash server currently")
else:
    sc.generate_report(
        display_modes=[DisplayModes.interactive],
    )
