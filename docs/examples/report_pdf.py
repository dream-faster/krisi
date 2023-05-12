"""
Static, PDF report
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from docs.examples.utils import handle_test
from krisi import score
from krisi.report.type import DisplayModes

sc = score(
    y=np.random.normal(0, 0.1, 1000),
    predictions=np.random.normal(0, 0.1, 1000),
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    project_name="Example Project",
    calculation="rolling",
)

sc.generate_report(
    display_modes=[DisplayModes.pdf],
)

handle_test()
