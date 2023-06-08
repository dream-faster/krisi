"""
A single figure with subplots of all other figures opened in browser or inline in Jupyter Notebook
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import score
from krisi.report.type import DisplayModes
from krisi.utils.devutils.environment_checks import handle_test

sc = score(
    y=np.random.random(1000),
    predictions=np.random.random(1000),
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    project_name="Example Project",
)

sc.generate_report(
    display_modes=[DisplayModes.direct_one_subplot],
)

handle_test()
