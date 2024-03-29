"""
Basic Loading
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import compare, score
from krisi.evaluate.type import PathConst
from krisi.utils.io import load_scorecards

project_name = "Example Load"

for _ in range(5):
    score(
        y=np.random.rand(1000),
        predictions=np.random.rand(1000),
        project_name=project_name,
    ).save()


print(
    compare(
        load_scorecards(project_name, PathConst.default_eval_output_path),
        metric_keys=["mse"],
    )
)
