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


compare(
    load_scorecards(project_name, PathConst.default_eval_output_path),
    metrics_to_display=["mse"],
)
