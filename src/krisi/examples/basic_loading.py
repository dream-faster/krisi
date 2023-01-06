import numpy as np

from krisi import compare, evaluate
from krisi.evaluate.compare import load_scorecards
from krisi.evaluate.type import PathConst, SaveModes

project_name = "Example Load"

for _ in range(5):
    evaluate(
        y=np.random.rand(1000),
        predictions=np.random.rand(1000),
        project_name=project_name,
    ).save()


compare(
    load_scorecards(PathConst.default_eval_output_path, project_name),
    metrics_to_display=["mse"],
)
