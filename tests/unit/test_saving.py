import os
from pathlib import Path

import numpy as np

from krisi import score
from krisi.evaluate.type import PathConst


def test_saving():
    project_name = "Example Load Overwritten 9832423"

    for _ in range(5):
        score(
            y=np.random.rand(1000),
            predictions=np.random.rand(1000),
            project_name=project_name,
            model_name="Example Model",
        ).save()

    path = Path(
        os.path.join(
            PathConst.default_eval_output_path,
            os.path.join(Path(project_name), PathConst.default_save_output_path),
        )
    )
    project_files = os.listdir(path)

    assert (
        len(set(project_files)) == 1
    ), "Models should be overwritten, we got more than one models loaded."
