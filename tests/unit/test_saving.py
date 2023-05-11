import os
import shutil
from pathlib import Path

import numpy as np

from krisi import score
from krisi.evaluate.type import PathConst


def test_saving():
    project_name = "Example Load Overwritten 9832423"
    path = Path(
        os.path.join(
            PathConst.default_eval_output_path,
            os.path.join(Path(project_name), PathConst.default_save_output_path),
        )
    )
    if os.path.exists(path):
        shutil.rmtree(path)
    for _ in range(5):
        score(
            y=np.random.rand(1000),
            predictions=np.random.rand(1000),
            project_name=project_name,
            model_name="Example Model",
        ).save()

    project_files = os.listdir(path)

    assert (
        len(set(project_files)) == 1
    ), "Models should be overwritten, we got more than one models loaded."

    project_name_2 = "Example Load Not Overwritten random model name"
    path = Path(
        os.path.join(
            PathConst.default_eval_output_path,
            os.path.join(Path(project_name_2), PathConst.default_save_output_path),
        )
    )
    if os.path.exists(path):
        shutil.rmtree(path)
    for _ in range(5):
        score(
            y=np.random.rand(1000),
            predictions=np.random.rand(1000),
            project_name=project_name_2,
        ).save()

    project_files = os.listdir(path)

    assert (
        len(set(project_files)) == 5
    ), f"Models should be not overwritten, we got {len(set(project_files))} models loaded."

    project_name_3 = "Example Load Not Overwritten timestamp boolean"
    path = Path(
        os.path.join(
            PathConst.default_eval_output_path,
            os.path.join(Path(project_name_3), PathConst.default_save_output_path),
        )
    )
    if os.path.exists(path):
        shutil.rmtree(path)
    for _ in range(5):
        score(
            y=np.random.rand(1000),
            predictions=np.random.rand(1000),
            project_name=project_name_3,
            model_name="Example Model",
        ).save(timestamp_no_overwrite=True)

    project_files = os.listdir(path)

    assert (
        len(set(project_files)) == 5
    ), "Models should be not overwritten, we set saving to timestamp."
