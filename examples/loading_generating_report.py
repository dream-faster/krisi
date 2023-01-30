import os

import numpy as np

from krisi import ScoreCard
from krisi.evaluate.type import PathConst
from krisi.utils.io import load_scorecards


def create_scorecard(project_name: str) -> None:
    target, predictions = np.random.rand(100), np.random.rand(100)

    sc = ScoreCard(
        target,
        predictions,
        model_name="<your_model_name>",
        dataset_name="<your_dataset_name>",
        project_name=project_name,
    )

    sc.evaluate(defaults=True)
    sc.save()


def load_and_generate(project_name: str) -> None:
    scorecard = load_scorecards(project_name, PathConst.default_eval_output_path)
    scorecard[-1].generate_report()


project_name = "Example Project"
create_scorecard(project_name)


if "PYTEST_CURRENT_TEST" in os.environ:
    print("Not testing Dash server currently")
else:
    load_and_generate(project_name)
