from typing import List

from krisi.utils.printing import bold

from .scorecard import ScoreCard


def compare(
    scorecards: List[ScoreCard],
    sort_metric_key: str = "rmse",
    metrics_to_display: List[str] = [],
) -> None:

    metric_title = "".join([f"{metric:<15s}" for metric in metrics_to_display])
    print(
        f"{'model_name':>30s}    {bold(f'{sort_metric_key:<15s}', rich=False)} {metric_title}"
    )
    scorecards.sort(reverse=True, key=lambda x: x[sort_metric_key].result)

    for scorecard in scorecards:
        metrics = "".join(
            [f"{scorecard[metric].result:<15.5}" for metric in metrics_to_display]
        )

        print(
            f"{scorecard.model_name:>30s}    {bold(f'{scorecard[sort_metric_key].result:<15.5}', rich=False)} {metrics}"
        )


def load_scorecards(path: str, project_name: str) -> List[ScoreCard]:
    import os
    import pickle

    files = os.listdir(f"{path}/{project_name}")

    loaded_scorecards = []
    for file in files:
        with open(f"{path}{project_name}/{file}/scorecard.pickle", "rb") as f:
            loaded_scorecards.append(pickle.load(f))

    return loaded_scorecards
