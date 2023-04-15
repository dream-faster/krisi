from typing import List, Optional

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.utils.printing import bold


def compare(
    scorecards: List[ScoreCard],
    sort_metric_key: str = "rmse",
    metrics_to_display: List[str] = [],
    dataframe: bool = True,
) -> Optional[pd.DataFrame]:
    scorecards.sort(reverse=True, key=lambda x: x[sort_metric_key].result)
    if dataframe:
        return pd.concat(
            [
                pd.Series(
                    [scorecard[metric].result for scorecard in scorecards],
                    name=metric,
                    index=[scorecard.metadata.model_name for scorecard in scorecards],
                )
                for metric in metrics_to_display
            ],
            axis="columns",
        )
    else:
        metric_title = "".join([f"{metric:<15s}" for metric in metrics_to_display])
        print(
            f"{'model_name':>30s}    {bold(f'{sort_metric_key:<15s}', rich=False)} {metric_title}"
        )
        for scorecard in scorecards:
            metrics = "".join(
                [f"{scorecard[metric].result:<15.5}" for metric in metrics_to_display]
            )

            print(
                f"{scorecard.metadata.model_name:>30s}    {bold(f'{scorecard[sort_metric_key].result:<15.5}', rich=False)} {metrics}"
            )
