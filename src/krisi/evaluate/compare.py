from typing import List, Optional, Tuple, Union

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.utils.printing import bold


def __handle_empty_metrics_to_display(
    scorecard: ScoreCard, sort_by: Optional[str], metric_keys: List[str]
) -> Tuple[List[str], Optional[str]]:
    if len(metric_keys) == 0:
        metric_keys = [
            metric.key
            for metric in scorecard.get_all_metrics(only_evaluated=True)
            if isinstance(metric.result, (float, int))
        ]

    if sort_by is not None:
        if sort_by not in metric_keys:
            metric_keys.insert(0, sort_by)
        else:
            metric_keys.remove(sort_by)
            metric_keys.insert(0, sort_by)

    else:
        sort_by = metric_keys[0]
    return metric_keys, sort_by


def compare(
    scorecards: List[ScoreCard],
    metric_keys: List[str] = [],
    sort_by: Optional[str] = None,
    dataframe: bool = True,
) -> Union[pd.DataFrame, str]:
    metric_keys, sort_by = __handle_empty_metrics_to_display(
        scorecards[0], sort_by, metric_keys
    )

    scorecards.sort(reverse=True, key=lambda x: x[sort_by].result)
    if dataframe:
        return pd.concat(
            [
                pd.Series(
                    [scorecard[metric_key].result for scorecard in scorecards],
                    name=metric_key,
                    index=[scorecard.metadata.model_name for scorecard in scorecards],
                )
                for metric_key in metric_keys
            ],
            axis="columns",
        )
    else:
        metric_keys.remove(sort_by)
        string_to_return = ""
        metric_title = "".join([f"{metric:<10s}" for metric in metric_keys])
        string_to_return += f"{'model_name':>30s}    {bold(f'{sort_by:<10s}', rich=False)} {metric_title}"

        for scorecard in scorecards:
            metrics = "".join(
                [f"{scorecard[metric].result:<10.5}" for metric in metric_keys]
            )

            string_to_return += f"\n{scorecard.metadata.model_name:>30s}    {bold(f'{scorecard[sort_by].result:<10.5}', rich=False)} {metrics}"

        return string_to_return
