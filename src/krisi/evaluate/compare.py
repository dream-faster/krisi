from typing import List, Union

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.utils.printing import bold


def __handle_empty_metrics_to_display(
    scorecard: ScoreCard, sort_metric_key: str, metrics_to_display: List[str]
) -> List[str]:
    if len(metrics_to_display) == 0:
        metrics_to_display = [
            metric.key
            for metric in scorecard.get_all_metrics(only_evaluated=True)
            if isinstance(metric.result, (float, int))
        ]

    if sort_metric_key not in metrics_to_display:
        metrics_to_display.insert(0, sort_metric_key)
    else:
        metrics_to_display.remove(sort_metric_key)
        metrics_to_display.insert(0, sort_metric_key)
    return metrics_to_display


def compare(
    scorecards: List[ScoreCard],
    sort_metric_key: str = "rmse",
    metrics_to_display: List[str] = [],
    dataframe: bool = True,
) -> Union[pd.DataFrame, str]:
    metrics_to_display = __handle_empty_metrics_to_display(
        scorecards[0], sort_metric_key, metrics_to_display
    )

    scorecards.sort(reverse=True, key=lambda x: x[sort_metric_key].result)
    if dataframe:
        return pd.concat(
            [
                pd.Series(
                    [
                        bold(f"{scorecard[metric_key].result:>10.5}", rich=False)
                        if metric_key == sort_metric_key
                        else scorecard[metric_key].result
                        for scorecard in scorecards
                    ],
                    name=bold(metric_key, rich=False)
                    if metric_key == sort_metric_key
                    else metric_key,
                    index=[scorecard.metadata.model_name for scorecard in scorecards],
                )
                for metric_key in metrics_to_display
            ],
            axis="columns",
        )
    else:
        metrics_to_display.remove(sort_metric_key)
        string_to_return = ""
        metric_title = "".join([f"{metric:<10s}" for metric in metrics_to_display])
        string_to_return += f"{'model_name':>30s}    {bold(f'{sort_metric_key:<10s}', rich=False)} {metric_title}"

        for scorecard in scorecards:
            metrics = "".join(
                [f"{scorecard[metric].result:<10.5}" for metric in metrics_to_display]
            )

            string_to_return += f"\n{scorecard.metadata.model_name:>30s}    {bold(f'{scorecard[sort_metric_key].result:<10.5}', rich=False)} {metrics}"

        return string_to_return
