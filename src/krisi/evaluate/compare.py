from typing import List, Optional, Union

import pandas as pd

from krisi.evaluate.scorecard import ScoreCard
from krisi.evaluate.utils import handle_empty_metrics_to_display
from krisi.utils.printing import bold


def compare(
    scorecards: List[ScoreCard],
    metric_keys: Optional[List[str]] = None,
    sort_by: Optional[str] = None,
    dataframe: bool = True,
) -> Union[pd.DataFrame, str]:
    """Creates a table where each column is a metric and each row is
    a scorecard and its corresponding results.

    Parameters
    ----------
    scorecards : List[ScoreCard]
        ScoreCards to compare.
    metric_keys : Optional[List[str]], optional
        List of metrics to display. If not set it will return all
        evaluated metrics on the first scorecard.
        Sorts the results by the first element of this list if `sort_by` is not specified., by default None
    sort_by : Optional[str], optional
        `Metric` to sort results by. Selected `Metric` will be displayed in the first row.
        If not specified metrics will be sorted by the first element of `metric_keys`.
        If `metric_keys` is not specified it will default to the first metric found on the first `ScoreCard`, by default None
    dataframe : bool, optional
        Whether it should return a `pd.DataFrame` or a `str`, by default True

    Returns
    -------
    Union[pd.DataFrame, str]
        A comparison table, either in `pd.DataFrame` or `string` format.
    """

    metric_keys, sort_by = handle_empty_metrics_to_display(
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
