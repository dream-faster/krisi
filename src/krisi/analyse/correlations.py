from functools import reduce
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from typing_extensions import Literal

from .utils import corr_without_symmetry, unroll


def get_unrolled_correlation_over_time(
    df_rolled_corr: List[pd.DataFrame], start_indecies: List[pd.DatetimeIndex]
) -> pd.DataFrame:
    df_unrolled_in_time = pd.concat(
        [unroll(df_, start_indecies[i]) for i, df_ in enumerate(df_rolled_corr)],
        axis="columns",
    ).T
    df_unrolled_in_time.index = pd.to_datetime(df_unrolled_in_time.index)
    return df_unrolled_in_time


def get_mean_corr_matrix(df_rolled_corr: List[pd.DataFrame]) -> pd.DataFrame:
    df_rolled_summed = reduce(lambda x, y: x.add(y, fill_value=0), df_rolled_corr)
    df_rolled_mean = df_rolled_summed / len(df_rolled_corr)

    return df_rolled_mean


def get_corr_rolled(
    df: pd.DataFrame, window: int, step: Optional[int] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    if step is None or step == 1 or step == 0:
        df_rolled = [df_ for df_ in df.rolling(window=window)][1:]
    else:
        df_rolled = [df_ for df_ in df.rolling(window=window, step=step)][1:]
    df_rolled_corr = [corr_without_symmetry(df_) for df_ in df_rolled]

    return df_rolled_corr, df_rolled


def create_rolled_correlation_metrics(
    df_rolled_corr: List[pd.DataFrame],
    start_indecies: List[pd.DatetimeIndex],
    threshold: float = 0.25,
) -> Tuple[
    List[pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series
]:
    coor_unrolled_in_time = get_unrolled_correlation_over_time(
        df_rolled_corr, start_indecies
    )

    # Correlations properties on unrolled correlation data
    corr_std = coor_unrolled_in_time.std()
    corr_mean = coor_unrolled_in_time.mean()

    # Mean correlations in a matrix for plotting into a network graph
    mean_corr_over_time = get_mean_corr_matrix(df_rolled_corr)

    mean_corr_over_time_treshold = mean_corr_over_time.where(
        mean_corr_over_time >= threshold, other=0.0
    )
    return (
        df_rolled_corr,
        mean_corr_over_time,
        mean_corr_over_time_treshold,
        coor_unrolled_in_time,
        corr_std,
        corr_mean,
    )


def create_summaries(
    corr_std: pd.Series,
    corr_mean: pd.Series,
    df_unrolled_in_time: pd.DataFrame,
    window: int,
    step: int,
    name: str,
    save_or_display: List[Literal["save", "display"]] = ["display"],
    save_location: str = "structured_data/plots",
    top_k: int = 2,
) -> None:
    fig, ax = plt.subplots(2, 1)
    title = f"Summary of {name} | window - {window} ~ step - {step}"
    fig.suptitle(title)

    # ranking = corr_mean/corr_std

    df_unrolled_in_time[corr_std.nsmallest(top_k).index.values].plot(
        title=f"Correlation over time (for top-{top_k} smallest standard deviation)",
        ax=ax[0],
        xticks=[],
    )
    df_unrolled_in_time[corr_mean.nlargest(top_k).index.values].plot(
        title=f"Correlation over time (for top-{top_k} largest mean)",
        ax=ax[1],
    )

    result = pd.concat([corr_std, corr_mean], axis="columns", keys=["std", "mean"])
    result.index.name = title
    print(result)

    if "save" in save_or_display:
        path_friendly = title.replace(" ", "_")
        plt.savefig(f"{save_location}/{path_friendly}.png", format="PNG")
    if "display" in save_or_display:
        plt.show()
