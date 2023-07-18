from functools import reduce
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from typing_extensions import Literal

from krisi.evaluate.type import PathConst
from krisi.report.graph import create_save_graphs
from krisi.utils.state import RunType, get_global_state

from .utils import corr_without_symmetry, unroll


def get_mean_corr_matrix(df_rolled_corr: List[pd.DataFrame]) -> pd.DataFrame:
    df_rolled_summed = reduce(lambda x, y: x.add(y, fill_value=0), df_rolled_corr)
    df_rolled_mean = df_rolled_summed / len(df_rolled_corr)

    return df_rolled_mean


def get_unrolled_correlation_over_time(
    df_rolled_corr: List[pd.DataFrame], start_indecies: List[pd.DatetimeIndex]
) -> pd.DataFrame:
    df_unrolled_in_time = pd.concat(
        [unroll(df_, start_indecies[i]) for i, df_ in enumerate(df_rolled_corr)],
        axis="columns",
    ).T
    df_unrolled_in_time.index = pd.to_datetime(df_unrolled_in_time.index)
    return df_unrolled_in_time


def get_corr_rolled(
    df: pd.DataFrame, window: int, step: Optional[int] = None
) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
    if step is None or step == 1 or step == 0:
        df_rolled = list(df.rolling(window=window))[1:]
    else:
        df_rolled = list(df.rolling(window=window, step=step))[1:]
    df_rolled_corr = [corr_without_symmetry(df_) for df_ in df_rolled]

    return df_rolled_corr, df_rolled


def __corr_mean(df_rolled_corr: List[pd.DataFrame], threshold: float):
    # Mean correlations in a matrix for plotting into a network graph
    mean_corr_over_time = get_mean_corr_matrix(df_rolled_corr)

    mean_corr_over_time_treshold = mean_corr_over_time.where(
        mean_corr_over_time >= threshold, other=0.0
    )

    return mean_corr_over_time, mean_corr_over_time_treshold


def __create_summary(
    corr_std: pd.Series,
    corr_mean: pd.Series,
    df_unrolled_in_time: pd.DataFrame,
    window: int,
    step: int,
    name: str,
    save_or_display: List[Literal["save", "display"]],
    save_location: str,
    top_k: int,
) -> None:
    import matplotlib.pyplot as plt

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
        if not get_global_state().run_type == RunType.test:
            plt.show()


def get_rolled_corr_metrics(
    df: pd.DataFrame,
    window: int,
    step: int = 1,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    df_rolled_corr, df_rolled = get_corr_rolled(df, window, step)

    start_indecies = [df_.index[0] for df_ in df_rolled]

    coor_unrolled_in_time = get_unrolled_correlation_over_time(
        df_rolled_corr, start_indecies
    )

    # Correlations properties on unrolled correlation data
    corr_std = coor_unrolled_in_time.std()
    corr_mean = coor_unrolled_in_time.mean()

    return (
        coor_unrolled_in_time,
        corr_std,
        corr_mean,
    )


def matrix_corr_over_time(
    df: pd.DataFrame,
    window: int,
    step: int = 1,
    threshold: float = 0.25,
    save_location: Path = PathConst.default_analyse_output_path,
    save_graphs: bool = False,
):
    df_rolled_corr, df_rolled = get_corr_rolled(df, window, step)
    df_rolled_mean, df_rolled_mean_treshold = __corr_mean(
        df_rolled_corr, threshold=threshold
    )

    if save_graphs:
        create_save_graphs([df_rolled_mean_treshold], save_location)

    return df_rolled_mean, df_rolled_mean_treshold


def plot_corr_over_time(
    df: pd.DataFrame,
    window: int,
    step: int = 1,
    name: str = "",
    save_or_display: List[Literal["save", "display"]] = ["display"],
    save_location: str = "output/structured_data/plots",
    top_k: int = 2,
) -> None:
    (
        coor_unrolled_in_time,
        corr_std,
        corr_mean,
    ) = get_rolled_corr_metrics(df, window, step)

    __create_summary(
        corr_std,
        corr_mean,
        coor_unrolled_in_time,
        window,
        step,
        name=name,
        save_or_display=save_or_display,
        save_location=save_location,
        top_k=top_k,
    )
