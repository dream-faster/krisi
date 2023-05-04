"""
Analysing rolling correlations
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np
import pandas as pd

from krisi.analyse.correlations import (
    create_summary,
    get_corr_rolled,
    get_rolled_corr_metrics,
)
from krisi.report.graph import create_save_graphs
from krisi.utils.io import ensure_path

measurement_per_hour = 4
hours_per_day = 24
days_per_week = 7
one_day_measurement = measurement_per_hour * hours_per_day
one_week_measurement = one_day_measurement * days_per_week

window = one_day_measurement
step = 1

df = pd.read_csv(
    "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets/energy/industrial_pv_load.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

df_returns = df.pct_change().fillna(0)
df_log_returns = pd.DataFrame(np.log(1 + df_returns))


for name, df_ in [
    # ("raw", df),
    # ("returns", df_returns),
    ("log_returns", df_log_returns),
]:
    save_location = f"output/analyse/correlations/{name}"
    ensure_path(save_location)

    df_rolled_corr, df_rolled = get_corr_rolled(df_, window, step)
    (
        df_rolled,
        df_rolled_mean,
        df_rolled_mean_treshold,
        coor_unrolled_in_time,
        corr_std,
        corr_mean,
    ) = get_rolled_corr_metrics(df_rolled_corr, [df_.index[0] for df_ in df_rolled])

    create_summary(
        corr_std,
        corr_mean,
        coor_unrolled_in_time,
        window,
        step,
        name=name,
        save_or_display=["display"],
        save_location=save_location,
    )

    create_save_graphs([df_rolled_mean_treshold], save_location)
