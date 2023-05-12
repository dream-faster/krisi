"""
Analysing rolling correlations
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np
import pandas as pd

from dev_utils.utils import handle_test
from krisi.analyse import (
    get_rolled_corr_metrics,
    matrix_corr_over_time,
    plot_corr_over_time,
)
from krisi.report.graph import create_save_graphs

df = pd.read_csv(
    "https://raw.githubusercontent.com/dream-faster/datasets/main/datasets/energy/industrial_pv_load.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)[:1000]

df_returns = df.pct_change().fillna(0)
df_log_returns = pd.DataFrame(np.log(1 + df_returns))

window = 96  # One day measurement
step = 1

for name, df_ in [
    ("raw", df),
    ("returns", df_returns),
    ("log_returns", df_log_returns),
]:
    coor_unrolled_in_time, corr_std, corr_mean = get_rolled_corr_metrics(
        df_, window=window, step=step
    )
    plot_corr_over_time(df, window, step, name=name, save_or_display=["display"])
    create_save_graphs([matrix_corr_over_time(df, window, step, threshold=0.25)[1]])


handle_test()
