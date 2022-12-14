# from typing import Callable, List, Optional, Union

# import numpy as np
# import pandas as pd
# import plotly.express as px
# import plotly.graph_objects as go
# from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
# from statsmodels.tsa.stattools import acf, adfuller, pacf

# pd.options.plotting.backend = "plotly"


# def plot_df(
#     df: pd.DataFrame, columns: Optional[Union[List[str], str]] = None
# ) -> go.Figure:
#     if columns is not None and len(columns) > 0:
#         if isinstance(columns, str):
#             columns = [columns]
#         df = df[columns]

#     fig = df.plot()
#     return fig


# def plot_rolling_mean(
#     df: pd.DataFrame, rolling: int = 52, columns: Optional[List[str]] = None
# ) -> go.Figure:
#     if columns:
#         df = df[columns]
#     else:
#         df = df.iloc[:, 0]

#     mean = df.rolling(window=rolling).mean()
#     std = df.rolling(window=rolling).std()

#     df_mean = mean.to_frame(name="mean")
#     df_mean["upper"] = mean + (2 * std)
#     df_mean["lower"] = mean - (2 * std)

#     fig = df_mean.plot()

#     return fig


# def plot_acf_pacf(
#     ds: pd.Series,
#     alpha: int,
#     plot_pacf: bool = False,
# ) -> go.Figure:
#     corr_array = (
#         pacf(ds.dropna(), alpha=alpha) if plot_pacf else acf(ds.dropna(), alpha=alpha)
#     )
#     lower_y = corr_array[1][:, 0] - corr_array[0]
#     upper_y = corr_array[1][:, 1] - corr_array[0]

#     fig = go.Figure()
#     [
#         fig.add_scatter(
#             x=(x, x), y=(0, corr_array[0][x]), mode="lines", line_color="#3f3f3f"
#         )
#         for x in range(len(corr_array[0]))
#     ]
#     fig.add_scatter(
#         x=np.arange(len(corr_array[0])),
#         y=corr_array[0],
#         mode="markers",
#         marker_color="#1f77b4",
#         marker_size=12,
#     )
#     fig.add_scatter(
#         x=np.arange(len(corr_array[0])),
#         y=upper_y,
#         mode="lines",
#         line_color="rgba(255,255,255,0)",
#     )
#     fig.add_scatter(
#         x=np.arange(len(corr_array[0])),
#         y=lower_y,
#         mode="lines",
#         fillcolor="rgba(32, 146, 230,0.3)",
#         fill="tonexty",
#         line_color="rgba(255,255,255,0)",
#     )
#     fig.update_traces(showlegend=False)
#     fig.update_xaxes(range=[-1, 42])
#     fig.update_yaxes(zerolinecolor="#000000")

#     title = "Partial Autocorrelation (PACF)" if plot_pacf else "Autocorrelation (ACF)"
#     fig.update_layout(title=title, width=900.0)

#     return fig


# def plot_predict(ds: pd.Series, model_res: ARIMAResults) -> go.Figure:

#     fig = go.Figure()
#     fig.add_scatter(x=list(range(len(ds))), y=ds.to_list(), mode="lines")
#     fig.add_scatter(
#         x=list(range(len(ds))),
#         y=model_res.predict(start=950, end=1010),
#         mode="lines",
#     )
#     title = "Predicting with ARIMA model"
#     fig.update_layout(title=title, width=900.0)

#     return fig


# def plot_table(df: pd.DataFrame) -> go.Figure:
#     fig = go.Figure(
#         data=[
#             go.Table(
#                 header=dict(
#                     values=["index"] + list(df.columns),
#                     fill_color="white",
#                     align="left",
#                 ),
#                 cells=dict(
#                     values=[df.index] + [df[column] for column in df.columns],
#                     fill_color="lightgrey",
#                     align="left",
#                 ),
#             )
#         ]
#     )

#     return fig


# def addfuller_test(ds: pd.Series) -> float:
#     """Check if we can reject the null-hypothesis, that is that the series is a random walk"""
#     result = adfuller(ds)
#     p_value = result[1]
#     return p_value
