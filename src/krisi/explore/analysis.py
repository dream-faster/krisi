from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA, ARIMAResults
from statsmodels.tsa.stattools import acf, adfuller, pacf


def plot_df(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df.iloc[:, 0].to_list(), mode="lines")
    return fig


def plot_rolling_mean(
    ds: pd.Series,
    rolling: int = 52,
) -> go.Figure:
    fig = go.Figure()

    colors = px.colors.qualitative.Plotly

    mean = ds.rolling(window=rolling).mean()
    std = ds.rolling(window=rolling).std()

    df = mean.to_frame(name="mean")
    df["upper"] = mean + (2 * std)
    df["lower"] = mean - (2 * std)

    fig.add_traces(
        go.Scatter(
            x=ds.index,
            y=mean,
            mode="lines",
            name=f"rolling mean {rolling}",
            line=dict(color=colors[0]),
        )
    )
    fig.add_traces(
        go.Scatter(
            x=ds.index,
            y=df["upper"],
            mode="lines",
            name="upper bounds (2*std)",
            line=dict(color=colors[1]),
        )
    )
    fig.add_traces(
        go.Scatter(
            x=ds.index,
            y=df["lower"],
            name="lower bounds (2*std)",
            mode="lines",
            line=dict(color=colors[2]),
        )
    )

    return fig


def plot_acf_pacf(
    ds: pd.Series,
    alpha: int,
    plot_pacf: bool = False,
) -> go.Figure:
    corr_array = (
        pacf(ds.dropna(), alpha=alpha) if plot_pacf else acf(ds.dropna(), alpha=alpha)
    )
    lower_y = corr_array[1][:, 0] - corr_array[0]
    upper_y = corr_array[1][:, 1] - corr_array[0]

    fig = go.Figure()
    [
        fig.add_scatter(
            x=(x, x), y=(0, corr_array[0][x]), mode="lines", line_color="#3f3f3f"
        )
        for x in range(len(corr_array[0]))
    ]
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=corr_array[0],
        mode="markers",
        marker_color="#1f77b4",
        marker_size=12,
    )
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=upper_y,
        mode="lines",
        line_color="rgba(255,255,255,0)",
    )
    fig.add_scatter(
        x=np.arange(len(corr_array[0])),
        y=lower_y,
        mode="lines",
        fillcolor="rgba(32, 146, 230,0.3)",
        fill="tonexty",
        line_color="rgba(255,255,255,0)",
    )
    fig.update_traces(showlegend=False)
    fig.update_xaxes(range=[-1, 42])
    fig.update_yaxes(zerolinecolor="#000000")

    title = "Partial Autocorrelation (PACF)" if plot_pacf else "Autocorrelation (ACF)"
    fig.update_layout(title=title, width=900.0)

    return fig


def plot_predict(ds: pd.Series, model_res: ARIMAResults) -> go.Figure:

    fig = go.Figure()
    fig.add_scatter(x=list(range(len(ds))), y=ds.to_list(), mode="lines")
    fig.add_scatter(
        x=list(range(len(ds))),
        y=model_res.predict(start=950, end=1010),
        mode="lines",
    )
    title = "Predicting with ARIMA model"
    fig.update_layout(title=title, width=900.0)

    return fig


def plot_table(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["index"] + list(df.columns),
                    fill_color="white",
                    align="left",
                ),
                cells=dict(
                    values=[df.index] + [df[column] for column in df.columns],
                    fill_color="lightgrey",
                    align="left",
                ),
            )
        ]
    )

    return fig


def addfuller_test(ds: pd.Series) -> float:
    """Check if we can reject the null-hypothesis, that is that the series is a random walk"""
    result = adfuller(ds)
    p_value = result[1]
    return p_value
