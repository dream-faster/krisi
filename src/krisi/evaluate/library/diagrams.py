from typing import TYPE_CHECKING, List

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import plotly.graph_objects as go

from statsmodels.tsa.stattools import acf, pacf

from krisi.evaluate.type import MetricResult


def display_time_series(data: List[MetricResult], title: str = "") -> "go.Figure":
    import plotly.express as px

    df = pd.DataFrame(data, columns=[title])
    df["iteration"] = list(range(len(data)))
    fig = px.line(
        df,
        x="iteration",
        y=title,
    )
    fig.update_layout(title=title)
    return fig


def display_single_value(data: MetricResult, title: str = "") -> "go.Figure":
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            value=data,
            delta={},
            gauge={"axis": {"visible": False}},
            domain={"row": 0, "column": 0},
        )
    )

    fig.update_layout(title=title)
    return fig


def display_acf_plot(
    data: MetricResult,
    title: str = "",
    plot_pacf: bool = False,
) -> "go.Figure":
    import plotly.graph_objects as go

    title = (
        title + " - Partial Autocorrelation (PACF)"
        if plot_pacf
        else "Autocorrelation (ACF)"
    )

    if not isinstance(data, pd.Series):
        data = pd.Series(data)

    corr_array = (
        pacf(data.dropna(), alpha=0.05) if plot_pacf else acf(data.dropna(), alpha=0.05)
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
    fig.update_xaxes(range=[-1, len(corr_array[0]) + 2])
    fig.update_yaxes(zerolinecolor="#000000")

    fig.update_layout(title=title)
    return fig


def display_density_plot(
    data: MetricResult,
    title: str = "",
    plot_pacf: bool = False,
) -> "go.Figure":
    import plotly.express as px

    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    fig = px.histogram(data, marginal="box")  # or violin, rug
    fig.update_layout(title=title)

    return fig


def callibration_plot(
    data: MetricResult, title: str = "", bin_size: float = 0.1
) -> "go.Figure":
    import plotly.graph_objects as go

    # example data
    y_true = data["y"]
    y_prob = data["probs"]
    # y_true = np.array([0, 1, 1, 0, 1])
    # y_prob = np.array([0.2, 0.3, 0.6, 0.4, 0.8])

    # sort probabilities and corresponding true labels in ascending order
    order = np.argsort(y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    # calculate fraction of positives at each probability bin
    bins = np.arange(0, 1.1, bin_size)
    bin_indices = np.digitize(y_prob, bins)
    bin_counts = np.bincount(bin_indices, minlength=len(bins) + 1)
    fraction_positives = np.cumsum(bin_counts[:-1]) / np.sum(y_true)

    scatter = go.Scatter(
        x=bins, y=fraction_positives, mode="lines+markers", name="Data Points"
    )

    perfect_calibration = go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect Calibration",
        line=dict(color="black", dash="dash"),
    )

    layout = go.Layout(
        title="Calibration Curve",
        xaxis=dict(title="Predicted Probability", tickvals=bins),
        yaxis=dict(title="Fraction of Positives"),
        showlegend=True,
    )

    fig = go.Figure(data=[scatter, perfect_calibration], layout=layout)
    return fig
