from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve

if TYPE_CHECKING:
    import plotly.graph_objects as go

from krisi.evaluate.type import MetricResult


def display_time_series(data: List[MetricResult], **kwargs) -> "go.Figure":
    import plotly.express as px

    title = kwargs.get("title", "")
    df = pd.DataFrame(data, columns=[title])
    df["iteration"] = list(range(len(data)))
    fig = px.line(
        df,
        x="iteration",
        y=title,
    )
    return fig


def display_single_value(data: MetricResult, **kwargs) -> "go.Figure":
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

    return fig


def display_acf_plot(
    data: MetricResult, plot_pacf: bool = False, **kwargs
) -> "go.Figure":
    import plotly.graph_objects as go
    from statsmodels.tsa.stattools import acf, pacf

    title = "Partial Autocorrelation (PACF)" if plot_pacf else "Autocorrelation (ACF)"

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


def display_density_plot(data: MetricResult, **kwargs) -> "go.Figure":
    import plotly.express as px

    if not isinstance(data, pd.Series):
        data = pd.Series(data)
    fig = px.histogram(data, marginal="box")  # or violin, rug

    return fig


def calculate_calibration_bins(
    data: MetricResult, pos_label: int, n_bins: int, **kwargs
) -> Tuple[np.ndarray, np.ndarray]:
    y_true = data["y"]
    y_prob = data["probs"]

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, pos_label=pos_label, n_bins=n_bins
    )
    return prob_true, prob_pred


def callibration_plot(
    data: MetricResult, n_bins: int = 8, pos_label: int = 1, **kwargs
) -> "go.Figure":
    import plotly.graph_objects as go

    fraction_positives, bins = calculate_calibration_bins(data, pos_label, n_bins)
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
