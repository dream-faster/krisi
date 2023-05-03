import pkgutil
from enum import Enum
from typing import TYPE_CHECKING, List, Optional, Union

import pandas as pd

from krisi.report.type import InteractiveFigure
from krisi.utils.iterable_helpers import wrap_in_list

if TYPE_CHECKING:
    import plotly.graph_objects as go


class VizualisationMethod(Enum):
    overlap = "overlap"
    seperate = "seperate"

    @staticmethod
    def from_str(value: Union[str, "VizualisationMethod"]) -> "VizualisationMethod":
        if isinstance(value, VizualisationMethod):
            return value
        for strategy in VizualisationMethod:
            if strategy.value == value:
                return strategy
        else:
            raise ValueError(f"Unknown VizualisationMethod: {value}")


def __calc_plot_names(
    df: pd.DataFrame,
    modes: List[VizualisationMethod],
    y_separate: bool,
    joint_plot_name: str,
    target_plot_name: str,
) -> List[str]:
    name_of_plots = [] + [target_plot_name] if y_separate else []

    for mode in modes:
        if mode == VizualisationMethod.seperate:
            name_of_plots += list(df.columns)

        if mode == VizualisationMethod.overlap:
            name_of_plots += [joint_plot_name]

    return name_of_plots


def __vizualise_with_plotly(
    y: pd.Series,
    preds: Union[pd.Series, List[pd.Series], pd.DataFrame],
    title: Optional[str],
    x_name: str,
    y_name: str,
    modes: List[VizualisationMethod],
    y_separate: bool,
    target_opacity: float,
    predictions_opacity: float,
    joint_plot_name: str,
    target_plot_name: str,
):
    import plotly as plt
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df = (
        pd.concat(preds, axis="columns")
        if isinstance(preds, List)
        else (preds.to_frame() if isinstance(preds, pd.Series) else preds)
    )
    df = df.reindex(y.index.union(df.index))

    name_of_plots = __calc_plot_names(
        df, modes, y_separate, joint_plot_name, target_plot_name
    )
    num_plots = len(name_of_plots)

    fig = make_subplots(
        rows=num_plots,
        cols=1,
        shared_xaxes=True,
        subplot_titles=name_of_plots,
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )
    y_trace = go.Scatter(
        x=df.index,
        y=y,
        name="y",
        line_color="red",
        legendwidth=0.0,
        legendgroup="y",
        showlegend=True,
        opacity=target_opacity,
    )
    traces = [
        go.Scatter(
            x=df.index,
            y=df[column],
            name=column,
            legendgroup="models",
            opacity=predictions_opacity,
            line_color=plt.colors.DEFAULT_PLOTLY_COLORS[i],
        )
        for i, column in enumerate(df.columns)
    ]

    if y_separate:
        y_trace.showlegend = False
        fig.add_trace(
            y_trace,
            row=1,
            col=1,
        )

    for mode in modes:
        if mode == VizualisationMethod.seperate:
            for i, column in enumerate(df.columns):
                fig.add_trace(
                    traces[i],
                    row=i + (2 if y_separate else 1),
                    col=1,
                )
                if not y_separate:
                    if i == 0:
                        y_trace.showlegend = True
                    else:
                        y_trace.showlegend = False
                    fig.add_trace(
                        y_trace,
                        row=i + 1,
                        col=1,
                    )

        if mode == VizualisationMethod.overlap:
            if not y_separate:
                if num_plots == 1:
                    y_trace.showlegend = True
                fig.add_trace(
                    y_trace,
                    row=num_plots,
                    col=1,
                )
            for i, column in enumerate(df.columns):
                fig.add_trace(
                    traces[i],
                    row=num_plots,
                    col=1,
                )

    for row in range(num_plots):
        fig.update_xaxes(title_text=x_name, row=row + 1, col=1)
        fig.update_yaxes(title_text=y_name, row=row + 1, col=1)
        fig.update_layout(**{f"xaxis{str(row+1)}_showticklabels": True})

    fig.update_layout(
        title=title,
        autosize=True,
        height=350.0 * num_plots,
        margin=dict(l=75, r=75, b=50, t=75, pad=0),
        hovermode="x unified",
    )
    fig.show()


def plot_y_predictions(
    y: pd.Series,
    preds: Union[List[pd.Series], pd.DataFrame],
    title: Optional[str] = "",
    x_name: str = "index",
    y_name: str = "value",
    mode: Union[str, VizualisationMethod, List[Union[str, VizualisationMethod]]] = [
        # VizualisationMethod.seperate,
        VizualisationMethod.overlap,
    ],
    y_separate: bool = False,
    target_opacity: float = 1.0,
    predictions_opacity: float = 0.75,
    joint_plot_name: str = "Joint Plot",
    target_plot_name: str = "Target (y)",
) -> None:
    modes = wrap_in_list(mode)
    if pkgutil.find_loader("plotly"):
        __vizualise_with_plotly(
            y,
            preds=preds,
            title=title,
            x_name=x_name,
            y_name=y_name,
            modes=[VizualisationMethod.from_str(mode) for mode in modes],
            y_separate=y_separate,
            target_opacity=target_opacity,
            predictions_opacity=predictions_opacity,
            joint_plot_name=joint_plot_name,
            target_plot_name=target_plot_name,
        )
    else:
        raise AssertionError(
            "`Plotly` is not installed. Install Plotly by running `pip install plotly` or unlock all reporting capability by `pip install krisi[plotting]`."
        )


def create_subplots_from_mutiple_plots(
    interactive_figures: List[InteractiveFigure], title: str
) -> "go.Figure":
    from plotly.subplots import make_subplots

    figures = [figure.get_figure(**figure.plot_args) for figure in interactive_figures]

    fig = make_subplots(
        rows=len(figures),
        cols=1,
        subplot_titles=[figure.title for figure in interactive_figures],
        row_heights=[500 for _ in range(len(figures))],
        specs=[[{"type": fig_.data[0].type}] for fig_ in figures],
    )

    for i, figure in enumerate(figures):
        fig.add_trace(figure.data[0], row=1 + i, col=1)

    fig.update_layout(height=500 * len(figures), title_text=title)

    return fig
