import pkgutil
from enum import Enum
from typing import List, Optional, Union

import pandas as pd


class VizualisationMethod(Enum):
    overlapped = "overlapped"
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


def plot_y_predictions(
    y: pd.Series,
    preds: Union[List[pd.Series], pd.DataFrame],
    title: Optional[str] = "",
    x_name: str = "index",
    y_name: str = "value",
    variable_name: str = "models",
    modes: List[Union[str, VizualisationMethod]] = [
        VizualisationMethod.seperate,
        # VizualisationMethod.overlapped,
    ],
) -> None:
    modes = [VizualisationMethod.from_str(mode) for mode in modes]
    df = pd.concat(preds, axis="columns") if isinstance(preds, List) else preds

    if pkgutil.find_loader("plotly"):
        import plotly as plt
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        name_of_plots = []
        for mode in modes:
            if mode == VizualisationMethod.seperate:
                name_of_plots = name_of_plots + list(df.columns)
            if mode == VizualisationMethod.overlapped:
                name_of_plots.append("Joint Plot")
        num_plots = len(name_of_plots)
        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            subplot_titles=name_of_plots,
        )

        y_trace = go.Scatter(
            x=df.index,
            y=y,
            name="y",
            line_color="red",
            legendwidth=0.0,
            legendgroup="y",
            showlegend=True,
        )
        traces = [
            go.Scatter(
                x=df.index,
                y=df[column],
                name=column,
                legendgroup="models",
                opacity=0.5,
                line_color=plt.colors.DEFAULT_PLOTLY_COLORS[i],
            )
            for i, column in enumerate(df.columns)
        ]
        for mode in modes:
            if mode == VizualisationMethod.seperate:
                for i, column in enumerate(df.columns):
                    fig.append_trace(
                        traces[i],
                        row=i + 1,
                        col=1,
                    )
                    if i == 1:
                        y_trace.showlegend = False
                    fig.append_trace(
                        y_trace,
                        row=i + 1,
                        col=1,
                    )

            if mode == VizualisationMethod.overlapped:
                if num_plots == 1:
                    y_trace.showlegend = True
                fig.append_trace(
                    y_trace,
                    row=num_plots,
                    col=1,
                )
                for i, column in enumerate(df.columns):
                    fig.append_trace(
                        traces[i],
                        row=num_plots,
                        col=1,
                    )

        for row in range(num_plots):
            fig.update_xaxes(title_text=x_name, row=row + 1, col=1)
            fig.update_yaxes(title_text=y_name, row=row + 1, col=1)
        fig.update_layout(title=title)
        fig.show()
    else:
        raise AssertionError(
            "`Plotly` is not installed. Install Plotly by running `pip install plotly` or unlock all reporting capability by `pip install krisi[plotting]`."
        )
