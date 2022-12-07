from typing import Callable, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from krisi.report.interactive import run_app
from krisi.report.static import create_pdf_report
from krisi.report.types_ import DisplayModes, InteractiveFigure, PlotlyInput


def plotly_interactive(
    plot_function: Callable,
    data_source: Union[pd.DataFrame, pd.Series],
    *args,
    **kwargs
) -> Callable:
    default_args = args
    default_kwargs = kwargs

    def wrapper(*args, **kwargs) -> go.Figure:
        width = kwargs.pop("width", None)
        title = kwargs.pop("title", "")

        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        fig = plot_function(data_source, *args, **kwargs)

        fig.update_layout(width=width)
        if title is not None:
            fig.update_layout(title=title)
        return fig

    return wrapper


class Report:
    display_modes = DisplayModes
    figure_type = InteractiveFigure
    plotly_input = PlotlyInput

    def __init__(self, modes: List[DisplayModes]) -> None:
        self.modes = modes
        self.figures: List[InteractiveFigure] = []

    def generate_launch(self):
        if DisplayModes.pdf in self.modes:
            create_pdf_report(
                [figure.get_figure(width=900.0) for figure in self.figures]
            )

        if DisplayModes.interactive in self.modes:
            run_app(self.figures)

        if DisplayModes.direct in self.modes:
            [figure.get_figure(width=900.0).show() for figure in self.figures]

    def add(self, figures: Union[InteractiveFigure, List[InteractiveFigure]]) -> None:
        if isinstance(figures, List):
            self.figures += figures
        else:
            self.figures.append(figures)


if __name__ == "__main__":

    def display_time_series(
        ticker: Optional[str] = "AAPL", width: Optional[float] = None
    ):
        df = px.data.stocks()
        fig = px.line(df, x="date", y=ticker, width=width)
        return fig

    report = Report(modes=[DisplayModes.interactive])
    report.add(
        [
            InteractiveFigure(
                id="time-series-chart",
                get_figure=display_time_series,
                inputs=[("ticker", "value")],
            ),
            InteractiveFigure(
                id="time-series-chart2",
                get_figure=display_time_series,
                inputs=[("ticker", "value")],
            ),
        ]
    )

    report = Report(modes=[DisplayModes.interactive])
    report.generate_launch()
