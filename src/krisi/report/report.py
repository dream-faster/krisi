from typing import Callable, List, Optional, Union

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from krisi.report.interactive import run_app
from krisi.report.static import create_pdf_report
from krisi.report.type import (
    DisplayModes,
    InteractiveFigure,
    PlotlyInput,
    plotly_interactive,
)


class Report:
    title: str
    display_modes = DisplayModes
    figure_type = InteractiveFigure
    plotly_input = PlotlyInput

    def __init__(self, title: str, modes: List[DisplayModes]) -> None:
        self.title = title
        self.modes = modes
        self.figures: List[InteractiveFigure] = []
        self.global_controllers: List[PlotlyInput] = []

    def generate_launch(self):
        if DisplayModes.pdf in self.modes:
            create_pdf_report(
                [figure.get_figure(width=900.0) for figure in self.figures],
                title=self.title,
            )

        if DisplayModes.interactive in self.modes:
            run_app(self.figures, self.global_controllers)

        if DisplayModes.direct in self.modes:
            [figure.get_figure(width=900.0).show() for figure in self.figures]

    def add(self, figures: Union[InteractiveFigure, List[InteractiveFigure]]) -> None:
        if isinstance(figures, List):
            self.figures += figures
        else:
            self.figures.append(figures)

    def add_global_controller(
        self, controller: Union[PlotlyInput, List[PlotlyInput]]
    ) -> None:
        if isinstance(controller, List):
            self.global_controllers += controller
        else:
            self.global_controllers.append(controller)


if __name__ == "__main__":

    def display_time_series(
        ticker: Optional[str] = "AAPL", width: Optional[float] = None
    ):
        df = px.data.stocks()
        fig = px.line(df, x="date", y=ticker, width=width)
        return fig

    report = Report(title=f"Report on Apple", modes=[DisplayModes.interactive])
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
    report.generate_launch()
