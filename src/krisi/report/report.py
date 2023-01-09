from typing import List, Optional, Union

import plotly.express as px

from .interactive import run_app
from .pdf import create_pdf_report
from .type import DisplayModes, InteractiveFigure, PlotlyInput


class Report:
    title: str
    modes: List[DisplayModes]

    def __init__(
        self,
        title: str,
        modes: List[DisplayModes] = [DisplayModes.pdf],
        figures: List[InteractiveFigure] = [],
        global_controllers: List[PlotlyInput] = [],
    ) -> None:
        self.title = title
        self.modes = modes
        self.figures = figures
        self.global_controllers = global_controllers

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
