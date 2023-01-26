import datetime
from typing import Callable, List, Optional, Tuple, Union

import plotly.express as px

from krisi.report.interactive import run_app
from krisi.report.pdf import append_sizes, convert_figures_to_html, create_pdf_report
from krisi.report.type import DisplayModes, InteractiveFigure, PlotlyInput
from krisi.utils.iterable_helpers import flatten, remove_nans


class Report:
    title: str
    modes: List[DisplayModes]

    def __init__(
        self,
        title: str,
        modes: List[DisplayModes] = [DisplayModes.pdf],
        figures: List[InteractiveFigure] = [],
        global_controllers: List[PlotlyInput] = [],
        html_template_url: str = "library/default/template.html",
        css_template_url: str = "library/default/template.css",
        get_html_elements: Optional[Callable] = None,
    ) -> None:
        self.title = title
        self.modes = modes
        self.figures = figures
        self.global_controllers = global_controllers
        self.html_template_url = html_template_url
        self.css_template_url = css_template_url
        self.get_html_elements = get_html_elements

    def generate_launch(self) -> None:

        if (
            DisplayModes.interactive in self.modes
            or DisplayModes.interactive.value in self.modes
        ):
            run_app(self.figures, self.global_controllers)

        if DisplayModes.pdf in self.modes or DisplayModes.pdf.value in self.modes:
            create_pdf_report(
                html_template_url=self.html_template_url,
                css_template_url=self.css_template_url,
                html_elements_to_inject=self.get_html_elements()
                if self.get_html_elements is not None
                else dict(),
            )

        if DisplayModes.direct in self.modes or DisplayModes.direct.value in self.modes:
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

    report = Report(title=f"Report on Apple", modes=[DisplayModes.pdf])
    report.add(
        [
            InteractiveFigure(
                id="time-series-chart",
                get_figure=display_time_series,
            ),
            InteractiveFigure(
                id="time-series-chart2",
                get_figure=display_time_series,
            ),
        ]
    )
    report.generate_launch()


def get_all_interactive_diagrams(metrics: List["Metric"]) -> List[InteractiveFigure]:
    interactive_diagrams = remove_nans(
        [metric.get_diagrams() for metric in metrics]
        + [metric.get_diagram_over_time() for metric in metrics]
    )
    return flatten(interactive_diagrams)


def get_waterfall_metric_html(
    metrics: List["Metric"],
) -> Tuple[str, List[InteractiveFigure]]:
    interactive_diagrams = get_all_interactive_diagrams(metrics)

    html_images = convert_figures_to_html(
        [
            interactive_diagram.get_figure(width=900.0)
            for interactive_diagram in interactive_diagrams
        ]
    )

    return html_images, interactive_diagrams


def get_html_elements_for_injection_scorecard(
    obj: "ScoreCard",
    author: str,
    project_name: str,
    date: str,
    custom_metric_html: str,
) -> Callable:
    def func() -> dict[str, str]:

        diagrams = obj.get_diagram_dictionary()
        diagrams = append_sizes(diagrams)

        diagrams_static = {
            interactive_figure.id: convert_figures_to_html(
                [
                    interactive_figure.get_figure(
                        width=interactive_figure.width,
                        height=interactive_figure.height,
                        title=interactive_figure.title,
                    )
                ]
            )
            for key, interactive_figure in diagrams.items()
        }

        return dict(
            author=author,
            project_name=project_name,
            date=date,
            custom_metric_html=custom_metric_html,
            **diagrams_static,
        )

    return func


def create_report_from_scorecard(
    obj: "ScoreCard",
    display_modes: List[DisplayModes],
    html_template_url: str,
    css_template_url: str,
    author: str,
) -> Report:

    custom_metric_html, interactive_figures = get_waterfall_metric_html(
        obj.get_all_metrics()
    )

    get_html_elements = None
    if DisplayModes.pdf in display_modes or DisplayModes.pdf.value in display_modes:
        get_html_elements = get_html_elements_for_injection_scorecard(
            obj=obj,
            author=author,
            project_name=obj.project_name,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            custom_metric_html=custom_metric_html,
        )

    return Report(
        title=f"{obj.project_name} - {obj.dataset_name} - {obj.model_name}",
        modes=display_modes,
        figures=interactive_figures,
        html_template_url=html_template_url,
        css_template_url=css_template_url,
        get_html_elements=get_html_elements,
    )
