import datetime
import os
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Union

from krisi.evaluate.type import MetricCategories, PathConst, ScoreCardMetadata
from krisi.report.interactive import run_app
from krisi.report.pdf import (
    convert_figure_to_html,
    convert_figures_to_html,
    convert_figures_to_side_by_side_html,
    create_pdf_report,
)
from krisi.report.type import DisplayModes, InteractiveFigure, PlotlyInput
from krisi.report.vizualise import create_subplots_from_mutiple_plots
from krisi.utils.io import ensure_path
from krisi.utils.iterable_helpers import (
    flatten,
    group_by_categories,
    remove_nans,
    replace_if_None,
    wrap_in_list,
)

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric
    from krisi.evaluate.scorecard import ScoreCard


class Report:
    title: str
    modes: List[Union[str, DisplayModes]]

    def __init__(
        self,
        title: str,
        general_description: str = "",
        modes: Union[str, List[str], DisplayModes, List[DisplayModes]] = [
            DisplayModes.pdf
        ],
        figures: List[InteractiveFigure] = [],
        global_controllers: List[PlotlyInput] = [],
        html_template_url: Path = PathConst.html_report_template_url,
        css_template_url: Path = Path("library/default/template.css"),
        get_html_elements: Optional[Callable] = None,
        scorecard_metadata: Optional[ScoreCardMetadata] = None,
        save_path: Optional[Path] = None,
    ) -> None:
        self.title = title
        self.general_description = general_description
        self.scorecard_metadata = scorecard_metadata

        self.modes = [DisplayModes.from_str(mode) for mode in wrap_in_list(modes)]

        self.figures = figures
        self.global_controllers = global_controllers
        self.html_template_url = html_template_url
        self.css_template_url = css_template_url
        self.get_html_elements = get_html_elements
        self.save_path = replace_if_None(
            save_path,
            Path(os.path.join(PathConst.default_eval_output_path, Path(title))),
        )

    def generate_launch(self) -> None:
        figures_by_category = group_by_categories(
            self.figures, [el.value for el in MetricCategories]
        )

        if DisplayModes.pdf in self.modes:
            create_pdf_report(
                path=self.save_path,
                title=self.title,
                html_template_url=self.html_template_url,
                css_template_url=self.css_template_url,
                html_elements_to_inject=self.get_html_elements()
                if self.get_html_elements is not None
                else dict(),
            )

        if DisplayModes.direct in self.modes:
            [figure.get_figure(**figure.plot_args).show() for figure in self.figures]

        if DisplayModes.direct_save in self.modes:
            diagram_path = Path(os.path.join(self.save_path, Path("diagrams")))
            ensure_path(diagram_path)
            for i, figure in enumerate(self.figures):
                figure.get_figure(**figure.plot_args).write_image(
                    os.path.join(diagram_path, f"{i}_{figure.title}.svg")
                )

        if DisplayModes.direct_one_subplot in self.modes:
            create_subplots_from_mutiple_plots(self.figures, title=self.title).show()

        if DisplayModes.interactive in self.modes:
            run_app(
                figures_by_category,
                self.global_controllers,
                self.title,
                self.general_description,
                self.scorecard_metadata,
            )


def get_all_interactive_diagrams(metrics: List["Metric"]) -> List[InteractiveFigure]:
    interactive_diagrams = remove_nans(
        [metric.get_diagrams() for metric in metrics]
        + [metric.get_diagram_over_time() for metric in metrics]
    )
    return flatten(interactive_diagrams)


def get_group_html(
    group_name: str, metrics: List["Metric"]
) -> Tuple[str, List[InteractiveFigure]]:
    html_images = f'<section class="mb-6"><br><h3 class="mb-6 text-sm font-bold">{group_name}</h3><br>'
    interactive_diagrams = get_all_interactive_diagrams(metrics)

    interactive_diagrams.sort(key=lambda x: x.plot_args["width"], reverse=True)

    for i in range(len(interactive_diagrams)):
        if (
            i + 1 != len(interactive_diagrams)
            and interactive_diagrams[i].plot_args["width"] < 1000.0
            and interactive_diagrams[i + 1].plot_args["width"] < 1000.0
        ):
            html_images += convert_figures_to_side_by_side_html(
                [
                    interactive_diagrams[i].get_figure(
                        **interactive_diagrams[i].plot_args
                    ),
                    interactive_diagrams[i + 1].get_figure(
                        **interactive_diagrams[i + 1].plot_args
                    ),
                ]
            )
        else:
            html_images += convert_figure_to_html(
                interactive_diagrams[i].get_figure(**interactive_diagrams[i].plot_args)
            )

    html_images += "</section>"

    return html_images, interactive_diagrams


def get_waterfall_metric_html(
    metrics: List["Metric"],
) -> Tuple[str, List[InteractiveFigure]]:
    category_groups = group_by_categories(
        metrics, categories=[el.value for el in MetricCategories]
    )
    html_and_diagrams = [
        get_group_html(group_name, metrics)
        for group_name, metrics in category_groups.items()
        if metrics is not None and len(metrics) > 0 and group_name is not None
    ]
    return "".join([html for html, _ in html_and_diagrams]), flatten(
        [diagrams for _, diagrams in html_and_diagrams]
    )


def get_html_elements_for_injection_scorecard(
    obj: "ScoreCard",
    author: str,
    project_name: str,
    date: str,
    custom_metric_html: str,
) -> Callable:
    def func() -> Dict[str, str]:
        diagrams = obj.get_diagram_dictionary()

        diagrams_static = {
            interactive_figure.id: convert_figures_to_html(
                [
                    interactive_figure.get_figure(
                        title=interactive_figure.title,
                        **interactive_figure.plot_args,
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
    html_template_url: Path,
    css_template_url: Path,
    author: str,
    report_title: str,
    save_path: Path,
) -> Report:
    custom_metric_html, interactive_figures = get_waterfall_metric_html(
        obj.get_all_metrics()
    )

    get_html_elements = None
    if DisplayModes.pdf in display_modes or DisplayModes.pdf.value in display_modes:
        get_html_elements = get_html_elements_for_injection_scorecard(
            obj=obj,
            author=author,
            project_name=obj.metadata.project_name,
            date=datetime.datetime.now().strftime("%Y-%m-%d"),
            custom_metric_html=custom_metric_html,
        )

    return Report(
        title=report_title,
        general_description="General Description",
        modes=display_modes,
        scorecard_metadata=obj.metadata,
        figures=interactive_figures,
        html_template_url=html_template_url,
        css_template_url=css_template_url,
        get_html_elements=get_html_elements,
        save_path=save_path,
    )
