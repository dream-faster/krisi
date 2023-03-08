from typing import TYPE_CHECKING, Dict, List, Optional

from krisi.evaluate.type import ScoreCardMetadata
from krisi.report.type import InteractiveFigure, PlotlyInput
from krisi.utils.iterable_helpers import flatten

if TYPE_CHECKING:
    from dash import dcc, html


external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]


def block(
    graph: "dcc.Graph",
    title: Optional["html.P"] = None,
    controllers: Optional["html.Div"] = None,
) -> "html.Div":
    return html.Div(
        children=[graph, title, controllers],
        className="flex flex-row flex-wrap",
    )


def figure_with_controller(figure: InteractiveFigure):
    if len(figure.inputs) > 0 or len(figure.global_input_ids) > 0:
        return block(
            graph=dcc.Graph(id=figure.id, className="h-full flex align-center w-1/2"),
            title=None,
            controllers=html.Div(
                className="h-full flex align-center",
                children=[
                    input_.type(
                        className="h-full min-w-[150px] flex justify-center align-center",
                        id=input_.id,
                        options=input_.options,
                        value=input_.default_value,
                        clearable=False,
                    )
                    for input_ in figure.inputs
                ],
            ),
        )

    else:
        return dcc.Graph(
            className="h-full flex align-center w-1/2",
            id=figure.id,
            figure=figure.get_figure(),
            style={"display": "inline-block"},
        )


def category_block(category: str, figures: List[InteractiveFigure]) -> "html.Div":
    return html.Div(
        className="flex flex-col shadow-lg mb-4",
        children=[
            html.Div(
                className="flex items-center w-full h-12 bg-yellow-400 ",
                children=[
                    html.H2(
                        category,
                        className="flex m-0 items-center text-lg font-normal text-white leading-normal mt-0 mb-2 pl-6",
                    )
                ],
            ),
            html.Div(
                className="flex flex-wrap flex-row  p-6",
                children=[
                    *[figure_with_controller(figure) for figure in figures],
                ],
            ),
        ],
    )


def global_input_controller_block(input_):
    return input_.type(
        className="h-full min-w-[150px] flex justify-center align-center",
        id=input_.id,
        options=input_.options,
        value=input_.default_value,
        clearable=False,
    )


def name_description_block(name: str, description: str) -> "html.Div":
    return html.Div(
        className="flex flex-col h-full w-full p-2",
        children=[
            html.H3(name, className="flex-auto text-xl font-semibold text-slate-600"),
            html.P(description),
        ],
    )


def create_description_component(
    scorecard_metadata: Optional[ScoreCardMetadata],
) -> Optional["html.Div"]:
    if scorecard_metadata is not None:
        return html.Div(
            className="flex flex-row w-full h-20 mb-12",
            children=[
                name_description_block(
                    scorecard_metadata.project_name,
                    scorecard_metadata.project_description,
                ),
                name_description_block(
                    scorecard_metadata.model_name, scorecard_metadata.model_description
                ),
                name_description_block(
                    scorecard_metadata.dataset_name,
                    scorecard_metadata.dataset_description,
                ),
            ],
        )
    else:
        return None


def run_app(
    components: Dict[str, List[InteractiveFigure]],
    global_controllers: List[PlotlyInput],
    title: str = "",
    description: str = "",
    scorecard_metadata: Optional[ScoreCardMetadata] = None,
) -> None:
    import sys

    from dash import Dash, Input, Output, dcc, html

    sys.modules[__name__].html = html
    sys.modules[__name__].dcc = dcc

    app = Dash(__name__, external_scripts=external_script)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        className="p-24",
        children=[
            html.H1(
                title,
                className="flex-auto text-3xl font-semibold text-yellow-400 p-2",
            ),
            create_description_component(scorecard_metadata),
            html.Div(
                children=[
                    global_input_controller_block(input_)
                    for input_ in global_controllers
                ]
            ),
            html.Div(
                className="flex flex-wrap flex-col",
                children=[
                    *[
                        category_block(category, figures)
                        for category, figures in components.items()
                        if len(figures) > 0
                    ],
                ],
            ),
        ],
    )

    for component in flatten(list(components.values())):
        if len(component.inputs) > 0 or len(component.global_input_ids) > 0:
            app.callback(
                Output(component.id, "figure"),
                [Input(input_.id, input_.value_name) for input_ in component.inputs]
                + [Input(input_id, "value") for input_id in component.global_input_ids],
            )(component.get_figure)

    app.run(debug=True, threaded=True)
