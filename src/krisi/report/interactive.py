from typing import Dict, List, Optional

from dash import Dash, Input, Output, dcc, html

from krisi.report.type import InteractiveFigure, PlotlyInput
from krisi.utils.iterable_helpers import flatten

external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]


def block(
    graph: dcc.Graph,
    title: Optional[html.P] = None,
    controllers: Optional[html.Div] = None,
) -> html.Div:
    return html.Div(
        children=[graph, title, controllers],
        className="flex flex-row flex-wrap min-h-[450px]",
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


def category_block(category: str, figures: List[InteractiveFigure]) -> html.Div:
    return html.Div(
        className="flex flex-col shadow-lg mb-4 p-6",
        children=[
            html.H2(
                category, className="text-2xl font-normal leading-normal mt-0 mb-2"
            ),
            html.Div(
                className="flex flex-wrap flex-row",
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


def run_app(
    components: Dict[str, List[InteractiveFigure]],
    global_controllers: List[PlotlyInput],
    title: str = "",
    description: str = "",
) -> None:
    app = Dash(__name__, external_scripts=external_script)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        className="p-24",
        children=[
            html.H1(
                title,
                className="py-3 text-5xl font-bold text-gray-800",
            ),
            html.P(description),
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
