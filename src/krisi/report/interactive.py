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
            graph=dcc.Graph(id=figure.id),
            title=None,  # html.P("Select rolling window:"),
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
            className="h-full flex align-center m-2",
            id=figure.id,
            figure=figure.get_figure(),
            style={"display": "inline-block"},
        )


# def category_block()->html.Div:


def run_app(
    components: Dict[str, InteractiveFigure], global_controllers: List[PlotlyInput]
) -> None:
    app = Dash(__name__, external_scripts=external_script)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        className="p-24",
        children=[
            html.H1(
                "Time Series Analysis",
                className="py-3 text-5xl font-bold text-gray-800",
            ),
            html.Div(
                children=[
                    input_.type(
                        className="h-full min-w-[150px] flex justify-center align-center",
                        id=input_.id,
                        options=input_.options,
                        value=input_.default_value,
                        clearable=False,
                    )
                    for input_ in global_controllers
                ]
            ),
            html.Div(
                className="flex flex-wrap flex-row ",
                children=[
                    *[
                        html.Div(
                            children=[
                                html.P(category),
                                html.Div(
                                    className="flex flex-wrap flex-row m-4 shadow-lg",
                                    children=[
                                        *[
                                            figure_with_controller(figure)
                                            for figure in list_of_figures
                                        ],
                                    ],
                                ),
                            ]
                        )
                        for category, list_of_figures in components.items()
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
