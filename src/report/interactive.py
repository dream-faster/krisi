from dash import Dash, dcc, html, Input, Output
from typing import List
from .types import InteractiveFigure

external_script = ["https://tailwindcss.com/", {"src": "https://cdn.tailwindcss.com"}]


def block(graph: dcc.Graph, title: html.P, controllers: html.Div) -> html.Div:
    return html.Div(
        children=[graph, title, controllers],
        className="flex flex-row flex-wrap w-full min-h-[450px]",
    )


def run_app(components: List[InteractiveFigure]) -> None:
    app = Dash(__name__, external_scripts=external_script)
    app.scripts.config.serve_locally = True

    app.layout = html.Div(
        className="p-24",
        children=[
            html.H1(
                "Stock price analysis",
                className="py-3 text-5xl font-bold text-gray-800",
            ),
            html.Div(
                children=[
                    *[
                        block(
                            dcc.Graph(
                                id=component.id  # , style={"display": "inline-block"}
                            ),
                            html.P("Select rolling window:"),
                            html.Div(
                                className="w-full h-full flex align-center",
                                children=[
                                    input_.type(
                                        className="w-full h-full flex justify-center align-center",
                                        id=input_.id,
                                        options=input_.options,
                                        value=input_.default_value,
                                        clearable=False,
                                    )
                                    for input_ in component.inputs
                                ],
                            ),
                        )
                        if len(component.inputs) > 0
                        else dcc.Graph(
                            className="w-full h-full flex align-center",
                            id=component.id,
                            figure=component.get_figure(),
                            style={"display": "inline-block"},
                        )
                        for component in components
                    ],
                ]
            ),
        ],
    )

    for component in components:
        if len(component.inputs) > 0:
            app.callback(
                Output(component.id, "figure"),
                [Input(input_.id, input_.value_name) for input_ in component.inputs],
            )(component.get_figure)

    app.run_server(debug=True, threaded=True)
