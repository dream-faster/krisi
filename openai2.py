import os
import webbrowser
from threading import Timer

import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate
from flask import request

# Create a sample dataframe
df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 11, 12, 13]})

# Create a line plot using the dataframe
data = [go.Scatter(x=df["x"], y=df["y"], mode="lines")]

# Initialize the Dash app
app = dash.Dash()

app.layout = html.Div(
    [
        dcc.Graph(id="line-plot", figure={"data": data}),
    ]
)


@app.callback(
    dash.dependencies.Output("line-plot", "figure"),
    [dash.dependencies.Input("line-plot", "figure")],
)
def download_pdf(figure):
    raise PreventUpdate
    # Use the `to_image` method from Plotly to convert the plot to a PDF
    # and download the file automatically
    # dcc.Graph(figure=figure).to_image(format='pdf').download('plot.pdf')
    # return figure


def shutdown():
    func = request.environ.get("werkzeug.server.shutdown")
    if func is None:
        raise RuntimeError("Not running with the Werkzeug Server")
    func()


@app.server.before_first_request
def download_pdf_on_launch():
    fig = go.Figure(go.Scatter(x=df["x"], y=df["y"], mode="lines")).to_image(
        format="pdf"
    )
    open(f"here.pdf", "wb").write(fig)
    shutdown()

    # Use the `to_image` method from Plotly to convert the plot to a PDF
    # and download the file automatically
    # dcc.Graph(figure={"data": data}).to_image(format="pdf").download("plot.pdf")


def open_browser():
    if not os.environ.get("WERKZEUG_RUN_MAIN"):
        webbrowser.open_new("http://127.0.0.1:1222/")


if __name__ == "__main__":
    Timer(1, open_browser).start()
    app.run_server(debug=True, port=1222)
