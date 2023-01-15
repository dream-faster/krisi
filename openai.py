import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import plotly.graph_objs as go

# Create a sample dataframe
df = pd.DataFrame({"x": [1, 2, 3, 4], "y": [10, 11, 12, 13]})

# Create a line plot using the dataframe
data = [go.Scatter(x=df["x"], y=df["y"], mode="lines")]

# Initialize the Dash app
app = dash.Dash()

# Create the layout for the app
app.layout = html.Div(
    [
        dcc.Graph(id="line-plot", figure={"data": data}),
        html.A(
            id="download-link",
            children="Download PDF",
            download="plot.pdf",
            href="",
            target="_blank",
        ),
    ]
)

# Add a callback for the PDF download button
@app.callback(
    dash.dependencies.Output("download-link", "href"),
    [dash.dependencies.Input("line-plot", "figure")],
)
def update_link(figure):
    # Use the `to_image` method from Plotly to convert the plot to a PNG
    return figure.to_image(format="pdf")


# Run the app
if __name__ == "__main__":
    app.run_server(debug=True)
