from typing import Optional

import pandas as pd
from dash import dcc

from krisi.explore.analysis import (
    addfuller_test,
    plot_acf_pacf,
    plot_df,
    plot_predict,
    plot_rolling_mean,
    plot_table,
)
from krisi.explore.modeling import fit_multiple_arimas
from krisi.explore.utils import generating_arima_synthetic_data, make_it_stationary
from krisi.report.report import Report, plotly_interactive
from krisi.report.types_ import DisplayModes, InteractiveFigure, PlotlyInput


def eda_pipeline(
    df: Optional[pd.DataFrame] = None, target_col: str = "target", plot: bool = False
):
    report = Report(modes=[DisplayModes.pdf, DisplayModes.interactive])

    if df is None:
        df = generating_arima_synthetic_data(target_col, nsample=10000).to_frame()

    if plot:
        report.add(
            [
                InteractiveFigure(
                    id="data-table",
                    get_figure=plotly_interactive(plot_table, df),
                ),
                InteractiveFigure(
                    id="data-describe",
                    get_figure=plotly_interactive(plot_table, df.describe()),
                ),
                InteractiveFigure(
                    id="raw-data",
                    get_figure=plotly_interactive(plot_df, df),
                    title="Synthetic ARIMA data",
                ),
                InteractiveFigure(
                    id="data-rolling-mean-std",
                    get_figure=plotly_interactive(plot_rolling_mean, df[target_col]),
                    inputs=[
                        PlotlyInput(
                            id="rolling",
                            type=dcc.Dropdown,
                            value_name="value",
                            options=[30, 60, 120, 360],
                            default_value=30,
                        )
                    ],
                    title="Rolling means with upper and lower bounds",
                ),
            ]
        )

    alpha = 0.05
    p_value = addfuller_test(df[target_col])

    print(f"Addfuller Test resulted in p_value: {p_value}, alpha is: {alpha}")
    if p_value > alpha:
        df[target_col] = make_it_stationary(df[target_col])

    fitted_arimas = fit_multiple_arimas(
        df[target_col],
        order_ranges=(
            [0, 1, 2],
            [0],
            [0, 1, 2],
        ),
    )
    if plot:
        report.add(
            [
                InteractiveFigure(
                    id="acf",
                    get_figure=plotly_interactive(
                        plot_acf_pacf,
                        df[target_col],
                        alpha=alpha,
                    ),
                ),
                InteractiveFigure(
                    id="pacf",
                    get_figure=plotly_interactive(
                        plot_acf_pacf, df[target_col], alpha=alpha, plot_pacf=True
                    ),
                    inputs=[("alpha", "value")],
                ),
                InteractiveFigure(
                    id="best_model_prediction",
                    get_figure=plotly_interactive(
                        plot_predict,
                        df[target_col],
                        model_res=fitted_arimas[0],
                    ),
                ),
            ]
        )

    report.generate_launch()


eda_pipeline(plot=True)
