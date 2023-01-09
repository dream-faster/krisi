# from typing import List, Optional

# import pandas as pd
# from dash import dcc

# from krisi.analyse.analysis import (
#     addfuller_test,
#     plot_acf_pacf,
#     plot_df,
#     plot_predict,
#     plot_rolling_mean,
#     plot_table,
# )
# from krisi.report.report import Report, plotly_interactive
# from krisi.report.type import DisplayModes, InteractiveFigure, PlotlyInput
# from krisi.utils.data import generating_arima_synthetic_data, make_it_stationary


# def eda_pipeline(
#     df: Optional[pd.DataFrame] = None, target_col: str = "target", plot: bool = False
# ):
#     report = Report(
#         title=f"EDA report on {target_col}", modes=[DisplayModes.interactive]
#     )

#     columns: List[str] = [f"{target_col}_{i}" for i in range(5)]
#     if df is None:
#         df = pd.DataFrame([])
#         for column in columns:
#             df[column] = generating_arima_synthetic_data(
#                 target_col, nsample=10000
#             ).to_frame()

#     if plot:
#         report.add_global_controller(
#             PlotlyInput(
#                 id="synthetic",
#                 type=dcc.Dropdown,
#                 value_name="value",
#                 options=columns,
#                 default_value=columns[0],
#             )
#         )
#         report.add(
#             [
#                 # InteractiveFigure(
#                 #     id="data-table",
#                 #     get_figure=plotly_interactive(plot_table, df),
#                 # ),
#                 # InteractiveFigure(
#                 #     id="data-describe",
#                 #     get_figure=plotly_interactive(plot_table, df.describe()),
#                 # ),
#                 InteractiveFigure(
#                     id="raw-data",
#                     get_figure=plotly_interactive(plot_df, df),
#                     title="Synthetic ARIMA data",
#                     global_input_ids=["synthetic"],
#                     inputs=[],
#                 ),
#                 InteractiveFigure(
#                     id="data-rolling-mean-std",
#                     get_figure=plotly_interactive(plot_rolling_mean, df),
#                     inputs=[
#                         PlotlyInput(
#                             id="rolling",
#                             type=dcc.Dropdown,
#                             value_name="value",
#                             options=[30, 60, 120, 360],
#                             default_value=30,
#                         ),
#                     ],
#                     global_input_ids=["synthetic"],
#                     title="Rolling means with upper and lower bounds",
#                 ),
#             ]
#         )

#     alpha = 0.05
#     p_value = addfuller_test(df[target_col])

#     print(f"Addfuller Test resulted in p_value: {p_value}, alpha is: {alpha}")
#     if p_value > alpha:
#         df[target_col] = make_it_stationary(df[target_col])

#     fitted_arimas = fit_multiple_arimas(
#         df[target_col],
#         order_ranges=(
#             [0, 1, 2],
#             [0],
#             [0, 1, 2],
#         ),
#     )
#     if plot:
#         report.add(
#             [
#                 InteractiveFigure(
#                     id="acf",
#                     get_figure=plotly_interactive(
#                         plot_acf_pacf,
#                         df[target_col],
#                         alpha=alpha,
#                     ),
#                 ),
#                 InteractiveFigure(
#                     id="pacf",
#                     get_figure=plotly_interactive(
#                         plot_acf_pacf, df[target_col], alpha=alpha, plot_pacf=True
#                     ),
#                     inputs=[("alpha", "value")],
#                 ),
#                 InteractiveFigure(
#                     id="best_model_prediction",
#                     get_figure=plotly_interactive(
#                         plot_predict,
#                         df[target_col],
#                         model_res=fitted_arimas[0],
#                     ),
#                 ),
#             ]
#         )

#     report.generate_launch()


# if __name__ == "__main__":
#     eda_pipeline(plot=True)
