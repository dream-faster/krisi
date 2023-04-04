import pandas as pd

from krisi.report import plot_y_predictions
from krisi.utils.data import generating_arima_synthetic_data


def test_plot_y_predictions():
    target_col = "arima_synthetic"
    y = generating_arima_synthetic_data(target_col)
    y.name = "y"

    cutoff = 800
    predictions = pd.DataFrame([])
    predictions["first_model"] = y[cutoff:] * -1.0
    predictions["second_model"] = y.shift(5)[cutoff:]
    predictions["thrid_model"] = y.shift(10)[cutoff:]
    predictions["fourth_model"] = y.shift(20)[cutoff:]
    predictions["fifth_model"] = y.shift(30)[cutoff:]

    plot_y_predictions(
        y,
        predictions,
        x_name="time",
        y_name="magnitude",
        y_separate=False,
        mode=["overlap"],
    )


test_plot_y_predictions()
