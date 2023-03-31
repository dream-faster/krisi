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

    plot_y_predictions(y, predictions, index_name="time", value_name="magnitude")


test_plot_y_predictions()
