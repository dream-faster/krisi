from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from krisi.evaluate import ScoreCard, evaluate_in_out_sample
from krisi.evaluate.type import CalculationTypes
from krisi.utils.data import generating_arima_synthetic_data
from krisi.utils.iterable_helpers import type_converter
from krisi.utils.models import default_arima_model
from krisi.utils.runner import fit_model, generate_univariate_predictions


def basic_report_rolling() -> Tuple[ScoreCard, ScoreCard]:
    """Create syntethic Data"""
    target_col = "arima_synthetic"
    df = generating_arima_synthetic_data(target_col).to_frame(name=target_col)

    """ Split Data into Train and Test splits"""
    train, test = train_test_split(df, test_size=0.2, shuffle=False)
    train, test = type_converter(pd.DataFrame)([train, test])

    """ Train an ARIMA model """
    fit_model(default_arima_model, train)

    """ Generate Predictions with the model """
    insample_prediction, outsample_prediction = generate_univariate_predictions(
        default_arima_model, train, len(test), target_col
    )

    """ Evaluate predictions """
    report_insample, report_outsample = evaluate_in_out_sample(
        model_name="default_arima_model",
        dataset_name="synthetic_arima",
        calculation_types=[CalculationTypes.rolling],
        y_insample=train[target_col],
        insample_predictions=insample_prediction,
        y_outsample=test[target_col],
        outsample_predictions=outsample_prediction,
    )

    """ Console log Reports """
    print(report_outsample)
    report_insample.print_summary()

    return report_insample, report_outsample


if __name__ == "__main__":
    basic_report_rolling()
