import pandas as pd
from sklearn.model_selection import train_test_split

from krisi.evaluate import ScoreCard, evaluate_in_out_sample
from krisi.utils.data import generating_arima_synthetic_data
from krisi.utils.models import default_arima_model
from krisi.utils.runner import fit_model, generate_univariate_predictions

target_col = "arima_synthetic"
df = generating_arima_synthetic_data(target_col).to_frame(name=target_col)
train, test = train_test_split(df, test_size=0.2, shuffle=False)

fit_model(default_arima_model, train)

insample_prediction, outsample_prediction = generate_univariate_predictions(
    default_arima_model, test, target_col
)


report_insample, report_outsample = evaluate_in_out_sample(
    model_name="default_arima_model",
    dataset_name="synthetic_arima",
    y_insample=train[target_col],
    insample_predictions=insample_prediction,
    y_outsample=test[target_col],
    outsample_predictions=outsample_prediction,
)

print(report_insample)
