import numpy as np
from sklearn.metrics import mean_squared_error

from krisi.evaluate import SampleTypes, ScoreCard

sc = ScoreCard(
    model_name="<your_model_name>",
    dataset_name="<your_dataset_name>",
    sample_type=SampleTypes.insample,
)

target, predictions = np.random.rand(1000), np.random.rand(1000)

""" A predefined metrics """
sc.mse = mean_squared_error(target, predictions, squared=True)

""" Adding a new metric """
sc["own_metric"] = (target - predictions).mean()

""" Print scorecard summary """
sc.print_summary()
