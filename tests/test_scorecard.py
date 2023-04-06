import numpy as np

from krisi.evaluate import Metric, SampleTypes, ScoreCard


def test_slicing():
    target, predictions = np.random.rand(1000), np.random.rand(1000)
    sc = ScoreCard(
        target,
        predictions,
        model_name="<your_model_name>",
        dataset_name="<your_dataset_name>",
        sample_type=SampleTypes.insample,
    )

    """ A predefined metrics """
    sc.evaluate(defaults=True)

    """ Adding a new metric """
    sc["own_metric"] = (target - predictions).mean()

    sc = sc[["mse", "rmse"]]

    sc.print("minimal_table", title="Results of Naive")

    assert list(
        [key_ for key_ in sc.__dict__.keys() if isinstance(sc[key_], Metric)]
    ) == ["mse", "rmse"]
