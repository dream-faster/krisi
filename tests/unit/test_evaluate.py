from krisi.evaluate.scorecard import ScoreCard
from krisi.utils.data import generate_random_classification


def test_sample_weights():
    y, predictions, probabilities, sample_weight = generate_random_classification(
        num_labels=3, num_samples=1000
    )

    sc = ScoreCard(y, predictions, probabilities, sample_weight=sample_weight)
    sc.evaluate()
    sc.print()


test_sample_weights()
