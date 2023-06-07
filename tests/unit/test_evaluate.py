from krisi.evaluate.library.default_metrics_classification import cross_entropy
from krisi.evaluate.scorecard import ScoreCard
from krisi.utils.data import generate_random_classification


def test_sample_weights():
    y, predictions, probabilities, sample_weight = generate_random_classification(
        num_labels=3, num_samples=1000
    )
    sc = ScoreCard(
        y,
        predictions,
        probabilities,
        sample_weight=sample_weight,
        default_metrics=[cross_entropy],
    )
    sc.evaluate()
    sc.evaluate_over_time()
    sc.print()


test_sample_weights()
