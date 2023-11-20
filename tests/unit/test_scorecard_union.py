from krisi import library, score
from krisi.evaluate.library.benchmarking_models import RandomClassifier
from krisi.utils.data import generate_random_classification


def test_scorecard_union():
    y, predictions, probabilities, sample_weight = generate_random_classification(
        num_labels=2, num_samples=1000
    )
    sc_1 = score(
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        sample_weight=sample_weight,
        default_metrics=library.MetricRegistryClassification().binary_classification_balanced_metrics,
        benchmark_models=RandomClassifier(),
    )
    sc_2 = score(
        y=y,
        predictions=predictions,
        probabilities=probabilities,
        sample_weight=sample_weight,
        default_metrics=library.MetricRegistryClassification().binary_classification_balanced_metrics,
        benchmark_models=RandomClassifier(),
    )
    metric_key = "precision_binary"
    sc_substracted = sc_1.subtract(sc_2)
    sc_added = sc_1.add(sc_2)
    sc_multiplied = sc_1.multiply(sc_2)
    sc_divided = sc_1.divide(sc_2)

    assert (
        sc_substracted[metric_key].result
        == sc_1[metric_key].result - sc_2[metric_key].result
    )
    assert (
        sc_added[metric_key].result == sc_1[metric_key].result + sc_2[metric_key].result
    )
    assert (
        sc_multiplied[metric_key].result
        == sc_1[metric_key].result * sc_2[metric_key].result
    )
    assert (
        sc_divided[metric_key].result
        == sc_1[metric_key].result / sc_2[metric_key].result
    )
    assert (
        sc_substracted[metric_key].comparison_result
        == sc_1[metric_key].comparison_result - sc_2[metric_key].comparison_result
    ).all() == True
    assert (
        sc_added[metric_key].comparison_result
        == sc_1[metric_key].comparison_result + sc_2[metric_key].comparison_result
    ).all() == True
    assert (
        sc_multiplied[metric_key].comparison_result
        == sc_1[metric_key].comparison_result * sc_2[metric_key].comparison_result
    ).all() == True
    assert (
        sc_divided[metric_key].comparison_result
        == sc_1[metric_key].comparison_result / sc_2[metric_key].comparison_result
    ).all() == True
