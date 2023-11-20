"""
Classification with multiple labels and probabilities
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


from krisi import library, score
from krisi.utils.data import generate_random_classification

y, preds, probs, sample_weight = generate_random_classification(
    num_labels=5, num_samples=1000
)
score(
    y=y,
    predictions=preds,
    probabilities=probs,
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
    calculation="single",
    default_metrics=library.MetricRegistryClassification().multiclass_classification_metrics,
).print()
