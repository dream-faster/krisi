"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


from krisi import library, score
from krisi.utils.data import generate_random_classification

y, preds, probs, sample_weight = generate_random_classification(
    num_labels=2, num_samples=1000
)
sc = score(
    y=y,
    predictions=preds,
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
    calculation="both",
    default_metrics=library.default_metrics_classification.binary_classification_metrics_balanced_benchmarking(),
)
sc.print()
