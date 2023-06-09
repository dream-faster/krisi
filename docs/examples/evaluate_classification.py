"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


from krisi import score
from krisi.utils.devutils.data import generate_random_classification

y, preds, probs, sample_weight = generate_random_classification(
    num_labels=2, num_samples=1000
)
score(
    y=y,
    predictions=preds,
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
).print()
