"""
Classification with Probabilities
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


from krisi import score
from krisi.evaluate.type import Calculation
from krisi.utils.data import generate_random_classification

y, preds, probs, sample_weight = generate_random_classification(
    num_labels=2, num_samples=1000
)
sc = score(
    y=y,
    predictions=preds,
    probabilities=probs,
    # dataset_type="classification_binary_balanced", # if automatic inference of dataset type fails
    calculation="single",
)
sc.print()

sc = score(
    y=y,
    predictions=preds,
    probabilities=probs,
    # dataset_type="classification_binary_balanced", # if automatic inference of dataset type fails
    calculation=[Calculation.single, Calculation.rolling],
)
sc.print()
sc.generate_report()
