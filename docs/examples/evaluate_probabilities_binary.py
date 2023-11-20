"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np

from krisi import score
from krisi.utils.data import create_probabilities

num_labels = 2
num_samples = 1000
probabilities = create_probabilities(num_labels=num_labels, num_samples=num_samples)

sc = score(
    y=np.random.randint(0, num_labels, num_samples),
    predictions=np.random.randint(0, num_labels, num_samples),
    probabilities=probabilities,
    calculation="both"
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
)
sc.print()
sc.generate_report()
