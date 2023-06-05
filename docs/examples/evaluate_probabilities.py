"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np
import pandas as pd

from krisi import score

num_labels = 3
num_samples = 1000
probabilities = np.random.rand(num_samples, num_labels)
probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
probabilities = list(zip(*probabilities))

sc = score(
    y=np.random.randint(0, num_labels, num_samples),
    predictions=np.random.randint(0, num_labels, num_samples),
    probabilities=pd.DataFrame({i: probabilities[i] for i in range(num_labels)}),
    calculation="both"
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
)
sc.print()
sc.generate_report()
