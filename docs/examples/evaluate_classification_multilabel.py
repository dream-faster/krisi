"""
Classification with multiple labels and probabilities
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np

from krisi import library, score

num_labels = 5
data_size = 1000
score(
    y=np.random.randint(0, num_labels, data_size),
    predictions=np.random.randint(0, num_labels, data_size),
    probabilities=np.random.uniform(0, 1, size=(data_size, num_labels)),
    # classification=True, # Optional, tries to decide based on if target contains integers
    calculation="single",
    default_metrics=library.multilabel_classification,
).print()
