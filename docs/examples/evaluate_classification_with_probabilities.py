"""
Classification with Probabilities
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np

from krisi import score

sc = score(
    y=np.random.randint(0, 2, 1000),
    predictions=np.random.randint(0, 2, 1000),
    probabilities=np.random.uniform(0, 1, (1000, 1)),
    # dataset_type="classification_binary", # if automatic inference of dataset type fails
    calculation="single",
)
sc.print()
sc.generate_report()

score(
    y=np.random.randint(0, 1, 1000),
    predictions=np.random.randint(0, 1, 1000),
    probabilities=np.random.uniform(0, 1, (1000, 1)),
    # dataset_type="classification_binary", # if automatic inference of dataset type fails
    calculation="rolling",
).print()
