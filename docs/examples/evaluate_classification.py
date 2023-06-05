"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np
import pandas as pd

from krisi import score
from krisi.evaluate.library import multiclass_classification_metrics

probs = np.random.random(1000)
score(
    y=np.random.randint(0, 2, 1000),
    predictions=np.random.randint(0, 2, 1000),
    probabilities=pd.DataFrame({"0": probs, "1": 1 - probs}),
    default_metrics=multiclass_classification_metrics,
    calculation="rolling"
    # dataset_type="classification_multilabel", # if automatic inference of dataset type fails
).print()
