"""
Default Metric Selection
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import score
from krisi.evaluate.library.default_metrics_regression import (
    all_regression_metrics,
    low_computation_regression_mterics,
    minimal_regression_metrics,
)

score(
    y=np.random.normal(0, 0.1, 1000),
    predictions=np.random.normal(0, 0.1, 1000),
    default_metrics=all_regression_metrics,  # This is the default
).print()

score(
    y=np.random.normal(0, 0.1, 1000),
    predictions=np.random.normal(0, 0.1, 1000),
    default_metrics=minimal_regression_metrics,
).print(input_analysis=False)

score(
    y=np.random.normal(0, 0.1, 1000),
    predictions=np.random.normal(0, 0.1, 1000),
    default_metrics=low_computation_regression_mterics,
).print(input_analysis=False)