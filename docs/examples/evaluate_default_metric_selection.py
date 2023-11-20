"""
Default Metric Selection
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import library, score

score(
    y=np.random.random(1000),
    predictions=np.random.random(1000),
    default_metrics=library.MetricRegistryRegression().all_regression_metrics,  # This is the default
).print()

score(
    y=np.random.random(1000),
    predictions=np.random.random(1000),
    default_metrics=library.MetricRegistryRegression().minimal_regression_metrics,
).print(input_analysis=False)

score(
    y=np.random.random(1000),
    predictions=np.random.random(1000),
    default_metrics=library.MetricRegistryRegression().low_computation_regression_metrics,
).print(input_analysis=False)
