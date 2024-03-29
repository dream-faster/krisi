"""
Compare Regression ScoreCards
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import compare, score

scorecards = [
    score(y=np.random.rand(1000), predictions=np.random.rand(1000)) for _ in range(5)
]
print(compare(scorecards, metric_keys=["mse", "mae"]))
print(
    compare(
        scorecards,
        metric_keys=["mse", "mae"],
        dataframe=False,
    ),
)
print(compare(scorecards, sort_by="mae"))
print(compare(scorecards, ["mse", "mae"]))
print(compare(scorecards, ["mae", "mse"]))
