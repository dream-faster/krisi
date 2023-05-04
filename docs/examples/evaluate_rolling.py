"""
Basic Loading
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import score
from krisi.evaluate.type import Calculation

datasize = 1000
print(
    score(
        y=np.random.rand(datasize),
        predictions=np.random.rand(datasize),
        calculation=Calculation.both,
    )
)
