"""
Minimal Regression
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import score

score(y=np.random.random(1000), predictions=np.random.random(1000)).print()
