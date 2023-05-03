"""
Quick Classification to Console
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'


import numpy as np

from krisi import score

score(
    y=np.random.randint(0, 2, 1000),
    predictions=np.random.randint(0, 2, 1000),
    # classification=True, # Optional, tries to decide based on if target contains integers
).print()