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
    probabilities=np.random.uniform(0, 1, 1000),
    # classification=True, # Optional, tries to decide based on if target contains integers
    calculation="single",
)

sc.print()
sc.generate_report()

score(
    y=np.random.randint(0, 2, 1000),
    predictions=np.random.randint(0, 2, 1000),
    probabilities=np.random.uniform(0, 1, 1000),
    # classification=True, # Optional, tries to decide based on if target contains integers
    calculation="rolling",
).print()
