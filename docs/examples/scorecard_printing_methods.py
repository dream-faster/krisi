"""
ScoreCard printing to console, three modes
===========================
"""
# mkdocs_gallery_thumbnail_path = 'images/example_thumnail.png'

import numpy as np

from krisi import score

scorecard = score(y=np.random.random(1000), predictions=np.random.random(1000))

scorecard.print()  # Same as "extended"
print(scorecard)  # Same as "minimal"
scorecard.print("extended")
scorecard.print("minimal")
scorecard.print("minimal_table")
