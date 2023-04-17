import numpy as np

from krisi import score

scorecard = score(
    y=np.random.normal(0, 0.1, 1000), predictions=np.random.normal(0, 0.1, 1000)
)

scorecard.print()
scorecard.print("extended")
scorecard.print("minimal")
scorecard.print("minimal_table")
