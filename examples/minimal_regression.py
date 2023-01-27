import numpy as np

from krisi import score

score(
    y=np.random.normal(0, 0.1, 1000), predictions=np.random.normal(0, 0.1, 1000)
).print_summary()
