import numpy as np

from krisi import score

score(y=np.random.rand(1000), predictions=np.random.rand(1000)).print_summary()
