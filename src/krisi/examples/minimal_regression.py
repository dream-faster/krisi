import numpy as np

from krisi import eval

eval(y=np.random.rand(1000), predictions=np.random.rand(1000)).print_summary()
