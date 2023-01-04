import numpy as np

from krisi import evaluate

evaluate(y=np.random.rand(1000), predictions=np.random.rand(1000)).save()
