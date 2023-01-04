import numpy as np

from krisi import evaluate
from krisi.evaluate.compare import compare

scorecards = [
    evaluate(y=np.random.rand(1000), predictions=np.random.rand(1000)) for _ in range(5)
]
compare(scorecards)
