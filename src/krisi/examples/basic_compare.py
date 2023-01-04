import numpy as np

from krisi import compare, evaluate

scorecards = [
    evaluate(y=np.random.rand(1000), predictions=np.random.rand(1000)) for _ in range(5)
]
compare(scorecards, metrics_to_display=["mse", "mae"])
