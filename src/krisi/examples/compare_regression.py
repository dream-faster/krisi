import numpy as np

from krisi import compare, eval

scorecards = [
    eval(y=np.random.rand(1000), predictions=np.random.rand(1000)) for _ in range(5)
]
compare(scorecards, sort_metric_key="rmse", metrics_to_display=["mse", "mae"])
