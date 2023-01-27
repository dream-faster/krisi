import numpy as np

from krisi import compare, score

scorecards = [
    score(y=np.random.randint(0, 2, 1000), predictions=np.random.randint(0, 2, 1000))
    for _ in range(5)
]
compare(
    scorecards, sort_metric_key="accuracy", metrics_to_display=["recall", "f_one_score"]
)
