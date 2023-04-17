import numpy as np

from krisi import compare, score

scorecards = [
    score(y=np.random.randint(0, 2, 1000), predictions=np.random.randint(0, 2, 1000))
    for _ in range(5)
]
compare(scorecards, sort_by="accuracy", metric_keys=["recall", "f_one_score"])
