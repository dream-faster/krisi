import numpy as np

from krisi import evaluate

evaluate(
    y=np.random.randint(0, 2, 1000),
    predictions=np.random.randint(0, 2, 1000),
    classification=True,
).print_summary()
