from time import monotonic

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

np.random.seed(42)
func = mean_absolute_error
lengthofdataset = 1000000

y = pd.Series(np.random.rand(lengthofdataset), name="y")
predictions = pd.Series(np.random.rand(lengthofdataset), name="predictions")


window = 30


def timeingit(func):
    start_time = monotonic()
    res = func()
    print(f"Run time: {monotonic() - start_time} seconds")
    return res


def func1():
    return [
        func(y_roll, predictions[y_roll.index])
        for y_roll in y.rolling(window=window, step=window)
    ]


def func2():
    return [
        func(single_window["y"], single_window["predictions"])
        for single_window in pd.concat([y, predictions], axis="columns").rolling(
            window=window, step=window
        )
    ]


def func3():
    return [
        func(y[i : i + window], predictions[i : i + window])
        for i in range(0, len(y) - 1, window)
    ]


def func4():
    return [
        func(*single_window.values.T)
        for single_window in pd.concat([y, predictions], axis="columns").rolling(
            window=window, step=window
        )
    ]


res1 = timeingit(func1)
res2 = timeingit(func2)
# res3 = timeingit(func3)
res4 = timeingit(func4)

assert res1 == res2 == res4
# assert res1 == res2 == res3 == res4
