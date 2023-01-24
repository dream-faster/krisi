import pandas as pd
import plotext as plx


def histogram_plot(data: pd.Series, width: float, height: float, title: str) -> str:
    plx.clf()
    plx.hist(data, 40)
    plx.plotsize(width, 10)
    plx.theme("dark")

    return plx.build()


def line_plot(data: pd.Series, width: float, height: float, title: str) -> str:
    plx.clf()
    plx.plot(data, marker="hd")
    plx.plotsize(width, 10)
    plx.theme("dark")

    return plx.build()


def distribution_plot(data: pd.Series, width: float, height: float, title: str) -> str:
    plx.clf()
    data_ = data.to_frame("values")

    groups = data_.groupby("values").size()
    plx.simple_bar(
        groups.index,
        groups.to_list(),
        width=width,
    )

    return plx.build()
