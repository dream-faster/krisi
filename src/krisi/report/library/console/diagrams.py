import pandas as pd
import plotext as plx

from krisi.evaluate.library.diagrams import calculate_calibration_bins


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


def callibration_plot(data: pd.Series, width: float, height: float, title: str) -> str:
    plx.clf()

    fraction_positives, bins = calculate_calibration_bins(data, pos_label=1, n_bins=8)
    plx.plot(
        bins, fraction_positives, marker="hd", color="red", label="Calibration Curve"
    )
    plx.scatter(
        [0, 0.25, 0.5, 0.75, 1],
        [0, 0.25, 0.5, 0.75, 1],
        label="Perfect Calibration",
        marker="braille",
        color="white",
    )
    plx.plotsize(width, 20)
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
