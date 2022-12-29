from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, List, Literal, Optional, Union

import pandas as pd
import plotly.graph_objects as go
from dash import dcc

from krisi.evaluate.type import MetricResult

PlotFunction = Callable[
    [List[MetricResult], str, Optional[float]],
    go.Figure,
]


@dataclass
class PlotlyInput:
    type: Literal[dcc.Dropdown, dcc.Input, dcc.Checklist, dcc.Slider, dcc.RangeSlider]
    id: str
    value_name: str
    default_value: Union[str, float, int]
    options: Optional[List[Union[str, float, int]]]


@dataclass
class InteractiveFigure:
    id: str
    get_figure: Callable
    inputs: List[PlotlyInput] = field(default_factory=list)
    global_input_ids: List[str] = field(default_factory=list)
    title: Optional[str] = ""


class DisplayModes(Enum):
    interactive = "interactive"
    pdf = "pdf"
    direct = "direct"


save_path = "report/"


def plotly_interactive(
    plot_function: PlotFunction,
    data_source: Union[pd.DataFrame, pd.Series],
    *args,
    **kwargs
) -> Callable:
    default_args = args
    default_kwargs = kwargs

    def wrapper(*args, **kwargs) -> go.Figure:
        width = kwargs.pop("width", None)
        title = kwargs.pop("title", "")

        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        fig = plot_function(data_source, *args, **kwargs)

        fig.update_layout(width=width)
        if title is not None:
            fig.update_layout(title=title)
        return fig

    return wrapper
