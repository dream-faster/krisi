from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from typing_extensions import Literal

from krisi.utils.enums import ParsableEnum

if TYPE_CHECKING:
    import plotly.graph_objects as go

from krisi.evaluate.type import MetricCategories, MetricResult

PlotFunction = Callable[
    [List[MetricResult], str, Optional[float]],
    "go.Figure",
]
PlotDefinition = Tuple[PlotFunction, Optional[Dict[str, Any]]]


@dataclass
class PlotlyInput:
    type: Literal[
        "dcc.Dropdown", "dcc.Input", "dcc.Checklist", "dcc.Slider", "dcc.RangeSlider"
    ]
    id: str
    value_name: str
    default_value: Union[str, float, int]
    options: Optional[List[Union[str, float, int]]] = None


@dataclass
class InteractiveFigure:
    id: str
    get_figure: Callable
    inputs: List[PlotlyInput] = field(default_factory=list)
    global_input_ids: List[str] = field(default_factory=list)
    title: Optional[str] = ""
    width: Optional[float] = 900.0
    height: Optional[float] = 600.0
    category: Optional[MetricCategories] = None
    plot_args: Dict[str, Any] = field(default_factory=dict)


class DisplayModes(ParsableEnum):
    interactive = "interactive"
    pdf = "pdf"
    direct = "direct"
    direct_save = "direct_save"
    direct_one_subplot = "direct_one_subplot"


def plotly_interactive(
    plot_function: PlotFunction,
    data_source: Union[pd.DataFrame, pd.Series],
    *args,
    **kwargs,
) -> Callable:
    # default_args = args
    default_kwargs = kwargs

    def wrapper(*args, **kwargs) -> "go.Figure":
        for key, value in default_kwargs.items():
            if key not in kwargs:
                kwargs[key] = value

        width = kwargs.pop("width", None)
        height = kwargs.pop("height", None)
        title = kwargs.get("title", None)

        fig = plot_function(data_source, *args, **kwargs)
        fig.update_layout(autosize=False, width=width, height=height)
        if title is not None:
            fig.update_layout(title=title)
        return fig

    return wrapper
