from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Callable, Optional, Any, Literal, Union
from dash import dcc


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
    title: Optional[str] = ""


class DisplayModes(Enum):
    interactive = "interactive"
    pdf = "pdf"
    direct = "direct"


save_path = "reporting/"
