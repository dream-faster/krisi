import numbers
import os
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import plotext as plx
from rich import box, print
from rich.console import Group
from rich.layout import Layout
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

from krisi.utils.console_plot import plotextMixin
from krisi.utils.iterable_helpers import group_by_categories, isiterable

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric
    from krisi.evaluate.scorecard import ScoreCard


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="main")

    # layout.split(
    #     # Layout(name="header", size=3),
    #     Layout(name="main"),
    #     # Layout(name="footer", size=3),
    # )
    return layout


def bold(text: str, rich: bool = True) -> str:
    return f"[bold]{text}[/bold]" if rich else f"\033[1m{text}\033[0m"


def get_term_size() -> int:
    term_size = os.get_terminal_size()
    return term_size.columns


def iterative_length(obj: Iterable) -> List[int]:
    object_shape = []
    num_obj = 0
    for el in obj:
        if isinstance(el, Iterable) and not isinstance(el, str):
            object_shape.append(iterative_length(el))
        else:
            num_obj += 1
    if num_obj > 0:
        object_shape.append(num_obj)
    return object_shape


def line_plot_rolling(data, width, height, title):
    plx.clf()

    plx.plot(data, marker="hd")
    plx.plotsize(width, 10)
    # plx.xaxes(1, 0)
    # plx.yaxes(1, 0)
    # plx.title(title)
    plx.theme("dark")
    # plx.ylim(-1, 1)

    return plx.build()


def __display_result(metric: "Metric") -> Union[Pretty, plotextMixin]:
    result = deepcopy(metric.result)
    if isinstance(result, Exception):
        result = str(result)
    elif isinstance(result, float):
        result = round(result, 3)

    if isiterable(result):
        if isiterable(result[0]):
            return Pretty("Result is a complex Iterable")
        else:
            # Create a Console Plot
            return plotextMixin(result, line_plot_rolling, title=metric.name)
    else:
        return Pretty(result, max_depth=2, max_length=3)


def __create_metric(metric: "Metric", with_info: bool) -> List[str]:
    metric_summarized = [
        f"{metric.name} ({metric.key})",
        __display_result(metric),
        Pretty(metric.parameters),
        Pretty(metric.info),
    ]
    metric_summarized = metric_summarized if with_info else metric_summarized[:-1]

    return metric_summarized


def __create_metric_table(
    title: str, metrics: List["Metric"], with_info: bool
) -> Layout:
    table = Table(
        title=title,
        # show_edge=False,
        show_footer=False,
        show_header=False,
        expand=True,
        box=box.ASCII2,
    )

    table.add_column(
        "Metric Name", justify="right", style="cyan", width=1, no_wrap=False
    )
    table.add_column("Result", style="magenta", width=5)
    table.add_column("parameters", style="green", width=1)
    if with_info:
        table.add_column("Info", width=3)

    for metric in metrics:
        if metric.result is None:
            continue
        metric_summarized = __create_metric(metric, with_info)
        table.add_row(*metric_summarized)

    return Layout(table)


def __metrics_empty_in_category(metrics: List["Metric"]) -> bool:
    return (
        metrics is None
        or len(metrics) < 1
        or all([metric.result is None for metric in metrics])
    )


def get_summary(
    obj: "ScoreCard", categories: List[str], repr: bool = False, with_info: bool = False
) -> Union[Panel, Layout]:

    layout = make_layout()

    category_groups = group_by_categories(list(vars(obj).values()), categories)

    metric_tables = Group(
        *[
            __create_metric_table(
                f"{category if category is not None else 'Unknown':>15s}",
                metrics,
                with_info,
            )
            for category, metrics in category_groups.items()
            if not __metrics_empty_in_category(metrics)
        ]
    )

    title = f"Result of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    return Panel(metric_tables, title=title, padding=3, box=box.ASCII2)


def handle_iterable_printing(obj: Any) -> Optional[str]:
    if obj is None:
        return "None"
    elif isinstance(obj, (str, float, int)):
        return str(obj)
    elif isinstance(obj, str):
        return obj
    elif isinstance(obj, np.ndarray):
        return f"List: {str(obj.shape)}"
    else:
        return f"List: {str(len(obj))}"


def print_metric(obj: "Metric", repr: bool = False) -> str:
    hyperparams = ""
    if obj.parameters is not None:
        hyperparams += "".join(
            [f"{key} - {value}" for key, value in obj.parameters.items()]
        )

    return f"{obj.name:>30s} ({obj.key}): {obj.result:^15.5s}{hyperparams:>15s}"
