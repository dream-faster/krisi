import numbers
import os
from collections.abc import Iterable
from copy import deepcopy
from typing import TYPE_CHECKING, Any, List, Optional, Union

import numpy as np
import pandas as pd
import plotext as plx
from rich import box
from rich.console import Console, Group
from rich.layout import Layout
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from krisi.evaluate.type import MetricCategories, SaveModes
from krisi.utils.console_plot import plotextMixin
from krisi.utils.iterable_helpers import group_by_categories, isiterable

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric
    from krisi.evaluate.scorecard import ScoreCard


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
    plx.theme("dark")

    return plx.build()


def __display_result(metric: "Metric") -> Union[Pretty, plotextMixin]:
    if metric.result is None:
        result = deepcopy(metric.result_rolling)
    else:
        result = deepcopy(metric.result)

    if isinstance(result, Exception):
        result = str(result)
    elif isinstance(result, float):
        result = round(result, 3)

    if isiterable(result):
        if isinstance(result, (pd.Series, pd.DataFrame)):
            return Pretty(result)
        elif isiterable(result[0]):
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
    title: str, metrics: List["Metric"], with_info: bool, show_header: bool = False
) -> Table:
    table = Table(
        title=title,
        # show_edge=False,
        show_footer=False,
        show_lines=True,
        show_header=show_header,
        expand=True,
        box=box.ROUNDED,
    )

    table.add_column(
        "Metric Name", justify="right", style="cyan", width=1, no_wrap=False
    )
    table.add_column("Result", style="magenta", width=5)
    table.add_column("Parameters", style="green", width=1)
    if with_info:
        table.add_column("Info", width=3)

    for metric in metrics:
        if metric.result is None and metric.result_rolling is None:
            continue
        metric_summarized = __create_metric(metric, with_info)
        table.add_row(*metric_summarized)

    return table


def __metrics_empty_in_category(metrics: List["Metric"]) -> bool:
    return (
        metrics is None
        or len(metrics) < 1
        or (
            all([metric.result is None for metric in metrics])
            and all([metric.result_rolling is None for metric in metrics])
        )
    )


def get_summary(
    obj: "ScoreCard", categories: List[str], repr: bool = True, with_info: bool = False
) -> Union[Panel, Layout]:

    category_groups = group_by_categories(list(vars(obj).values()), categories)

    metric_tables = Group(
        *[
            __create_metric_table(
                f"{category if category is not None else 'Unknown Category':>15s}",
                metrics,
                with_info,
                show_header=True if index == 0 else False,
            )
            for index, (category, metrics) in enumerate(category_groups.items())
            if not __metrics_empty_in_category(metrics)
        ],
    )

    title = f"Result of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    return Panel(metric_tables, title=title, padding=1, box=box.HEAVY_EDGE, expand=True)


def get_minimal_summary(obj: "ScoreCard") -> str:
    return f"\n".join(
        [
            f"{metric.name:>40s} - {metric.result:<15.5}"
            for metric in obj.get_all_metrics()
            if isinstance(metric.result, (float, int))
        ]
    )


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
    if obj.result is None and obj.result_rolling is not None:
        result_ = (
            "[" + ", ".join([f"{result:<0.5}" for result in obj.result_rolling]) + "]"
        )
    else:
        result_ = f"{obj.result:<15.5}"

    return f"{obj.name:>40s} ({obj.key}): {result_}{hyperparams:>15s}"


def save_object(obj: "ScoreCard", path: str) -> None:
    import dill

    with open(f"{path}/scorecard.pickle", "wb") as file:
        dill.dump(obj, file)


def save_console(
    obj: "ScoreCard",
    path: str,
    with_info: bool,
    save_modes: List[Union[SaveModes, str]],
) -> None:

    summary = get_summary(
        obj,
        repr=True,
        categories=[el.value for el in MetricCategories],
        with_info=with_info,
    )

    console = Console(record=True, width=120)
    with console.capture() as capture:
        console.print(summary)

    if SaveModes.text in save_modes or SaveModes.text.value in save_modes:
        console.save_text(f"{path}/console.txt", clear=False)
    if SaveModes.html in save_modes or SaveModes.html.value in save_modes:
        console.save_html(f"{path}/console.html", clear=False)
    if SaveModes.svg in save_modes or SaveModes.svg.value in save_modes:
        console.save_svg(f"{path}/console.svg", title="save_table_svg.py", clear=False)

    # console.clear()


def save_minimal_summary(obj: "ScoreCard", path: str) -> None:
    text_summary = get_minimal_summary(obj)

    with open(f"{path}/minimal.txt", "w") as f:
        f.write(text_summary)
