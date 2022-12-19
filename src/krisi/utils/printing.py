import os
from collections.abc import Iterable
from typing import Any, List, Optional, Union

import numpy as np
from rich import box, print
from rich.console import Group
from rich.layout import Layout
from rich.padding import Padding
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table
from rich.text import Text

from krisi.utils.iterable_helpers import group_by_categories


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        # Layout(name="header", size=3),
        Layout(name="main"),
        # Layout(name="footer", size=3),
    )
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


def get_summary(
    obj: "ScoreCard", categories: List[str], repr: bool = False, with_info: bool = False
) -> Union[Panel, Layout]:

    title = f"Result of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"

    layout = make_layout()

    table_header = f"\n{'name':^30s}| {'result':^15s}| {'hyperparams':^15s}"
    # layout["header"].update(Panel(Text.from_ansi(title, justify="center")))

    category_groups = group_by_categories(list(vars(obj).values()), categories)

    category_layouts: List[Union[Table, Panel]] = []
    for category, metrics in category_groups.items():
        if metrics is None or len(metrics) < 1:
            continue
        category_title = f"{category if category is not None else 'Unknown':>15s}"

        category_layout = Layout(name=category_title, minimum_size=3)

        category_layout.split_row(
            Layout(
                Panel(category_title, padding=1, box=box.MINIMAL),
                ratio=1,
                minimum_size=3,
            ),
            Layout(name="metrics", ratio=5, minimum_size=3),
        )

        table = Table(
            title="",
            # show_edge=False,
            show_footer=False,
            show_header=False,
            expand=True,
            box=box.ASCII2,
        )

        table.add_column(
            "Metric Name", justify="right", style="cyan", width=1, no_wrap=False
        )
        table.add_column("Result", style="magenta", width=2)
        table.add_column("Hyperparameters", style="green", width=3)
        if with_info:
            table.add_column("Info", width=3)

        for metric in metrics:
            metric_summarized = [
                f"{metric.name if metric.full_name is None else metric.full_name} ({metric.name})",
                Pretty(metric.result)
                if not isinstance(metric.result, Iterable)
                else Pretty("Result is an Iterable"),
                Pretty(metric.hyperparameters),
                Pretty(metric.info),
            ]
            metric_summarized = (
                metric_summarized if with_info else metric_summarized[:-1]
            )
            table.add_row(*metric_summarized)

        category_layout["metrics"].update(Panel(table, padding=0, box=box.MINIMAL))

        category_layouts.append(category_layout)

    layout["main"].split_column(*category_layouts)

    return Panel(layout, title=title, padding=3)


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
    if obj.hyperparameters is not None:
        hyperparams += "".join(
            [f"{key} - {value}" for key, value in obj.hyperparameters.items()]
        )

    return f"{obj.full_name if isinstance(obj.full_name, str) else obj.name:>30s} ({obj.name}): {handle_iterable_printing(obj.result):^15.5s}{hyperparams:>15s}"
