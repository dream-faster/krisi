import os
from collections.abc import Iterable
from typing import Any, List, Union

import numpy as np
from rich import print
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.pretty import Pretty
from rich.table import Table

from krisi.utils.iterable_helpers import group_by_categories


def make_layout() -> Layout:
    """Define the layout."""
    layout = Layout(name="root")

    layout.split(
        Layout(name="header", size=3),
        Layout(name="main"),
        Layout(name="footer", size=3),
    )
    # layout["main"].split_row(
    #     Layout(name="side"),
    #     Layout(name="body", ratio=2, minimum_size=20),
    # )
    # layout["side"].split(Layout(name="box1"), Layout(name="box2"))
    return layout


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


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


def get_summary(obj: "ScoreCard", categories: List[str], repr: bool = False) -> Layout:

    title = f"Result of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"

    layout = make_layout()
    layout["header"].update(Panel(title))

    table_header = f"\n{'name':^30s}| {'result':^15s}| {'hyperparams':^15s}"

    category_groups = group_by_categories(list(vars(obj).values()), categories)

    category_layouts: List[Union[Table, Panel]] = []
    for category, metrics in category_groups.items():
        category_title = f"{category if category is not None else 'Unknown':>15s}"

        category_layout = Layout(name=category_title, size=3)

        category_layout.split_row(
            Layout(category_title, ratio=1), Layout(name="metrics", ratio=5)
        )

        category_layout["metrics"].update(
            Group(
                *[
                    Panel(
                        Group(
                            metric.name,
                            Pretty(
                                iterative_length(metric.result),
                                max_length=10,
                                max_depth=2,
                            )
                            if isinstance(metric.result, Iterable)
                            else Pretty(metric.result),
                            Pretty(metric.hyperparameters),
                        ),
                        height=300,
                    )
                    for metric in metrics
                ]
            )
        )

        category_layouts.append(category_layout)

    layout["main"].split_column(*category_layouts)

    return layout


def handle_iterable_printing(obj: Any) -> str:
    if isinstance(obj, (str, float, int)):
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

    return f"{obj.name:>30s}: {handle_iterable_printing(obj.result):^15.5s}{hyperparams:>15s}"
