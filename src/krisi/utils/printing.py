import os
from copy import deepcopy
from typing import TYPE_CHECKING, List, Union

import numpy as np
import pandas as pd
from rich import box
from rich.pretty import Pretty
from rich.table import Table

from krisi.evaluate.type import Predictions, Targets
from krisi.report.library.console.diagrams import (
    distribution_plot,
    histogram_plot,
    line_plot,
)
from krisi.utils.console_plot import plotextMixin
from krisi.utils.iterable_helpers import calculate_nans, isiterable

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric


def bold(text: str, rich: bool = True) -> str:
    return f"[bold]{text}[/bold]" if rich else f"\033[1m{text}\033[0m"


def get_term_size() -> int:
    term_size = os.get_terminal_size()
    return term_size.columns


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
            return plotextMixin(result, line_plot, title=metric.name)
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


def create_metric_table(
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


def metrics_empty_in_category(metrics: List["Metric"]) -> bool:
    return (
        metrics is None
        or len(metrics) < 1
        or (
            all([metric.result is None for metric in metrics])
            and all([metric.result_rolling is None for metric in metrics])
        )
    )


def create_y_pred_table(classification: bool, y: Targets, preds: Predictions) -> Table:
    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    if isinstance(preds, np.ndarray):
        preds = pd.Series(preds)

    table = Table(
        title="Targets and Predictions Analysis",
        # show_edge=False,
        show_footer=False,
        show_lines=True,
        show_header=True,
        expand=True,
        box=box.ROUNDED,
    )

    vizualisation_name = "Category Imbalance" if classification else "Histogram"
    vizualisation_func = distribution_plot if classification else histogram_plot
    table.add_column(
        "Series Type", justify="right", style="cyan", width=2, no_wrap=False
    )
    table.add_column(
        vizualisation_name, justify="left", style="cyan", width=10, no_wrap=False
    )
    table.add_column(
        "Types",
        justify="right",
        style="cyan",
        width=1,
        no_wrap=False,
    )
    table.add_column(
        "Indicies",
        justify="right",
        style="cyan",
        width=1,
        no_wrap=False,
    )

    table.add_row(
        "Targets",
        plotextMixin(y, vizualisation_func, title=vizualisation_name),
        f"NaNs: {str(calculate_nans(y))}\ndtype: {str(y.dtype)}",
        f"Start: {str(y.index[0])}\nEnd: {str(y.index[-1])}",
    )
    table.add_row(
        "Predictions",
        plotextMixin(preds, vizualisation_func, title=vizualisation_name),
        f"NaNs: {str(calculate_nans(preds))}\ndtype: {str(preds.dtype)}",
        f"Start: {str(preds.index[0])}\nEnd: {str(preds.index[-1])}",
    )
    return table
