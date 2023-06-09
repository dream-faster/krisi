import os
from copy import deepcopy
from typing import TYPE_CHECKING, List, Tuple, Union

import numpy as np
import pandas as pd
from rich import box
from rich.pretty import Pretty
from rich.table import Table

from krisi.evaluate.type import Predictions, Targets
from krisi.report.library.console.diagrams import (
    callibration_plot,
    distribution_plot,
    histogram_plot,
    line_plot,
)
from krisi.utils.console_plot import plotextMixin
from krisi.utils.iterable_helpers import calculate_nans, filter_none, isiterable

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric


def bold(text: str, rich: bool = True) -> str:
    return f"[bold]{text}[/bold]" if rich else f"\033[1m{text}\033[0m"


def get_term_size() -> int:
    term_size = os.get_terminal_size()
    return term_size.columns


def __convert_result(
    result, title: str, return_string: bool = False
) -> Union[str, Pretty, plotextMixin]:
    result = deepcopy(result)
    if result is None:
        return ""
    if isinstance(result, float):
        result = round(result, 3)

        if return_string:
            return str(result)
    if isinstance(result, str) and return_string:
        return result
    elif isinstance(result, Exception):
        result = str(result)

    if isinstance(result, dict):
        if len(result) > 0:
            return "\n".join([f"{key}:\n {value}" for key, value in result.items()])

        else:
            return ""
    if isinstance(result, Tuple):

        def string_from_ds(ds: pd.Series, title: str) -> str:
            return "\n".join(
                [
                    f"({key}\n {__convert_result(value, title, True)})"
                    for key, value in ds.items()
                ]
            )

        result = filter_none(list(result))
        if len(result) == 1:
            return __convert_result(result[0], title, return_string=True)

        return "\n".join(
            [
                f"{__convert_result(res, title, return_string=True)}"
                if not isinstance(res, pd.Series)
                else string_from_ds(res, title)
                for res in result
                if res is not None
            ]
        )

    if isiterable(result):
        if isinstance(result, pd.DataFrame):
            if "probs" in result:
                return plotextMixin(result, callibration_plot, title=title)
            else:
                return Pretty(result)
        if isinstance(result, pd.Series):
            if len(result) > 10:
                return plotextMixin(result.tolist(), line_plot, title=title)
            else:
                return Pretty(result)
        elif isiterable(result[0]):
            return Pretty("Result is a complex Iterable")
        else:
            # Create a Console Plot
            return plotextMixin(result, line_plot, title=title)
    else:
        return Pretty(result, max_depth=2, max_length=3)


def __display_result(metric: "Metric") -> List[Union[str, Pretty, plotextMixin]]:
    return [
        __convert_result(res, title=metric.name)
        for res in [
            (metric.result, metric.comparison_result),
            metric.result_rolling,
            metric.rolling_properties,
        ]
    ]


def __create_metric(
    metric: "Metric", with_info: bool, with_parameters: bool, with_diagnostics: bool
) -> List[str]:
    results_as_strings = __display_result(metric)
    if isinstance(results_as_strings[0], plotextMixin):
        results_as_strings = ["", *results_as_strings[:-1]]
    metric_summarized = (
        [
            f"{metric.name} ({metric.key})",
            *results_as_strings,
        ]
        + ([__convert_result(metric.parameters, title="")] if with_parameters else [])
        + ([Pretty(metric.diagnostics)] if with_diagnostics else [])
        + ([Pretty(metric.info)] if with_info else [])
    )

    return metric_summarized


def create_metric_table(
    title: str,
    metrics: List["Metric"],
    with_info: bool,
    with_parameters: bool,
    with_diagnostics: bool,
    show_header: bool = False,
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
        "Metric Name", justify="right", style="cyan", width=2, no_wrap=False
    )
    table.add_column("Result", style="magenta", width=1)
    table.add_column("Rolling", width=8)
    table.add_column("Rolling Props", width=1)
    if with_parameters:
        table.add_column("Parameters", style="green", width=1)
    if with_info:
        table.add_column("Info", width=3)
    if with_diagnostics:
        table.add_column("Diagnostics", width=3)

    for metric in metrics:
        if metric.result is None and metric.result_rolling is None:
            continue
        metric_summarized = __create_metric(
            metric, with_info, with_parameters, with_diagnostics
        )
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
