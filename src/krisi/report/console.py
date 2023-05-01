from typing import TYPE_CHECKING, Iterable, List, Union

import pandas as pd
from rich import box
from rich.console import Group
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table

from krisi.utils.iterable_helpers import group_by_categories
from krisi.utils.printing import (
    create_metric_table,
    create_y_pred_table,
    metrics_empty_in_category,
)

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric
    from krisi.evaluate.scorecard import ScoreCard

from krisi.utils.printing import bold


def print_metric(obj: "Metric", repr: bool = False) -> str:
    if repr:
        return f"{obj.result} | {obj.name}"
    hyperparams = ""
    if obj.parameters is not None:
        hyperparams += "".join(
            [f"{key} - {value}" for key, value in obj.parameters.items()]
        )
    if (
        obj.result is None
        and obj.result_rolling is not None
        and isinstance(obj.result_rolling, Iterable)
    ):
        result_ = (
            "[" + ", ".join([f"{result:<0.5}" for result in obj.result_rolling]) + "]"
        )
    elif obj.result is None:
        result_ = ""
    elif isinstance(obj.result, Iterable) and not isinstance(obj.result, str):
        result_ = obj.result
    else:
        result_ = f"{obj.result:<15.5}"

    return f"{obj.name:>40s} ({obj.key}): {result_}{hyperparams:>15s}"


def get_minimal_summary(obj: "ScoreCard", dataframe: bool) -> Union[pd.DataFrame, str]:
    if dataframe:
        all_metrics = [
            metric
            for metric in obj.get_all_metrics()
            if isinstance(metric.result, (float, int))
        ]
        return pd.Series(
            [metric.result for metric in all_metrics],
            name=obj.metadata.model_name,
            index=[f"{metric.name:>40s}" for metric in all_metrics],
        ).to_frame()

    return "\n".join(
        [
            f"{metric.name:>40s} - {metric.result:<15.5}"
            for metric in obj.get_all_metrics()
            if isinstance(metric.result, (float, int))
        ]
    )


def get_large_metric_summary(obj: "ScoreCard", title: str) -> Table:
    return create_metric_table(
        title=title,
        metrics=obj.get_all_metrics(),
        with_info=False,
        with_parameters=False,
    )


def get_summary(
    obj: "ScoreCard",
    categories: List[str],
    repr: bool = True,
    with_info: bool = False,
    input_analysis: bool = True,
) -> Union[Panel, Layout]:
    category_groups = group_by_categories(list(vars(obj).values()), categories)
    input_analysis_table = (
        [create_y_pred_table(obj.classification, obj.y, obj.predictions)]
        if input_analysis
        else []
    )

    metric_tables = Group(
        *input_analysis_table
        + [
            create_metric_table(
                f"{category if category is not None else 'Unknown Category':>15s}",
                metrics,
                with_info,
                show_header=True if index == 0 else False,
            )
            for index, (category, metrics) in enumerate(category_groups.items())
            if not metrics_empty_in_category(metrics)
        ],
    )

    title = f"Result of {obj.metadata.model_name if repr else bold(obj.metadata.model_name)} on {obj.metadata.dataset_name if repr else bold(obj.metadata.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    return Panel(metric_tables, title=title, padding=1, box=box.HEAVY_EDGE, expand=True)
