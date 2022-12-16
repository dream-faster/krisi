import os
from collections.abc import Iterable
from typing import Any, List, Union

import numpy as np


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


def group_by_categories(flat_list: List[dict[str, Any]], categories: List[str]) -> dict:
    category_groups = dict()
    for category in categories:
        category_groups[category] = list(
            filter(
                lambda x: x["category"].value == category
                if hasattr(x, "category")
                else False,
                flat_list,
            )
        )
    category_groups[None] = list(
        filter(
            lambda x: x["category"].value == None if hasattr(x, "category") else False,
            flat_list,
        )
    )
    return category_groups


def print_summary(obj: "ScoreCard", categories: List[str], repr: bool = False) -> str:
    divider_len: int = get_term_size()
    full_str = ""
    full_str += f"\n\nResult of {obj.model_name if repr else bold(obj.model_name)} on {obj.dataset_name if repr else bold(obj.dataset_name)} tested on {obj.sample_type.value if repr else bold(obj.sample_type.value)}"
    full_str += f"\n{'―'*divider_len}"

    category_groups = group_by_categories(list(vars(obj).values()), categories)

    full_str += f"\n{'name':^30s}| {'result':^15s}| {'hyperparams':^15s}"

    for category, metrics in category_groups.items():
        full_str += f"\n\n\n{category if category is not None else 'Unknown':>15s}"
        full_str += f"\n{'.'*divider_len:>15s}"
        for metric in metrics:
            full_str += f"\n{str(metric):>15s}"

    return full_str


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

    return f"{obj.name:>30s}: {handle_iterable_printing(obj.metric_result):^15.5s}{hyperparams:>15s}"
