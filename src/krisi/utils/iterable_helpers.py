from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    TypeVar,
    Union,
)

import numpy as np
import pandas as pd

from krisi.evaluate.type import Predictions, Targets

if TYPE_CHECKING:
    from krisi.evaluate.metric import Metric
    from krisi.report.type import InteractiveFigure


def map_newdict_on_olddict(
    old_dict: dict, new_dict: dict, exclude: List[str] = []
) -> dict:
    merged_dict = old_dict.copy()
    for key_, value_ in new_dict.items():
        if key_ in old_dict and value_ is not None and key_ not in exclude:
            merged_dict[key_] = value_

    return merged_dict


def group_by_categories(
    flat_list: List[Union["Metric", "InteractiveFigure"]], categories: List[str]
) -> Dict[str, Union["Metric", "InteractiveFigure"]]:
    category_groups = dict()
    for category in categories:
        category_groups[category] = list(
            filter(
                lambda x: x.category.value == category
                if (hasattr(x, "category") and x.category is not None)
                else False,
                flat_list,
            )
        )
    category_groups[None] = list(
        filter(
            lambda x: x.category is None if hasattr(x, "category") else False,
            flat_list,
        )
    )
    return category_groups


def type_converter(type_to_ensure: Any) -> Callable:
    def ensure_format(
        obj_s: Union[List[Any], Any]
    ) -> Union[List[type_to_ensure], type_to_ensure]:
        if isinstance(obj_s, Iterable) and not isinstance(obj_s, str):
            return [
                type_to_ensure(el) if not isinstance(el, type_to_ensure) else el
                for el in obj_s
            ]
        else:
            if isinstance(obj_s, type_to_ensure):
                return obj_s
            else:
                return type_to_ensure(obj_s)

    return ensure_format


def string_to_id(s: str) -> str:
    s = "".join(filter(lambda c: str.isidentifier(c) or str.isdecimal(c), s))
    return s


def isiterable(obj: Any) -> bool:
    if isinstance(obj, Iterable) and not isinstance(obj, str):
        return True
    else:
        return False


def isnumber(obj: Any) -> bool:
    if isinstance(obj, (int, float, np.number)):
        return True
    else:
        return False


def check_iterable_with_number(iterable: Any):
    if isinstance(iterable, Exception) or iterable is None:
        return False

    if isiterable(iterable) and len(iterable) > 0:
        if isnumber(iterable[0]):
            return True
    return False


def strip_builtin_functions(dict_to_strip: dict) -> dict:
    return {key_: value_ for key_, value_ in dict_to_strip.items() if key_[:2] != "__"}


def __flatten(xs: Iterable[Any]) -> Any:
    for x in xs:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            yield from flatten(x)
        else:
            yield x


def flatten(xs: Iterable[Any]) -> Iterable[Any]:
    return list(__flatten(xs))


def remove_nans(iter):
    if isinstance(iter, List):
        return [el for el in iter if el is not None]
    else:
        return {key: el for key, el in iter.items() if el is not None}


def calculate_nans(ds: Union[Targets, Predictions]) -> int:
    return ds.isna().sum() if isinstance(ds, pd.Series) else sum(np.isnan(ds))


T = TypeVar("T")


def wrap_in_list(input: Union[T, List[T]]) -> List[T]:
    return input if isinstance(input, List) else [input]


def replace_if_none(input: Any, replacement: Any) -> Any:
    return replacement if input is None else input


def is_int(s: Any):
    if isinstance(s, str) and "." in s:
        return False
    try:
        int(s)
    except ValueError:
        return False
    else:
        return True


def del_dict_keys(d: dict, keys: Union[str, List[str]]) -> dict:
    keys = wrap_in_list(keys)
    return {k: v for k, v in d.items() if k not in keys}


def empty_if_none(metrics: Optional[List]) -> List:
    return metrics if metrics is not None else []


def filter_none(list: List) -> List:
    return [el for el in list if el is not None]
