from typing import Any, Callable, Iterable, List, Union


def map_newdict_on_olddict(
    old_dict: dict, new_dict: dict, exclude: List[str] = []
) -> dict:
    merged_dict = old_dict.copy()
    for key_, value_ in new_dict.items():
        if key_ in old_dict and value_ is not None and key_ not in exclude:
            merged_dict[key_] = value_

    return merged_dict


def group_by_categories(
    flat_list: List[dict[str, Any]], categories: List[str]
) -> dict[str, "Metric"]:
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


def strip_builtin_functions(dict_to_strip: dict) -> dict:
    return {key_: value_ for key_, value_ in dict_to_strip.items() if key_[:2] != "__"}
