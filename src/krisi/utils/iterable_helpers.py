from typing import Any, List, Optional


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
