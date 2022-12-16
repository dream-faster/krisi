from typing import List, Optional


def map_newdict_on_olddict(
    old_dict: dict, new_dict: dict, exclude: List[str] = []
) -> dict:
    merged_dict = old_dict.copy()
    for key_, value_ in new_dict.items():
        if key_ in old_dict and value_ is not None and key_ not in exclude:
            merged_dict[key_] = value_

    return merged_dict
