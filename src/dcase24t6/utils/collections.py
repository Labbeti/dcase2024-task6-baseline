#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Mapping, Sequence, TypeVar

T = TypeVar("T")


def dict_list_to_list_dict(dic: Mapping[str, Sequence[T]]) -> list[dict[str, T]]:
    """Convert dict of lists with same sizes to list of dicts.

    Example 1
    ----------
    ```
    >>> dic = {"a": [1, 2], "b": [3, 4]}
    >>> dict_list_to_list_dict(dic)
    ... [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    ```
    """
    length = len(next(iter(dic.values())))
    result = [{k: v[i] for k, v in dic.items()} for i in range(length)]
    return result
