#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Literal, Mapping, Sequence, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def dict_list_to_list_dict(dic: Mapping[T, Sequence[U]]) -> list[dict[T, U]]:
    """Convert dict of lists with same sizes to list of dicts.

    Example 1
    ----------
    ```
    >>> dic = {"a": [1, 2], "b": [3, 4]}
    >>> dict_list_to_list_dict(dic)
    ... [{"a": 1, "b": 3}, {"a": 2, "b": 4}]
    ```
    """
    if len(dic) == 0:
        raise ValueError(f"Invalid argument {dict=}. (expected non-empty dict)")

    length = len(next(iter(dic.values())))
    result = [{k: v[i] for k, v in dic.items()} for i in range(length)]
    return result


def unflat_dict_of_dict(
    dic: Mapping[str, Any],
    sep: str = ".",
    duplicate_mode: Literal["error", "override"] = "error",
) -> dict[str, Any]:
    """Unflat a dictionary.

    Example 1
    ----------
    ```
    >>> dic = {
        "a.a": 1,
        "b.a": 2,
        "b.b": 3,
        "c": 4,
    }
    >>> unflat_dict(dic)
    ... {"a": {"a": 1}, "b": {"a": 2, "b": 3}, "c": 4}
    ```
    """
    DUPLICATE_MODES = ("error", "override")
    if duplicate_mode not in DUPLICATE_MODES:
        raise ValueError(
            f"Invalid argument {duplicate_mode=}. (expected one of {DUPLICATE_MODES})"
        )

    output = {}
    for k, v in dic.items():
        if sep not in k:
            if k not in output or duplicate_mode == "override":
                output[k] = v
            else:
                raise ValueError(
                    f"Invalid keys in dict argument. (found key {k} and at least one another key starting with {k}{sep} in {tuple(dic.keys())})"
                )
        else:
            idx = k.index(sep)
            k, kk = k[:idx], k[idx + 1 :]
            if k not in output:
                output[k] = {kk: v}
            elif isinstance(output[k], Mapping):
                output[k][kk] = v
            elif duplicate_mode == "override":
                output[k] = {kk: v}
            else:
                raise ValueError(
                    f"Invalid keys in dict argument. (found keys {k} and {k}{sep}{kk} in {tuple(dic.keys())})"
                )

    output = {
        k: (
            unflat_dict_of_dict(v, sep, duplicate_mode) if isinstance(v, Mapping) else v
        )
        for k, v in output.items()
    }
    return output
