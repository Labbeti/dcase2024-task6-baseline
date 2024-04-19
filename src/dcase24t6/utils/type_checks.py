#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Iterable, TypeGuard


def is_list_int(x: Any) -> TypeGuard[list[int]]:
    return isinstance(x, list) and all(isinstance(xi, int) for xi in x)


def is_list_str(x: Any) -> TypeGuard[list[str]]:
    return isinstance(x, list) and all(isinstance(xi, str) for xi in x)


def is_iterable_str(x: Any, *, accept_str: bool) -> TypeGuard[Iterable[str]]:
    return (accept_str and isinstance(x, str)) or (
        not isinstance(x, str)
        and isinstance(x, Iterable)
        and all(isinstance(xi, str) for xi in x)
    )
